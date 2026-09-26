"""Train a small GPT with an optional tokenizer-graph penalty on its embedding.

    python experiments/train_gpt.py --condition baseline      --seed 0 --out runs/
    python experiments/train_gpt.py --condition prefix        --seed 0 --out runs/
    python experiments/train_gpt.py --condition prefix:shuffled --seed 0 --out runs/

``--condition`` is ``baseline`` or ``<graph>[:shuffled]`` with ``<graph>`` one of
``contain`` (substring poset), ``prefix`` (prefix trie + space variants) or
``merge``.  The loss is

    cross_entropy + lam * tr(E^T L E) / ||E||_F^2

on the (tied) token embedding ``E``.  ``:shuffled`` uses the same graph with its
vertices randomly relabelled -- identical spectrum and degrees, no meaning --
which separates a structural effect from generic regularisation.

Every ``--eval_every`` steps it appends to ``<out>/<run>/log.jsonl``: validation
loss, the (unpenalised) Dirichlet ratio of ``E`` on every graph against a
length-stratified null, and throughput.  A checkpoint is saved at each eval and
the run resumes from it automatically (Colab disconnects).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenix import alignment  # noqa: E402
from tokenix.graphs import containment_graph, merge_graph, prefix_graph  # noqa: E402
from tokenix.spectral import laplacian  # noqa: E402
from tokenix.torch_penalty import GraphPenalty  # noqa: E402

GRAPHS = {"contain": containment_graph, "prefix": prefix_graph, "merge": merge_graph}
SHUFFLE_SEED = 12345  # fixed relabelling, shared by all seeds of a shuffled condition


# ----------------------------------------------------------------------------- model


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # 50257 padded to a multiple of 64
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0


PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, block_size=64),  # CPU smoke tests
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=512),  # ~30M params
    "base": dict(n_layer=8, n_head=8, n_embd=512, block_size=512),  # ~51M params
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),  # ~124M params
}


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.fc_out = nn.Linear(4 * c.n_embd, c.n_embd)
        self.n_head = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 residual scaling
            if name.endswith(("proj.weight", "fc_out.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln_f(x) @ self.wte.weight.T  # tied output embedding
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


# ----------------------------------------------------------------------------- data


class Batches:
    def __init__(self, path: Path, block: int, batch: int, device: str, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch, self.device = block, batch, device
        self.rng = np.random.default_rng(seed)

    def get(self):
        ix = self.rng.integers(0, len(self.data) - self.block - 1, self.batch)
        x = np.stack([self.data[i : i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.block] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y


# ----------------------------------------------------------------------------- probes


class GraphProbe:
    """Dirichlet ratio of the embedding on each tokenizer graph vs a length-stratified null."""

    def __init__(self, spec, graphs: dict, n_perm: int = 5):
        self.V = len(spec)
        self.L = {k: laplacian(W) for k, W in graphs.items()}
        self.lengths = np.minimum([len(t) for t in spec.tokens], 12)
        self.n_perm = n_perm

    def __call__(self, E: np.ndarray) -> dict:
        X = E[: self.V].astype(np.float64)
        X = X - X.mean(axis=0)
        out = {}
        for k, L in self.L.items():
            t = alignment.permutation_test(lambda M, L=L: alignment.dirichlet_energy(M, L), X,
                                           self.n_perm, np.random.default_rng(0), self.lengths)
            out[k] = t["ratio"]
        return out


# ----------------------------------------------------------------------------- main


def load_spec(tokenizer_json: str | None):
    from tokenix.io import load_hf_tokenizer_json

    if tokenizer_json is None:
        from tokenix.hub import fetch_file

        tokenizer_json = fetch_file("openai-community/gpt2", "tokenizer.json")
    return load_hf_tokenizer_json(tokenizer_json)


@torch.no_grad()
def evaluate(model, val: Batches, iters: int, ctx) -> float:
    model.eval()
    val.rng = np.random.default_rng(1234)  # identical validation batches every eval
    losses = []
    for _ in range(iters):
        x, y = val.get()
        with ctx:
            losses.append(model(x, y).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fineweb_gpt2"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--lam", type=float, default=0.1, help="graph penalty weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="base", choices=PRESETS)
    ap.add_argument("--tokens", type=float, default=5e8, help="training tokens")
    ap.add_argument("--batch", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--tokenizer_json", default=None)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--name", default=None, help="run directory name (default: from condition/lam/seed)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compile_ = args.compile if args.compile is not None else device == "cuda"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    name = args.name or (args.condition.replace(":", "-") + ("" if args.condition == "baseline" else f"_lam{args.lam:g}") + f"_s{args.seed}")
    run = args.out / name
    run.mkdir(parents=True, exist_ok=True)

    # --- tokenizer graphs (penalty + probes)
    spec = load_spec(args.tokenizer_json)
    graphs = {k: f(spec) for k, f in GRAPHS.items()}
    probe = GraphProbe(spec, {"contain": graphs["contain"], "prefix": graphs["prefix"]})
    penalty = None
    if args.condition != "baseline":
        gname, _, mod = args.condition.partition(":")
        if gname not in GRAPHS or mod not in ("", "shuffled"):
            raise SystemExit(f"bad --condition {args.condition!r}")
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(spec)) if mod else None
        penalty = GraphPenalty(graphs[gname], perm).to(device)

    # --- model / optimiser
    cfg = GPTConfig(**PRESETS[args.preset])
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda")
    tokens_per_step = args.batch * args.accum * cfg.block_size
    steps = int(args.tokens // tokens_per_step)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, steps - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    step = 0
    ckpt = run / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed {name} at step {step}")
    else:
        (run / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                     "model": asdict(cfg), "params": n_params,
                                                     "steps": steps, "tokens_per_step": tokens_per_step}, indent=1))
        (run / "log.jsonl").unlink(missing_ok=True)
    fwd = torch.compile(model) if compile_ else model

    train = Batches(args.data / "train.bin", cfg.block_size, args.batch, device, args.seed * 1000 + step)
    val = Batches(args.data / "val.bin", cfg.block_size, args.batch, device, 0)
    print(f"{name}: {n_params / 1e6:.1f}M params, {steps} steps x {tokens_per_step} tokens, device={device}")

    def log(rec):
        with open(run / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0, tok_count = time.time(), 0
    while step <= steps:
        if step % args.eval_every == 0 or step == steps:
            vl = evaluate(fwd, val, args.eval_iters, ctx)
            ratios = probe(model.wte.weight.detach().float().cpu().numpy())
            dt = time.time() - t0
            rec = {"step": step, "tokens": step * tokens_per_step, "val_loss": vl,
                   "dirichlet_contain": ratios["contain"], "dirichlet_prefix": ratios["prefix"],
                   "tok_per_s": tok_count / dt if dt > 0 else None}
            log(rec)
            print(json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "torch_rng": torch.get_rng_state()}, ckpt)
            t0, tok_count = time.time(), 0
            if step == steps:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(args.accum):
            x, y = train.get()
            with ctx:
                loss = fwd(x, y)
            if penalty is not None:
                loss = loss + args.lam * penalty(model.wte.weight)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_count += tokens_per_step

    (run / "DONE").write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
