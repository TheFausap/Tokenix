"""Validation loss broken down by how often the target token occurs in training.

    python experiments/eval_by_frequency.py runs/ --data data/fineweb_gpt2

The average validation loss is dominated by frequent tokens.  A graph penalty
lets a rare token borrow from its tokenizer-graph neighbours, so it may help
rare tokens while costing a little on frequent ones -- a trade the average
hides.  For every finished run this script evaluates the final checkpoint on
the same contiguous validation windows, keeps the per-token loss, and groups
it by the target token's training-set frequency (log10 buckets).

Per-run results are cached in ``<run>/eval_freq.json``; the token counts of
``train.bin`` in ``<data>/train_counts.npy``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_gpt import GPT, GPTConfig  # noqa: E402

EDGES = [0, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf]
LABELS = ["<10", "10–100", "100–1k", "1k–10k", "10k–100k", "100k–1M", "≥1M"]


def train_counts(data: Path, vocab: int) -> np.ndarray:
    path = data / "train_counts.npy"
    if path.exists():
        return np.load(path)
    arr = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    counts = np.zeros(vocab, dtype=np.int64)
    step = 50_000_000
    for i in range(0, len(arr), step):  # chunked: bincount of 500M ids at once needs ~4 GB
        counts += np.bincount(arr[i : i + step], minlength=vocab)[:vocab]
    np.save(path, counts)
    return counts


@torch.no_grad()
def per_token_loss(run: Path, val: np.ndarray, n_tokens: int, batch: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = GPTConfig(**json.loads((run / "config.json").read_text())["model"])
    model = GPT(cfg).to(device)
    state = torch.load(run / "ckpt.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    T = cfg.block_size
    n_win = min(n_tokens, len(val) - 1) // T
    starts = np.arange(n_win) * T  # contiguous, non-overlapping windows
    losses, targets = [], []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    for i in range(0, n_win, batch):
        s = starts[i : i + batch]
        x = torch.from_numpy(np.stack([val[j : j + T] for j in s]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[j + 1 : j + 1 + T] for j in s]).astype(np.int64)).to(device)
        with ctx:
            logits = model(x)
        l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        losses.append(l.cpu().numpy())
        targets.append(y.reshape(-1).cpu().numpy())
    return np.concatenate(losses), np.concatenate(targets)


def bucket_losses(loss: np.ndarray, tgt: np.ndarray, counts: np.ndarray) -> dict:
    b = np.digitize(counts[tgt], EDGES[1:-1])
    out = {"all": {"loss": float(loss.mean()), "n": int(len(loss))}}
    for k, label in enumerate(LABELS):
        m = b == k
        out[label] = {"loss": float(loss[m].mean()) if m.any() else None, "n": int(m.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/fineweb_gpt2"))
    ap.add_argument("--tokens", type=int, default=5_000_000, help="validation tokens to evaluate")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = train_counts(args.data, args.vocab)
    val = np.memmap(args.data / "val.bin", dtype=np.uint16, mode="r")

    groups: dict[str, list[dict]] = defaultdict(list)
    for run in sorted(p for p in args.runs.iterdir() if (p / "DONE").exists() and (p / "ckpt.pt").exists()):
        cache = run / "eval_freq.json"
        if cache.exists() and not args.recompute:
            res = json.loads(cache.read_text())
        else:
            loss, tgt = per_token_loss(run, val, args.tokens, args.batch, device)
            res = bucket_losses(loss, tgt, counts)
            cache.write_text(json.dumps(res, indent=1))
            print(f"evaluated {run.name}: {res['all']['n']:,} tokens, loss {res['all']['loss']:.4f}", flush=True)
        groups[re.sub(r"_s\d+$", "", run.name)].append(res)

    if "baseline" not in groups:
        raise SystemExit("no finished baseline run found")
    cols = ["all"] + LABELS
    share = {c: groups["baseline"][0][c]["n"] / groups["baseline"][0]["all"]["n"] for c in cols}

    def mean(name, c):
        v = [r[c]["loss"] for r in groups[name] if r[c]["loss"] is not None]
        return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan) if v else (np.nan, np.nan)

    w = 11
    print("\ntraining-set frequency of the target token →")
    print(f"{'':<26}" + "".join(f"{c:>{w}}" for c in cols))
    print(f"{'share of val tokens':<26}" + "".join(f"{share[c]:>{w}.1%}" for c in cols))
    print(f"{'baseline loss':<26}" + "".join(f"{mean('baseline', c)[0]:>{w}.3f}" for c in cols))
    print("\nΔ loss vs baseline (mean over seeds; negative = better)")
    for name in sorted(g for g in groups if g != "baseline"):
        print(f"{name:<26}" + "".join(f"{mean(name, c)[0] - mean('baseline', c)[0]:>+{w}.4f}" for c in cols))
    print("\nseed spread (std of loss across seeds)")
    for name in sorted(groups):
        print(f"{name:<26}" + "".join(f"{mean(name, c)[1]:>{w}.4f}" for c in cols))


if __name__ == "__main__":
    main()
