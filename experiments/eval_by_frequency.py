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


def coarse(res: dict, labels: list[str]) -> dict:
    """n-weighted mean loss over several fine buckets."""
    parts = [res[l] for l in labels if res[l]["n"] and res[l]["loss"] is not None]
    n = sum(p["n"] for p in parts)
    return {"loss": sum(p["loss"] * p["n"] for p in parts) / n if n else None, "n": n}


RARE, FREQ = LABELS[:3], LABELS[3:]  # < 1k vs >= 1k training occurrences


def evaluate_set(runs: Path, name: str, val: np.ndarray, counts: np.ndarray, args, device) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    fname = "eval_freq.json" if name == "fineweb" else f"eval_freq_{name}.json"
    for run in sorted(p for p in runs.iterdir() if (p / "DONE").exists() and (p / "ckpt.pt").exists()):
        cache = run / fname
        if cache.exists() and not args.recompute:
            res = json.loads(cache.read_text())
        else:
            loss, tgt = per_token_loss(run, val, args.tokens, args.batch, device)
            res = bucket_losses(loss, tgt, counts)
            cache.write_text(json.dumps(res, indent=1))
            print(f"[{name}] evaluated {run.name}: {res['all']['n']:,} tokens, loss {res['all']['loss']:.4f}", flush=True)
        res["rare<1k"], res["freq≥1k"] = coarse(res, RARE), coarse(res, FREQ)
        groups[re.sub(r"_s\d+$", "", run.name)].append(res)
    if "baseline" not in groups:
        raise SystemExit("no finished baseline run found")
    return groups


def mean(groups, name, c):
    v = [r[c]["loss"] for r in groups[name] if r[c]["loss"] is not None]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan) if v else (np.nan, np.nan)


def print_detail(name: str, groups: dict) -> None:
    cols = ["all"] + LABELS
    base = groups["baseline"][0]
    w = 11
    print(f"\n=== {name}: training-set frequency of the target token →")
    print(f"{'':<26}" + "".join(f"{c:>{w}}" for c in cols))
    print(f"{'share of val tokens':<26}" + "".join(f"{base[c]['n'] / base['all']['n']:>{w}.1%}" for c in cols))
    print(f"{'val tokens (n)':<26}" + "".join(f"{base[c]['n']:>{w},}" for c in cols))
    print(f"{'baseline loss':<26}" + "".join(f"{mean(groups, 'baseline', c)[0]:>{w}.3f}" for c in cols))
    print("Δ loss vs baseline (mean over seeds; negative = better)")
    for g in sorted(k for k in groups if k != "baseline"):
        print(f"{g:<26}" + "".join(f"{mean(groups, g, c)[0] - mean(groups, 'baseline', c)[0]:>+{w}.4f}" for c in cols))
    print("seed spread (std of loss across seeds)")
    for g in sorted(groups):
        print(f"{g:<26}" + "".join(f"{mean(groups, g, c)[1]:>{w}.4f}" for c in cols))


def print_summary(results: dict) -> None:
    conds = sorted(k for k in next(iter(results.values())) if k != "baseline")
    short = {c: c.replace("_lam", " λ").replace("-shuffled", ":shuf") for c in conds}
    w = max(12, max(len(v) for v in short.values()) + 2)
    for col, title in (("all", "all tokens"), ("rare<1k", "rare targets (<1k train occurrences)")):
        print(f"\n=== Δ loss vs baseline on {title} (mean over seeds; negative = better)")
        print(f"{'eval set':<14}{'rare share':>11}{'base loss':>10}" + "".join(f"{short[c]:>{w}}" for c in conds)
              + f"{'base std':>10}")
        for name, groups in results.items():
            base = groups["baseline"][0]
            b, sd = mean(groups, "baseline", col)
            print(f"{name:<14}{base['rare<1k']['n'] / base['all']['n']:>11.1%}{b:>10.3f}"
                  + "".join(f"{mean(groups, c, col)[0] - b:>+{w}.4f}" for c in conds) + f"{sd:>10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/fineweb_gpt2"))
    ap.add_argument("--eval_dir", type=Path, default=None,
                    help="directory of extra <name>.bin sets (experiments/prepare_eval_sets.py)")
    ap.add_argument("--tokens", type=int, default=5_000_000, help="validation tokens to evaluate per set")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--detail", action="store_true", help="print the per-bucket table for every set")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = train_counts(args.data, args.vocab)
    sets = {"fineweb": args.data / "val.bin"}
    if args.eval_dir:
        sets.update({p.stem: p for p in sorted(args.eval_dir.glob("*.bin"))})

    results = {}
    for name, path in sets.items():
        val = np.memmap(path, dtype=np.uint16, mode="r")
        results[name] = evaluate_set(args.runs, name, val, counts, args, device)
        if args.detail or len(sets) == 1:
            print_detail(name, results[name])
    if len(sets) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
