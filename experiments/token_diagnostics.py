"""Which token types drive a condition's loss change vs baseline, per eval set?

    python experiments/token_diagnostics.py runs/ --data data/fineweb_gpt2 --eval_dir data/eval

For every finished run and eval set it stores the summed loss and count per
target token type (``<run>/type_loss_<set>.npz``, reused on re-runs).  Then,
for each eval set and each condition, it reports:

* a breakdown of the change in mean loss vs baseline by token category
  (whitespace, digits, punctuation/symbols, words, byte fragments, other),
* the ``--top`` token types that add the most loss and the ones that remove the
  most, decoded, with occurrences, training count and per-occurrence change.

A type's *contribution* is its share of the change in the set's mean loss:
``(sum of cond losses - sum of baseline losses) / total tokens``, with each run
group averaged over seeds, so the contributions add up to the Δ in the summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_by_frequency import per_token_loss, train_counts  # noqa: E402

CATEGORIES = ["whitespace", "digits", "punct/symbol", "word", "byte fragment", "other"]


def category(tok: bytes) -> str:
    s = tok.decode("utf-8", errors="replace")
    if "�" in s:
        return "byte fragment"
    if s.isspace():
        return "whitespace"
    core = s.strip()
    if core.isdigit():
        return "digits"
    if core.isalpha():
        return "word"
    if core and not any(c.isalnum() for c in core):
        return "punct/symbol"
    return "other"


def show(tok: bytes) -> str:
    """Printable form: spaces as '·', newlines/tabs escaped, quoted so edges are visible."""
    s = repr(tok.decode("utf-8", errors="backslashreplace"))[1:-1].replace(" ", "·")
    return f"'{s[:22]}'"


def type_sums(run: Path, name: str, val, args, device, vocab: int):
    cache = run / f"type_loss_{name}.npz"
    if cache.exists() and not args.recompute:
        z = np.load(cache)
        return z["loss"], z["count"]
    loss, tgt = per_token_loss(run, val, args.tokens, args.batch, device)
    s = np.bincount(tgt, weights=loss, minlength=vocab)[:vocab]
    c = np.bincount(tgt, minlength=vocab)[:vocab]
    np.savez(cache, loss=s, count=c)
    print(f"[{name}] per-type losses for {run.name}", flush=True)
    return s, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/fineweb_gpt2"))
    ap.add_argument("--eval_dir", type=Path, default=None)
    ap.add_argument("--sets", nargs="*", default=None, help="restrict to these eval set names")
    ap.add_argument("--conditions", nargs="*", default=None, help="default: every non-baseline group")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    from tokenix.hub import fetch_file
    from tokenix.io import load_hf_tokenizer_json

    spec = load_hf_tokenizer_json(fetch_file("openai-community/gpt2", "tokenizer.json"))
    cats = np.array([category(t) for t in spec.tokens[: args.vocab]])
    counts = train_counts(args.data, args.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sets = {"fineweb": args.data / "val.bin"}
    if args.eval_dir:
        sets.update({p.stem: p for p in sorted(args.eval_dir.glob("*.bin"))})
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}

    runs = sorted(p for p in args.runs.iterdir() if (p / "DONE").exists() and (p / "ckpt.pt").exists())
    for name, path in sets.items():
        val = np.memmap(path, dtype=np.uint16, mode="r")
        groups: dict[str, list] = defaultdict(list)
        for run in runs:
            groups[re.sub(r"_s\d+$", "", run.name)].append(type_sums(run, name, val, args, device, args.vocab))
        mean_sum = {g: np.mean([s for s, _ in v], axis=0) for g, v in groups.items()}
        n_type = groups["baseline"][0][1]
        N = n_type.sum()
        conds = args.conditions or sorted(g for g in groups if g != "baseline")

        print(f"\n{'=' * 100}\n=== {name}: {int(N):,} tokens, baseline mean loss {mean_sum['baseline'].sum() / N:.4f}")
        print(f"\nΔ mean loss vs baseline by category of the target token (contributions sum to the total)")
        share = {c: n_type[cats == c].sum() / N for c in CATEGORIES}
        w = 15
        print(f"{'':<26}{'total':>{w}}" + "".join(f"{c:>{w}}" for c in CATEGORIES))
        print(f"{'share of tokens':<26}{1:>{w}.1%}" + "".join(f"{share[c]:>{w}.1%}" for c in CATEGORIES))
        for g in conds:
            d = (mean_sum[g] - mean_sum["baseline"]) / N
            print(f"{g:<26}{d.sum():>+{w}.4f}" + "".join(f"{d[cats == c].sum():>+{w}.4f}" for c in CATEGORIES))

        for g in conds:
            d = (mean_sum[g] - mean_sum["baseline"]) / N
            seen = n_type > 0
            per_occ = np.where(seen, (mean_sum[g] - mean_sum["baseline"]) / np.maximum(n_type, 1), 0)
            for title, order in (("add the most loss", np.argsort(-d)), ("remove the most loss", np.argsort(d))):
                print(f"\n[{name}] {g}: token types that {title} (total Δ {d.sum():+.4f})")
                print(f"  {'token':<26}{'category':<15}{'val occ':>9}{'train cnt':>11}{'base loss':>10}"
                      f"{'Δ/occ':>9}{'contrib':>10}")
                for t in order[: args.top]:
                    if not seen[t]:
                        continue
                    print(f"  {show(spec.tokens[t]):<26}{cats[t]:<15}{int(n_type[t]):>9,}{int(counts[t]):>11,}"
                          f"{mean_sum['baseline'][t] / n_type[t]:>10.2f}{per_occ[t]:>+9.2f}{d[t]:>+10.4f}")


if __name__ == "__main__":
    main()
