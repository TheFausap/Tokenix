"""Summarise ``train_gpt.py`` runs: final validation loss per condition, speed-up
versus baseline, and the embedding's Dirichlet ratios.

    python experiments/summarize_runs.py runs/ [--plot runs/curves.png]

Runs are grouped by name minus the ``_s<seed>`` suffix.  "tokens to baseline"
is the first eval at which a run's validation loss is at or below the mean
final baseline loss (linear interpolation between evals), as a fraction of the
baseline's token budget.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run: Path) -> list[dict]:
    recs = {}
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[r["step"]] = r  # a resumed run re-logs its resume step: keep the last
    return [recs[k] for k in sorted(recs)]


def tokens_to(log: list[dict], target: float) -> float | None:
    for a, b in zip(log, log[1:]):
        if b["val_loss"] <= target:
            if a["val_loss"] == b["val_loss"]:
                return b["tokens"]
            f = (a["val_loss"] - target) / (a["val_loss"] - b["val_loss"])
            return a["tokens"] + f * (b["tokens"] - a["tokens"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    groups: dict[str, list] = defaultdict(list)
    for run in sorted(p for p in args.runs.iterdir() if (p / "log.jsonl").exists()):
        log = load(run)
        if len(log) < 2:
            continue
        groups[re.sub(r"_s\d+$", "", run.name)].append((run.name, log, (run / "DONE").exists()))

    base = groups.get("baseline", [])
    target = np.mean([log[-1]["val_loss"] for _, log, _ in base]) if base else None
    budget = np.mean([log[-1]["tokens"] for _, log, _ in base]) if base else None

    print(f"{'condition':<26}{'seeds':>6}{'final val loss':>18}{'Δ vs base':>11}"
          f"{'tokens→base':>13}{'Dir(prefix)':>12}{'Dir(contain)':>13}")
    for name in sorted(groups, key=lambda k: (k != "baseline", k)):
        runs = groups[name]
        finals = np.array([log[-1]["val_loss"] for _, log, _ in runs])
        sd = finals.std(ddof=1) if len(finals) > 1 else float("nan")
        delta = finals.mean() - target if target is not None else float("nan")
        reach = [tokens_to(log, target) for _, log, _ in runs] if target is not None else []
        reach_s = ("—" if not reach or any(r is None for r in reach)
                   else f"{np.mean(reach) / budget:.2f}×")
        dp = np.mean([log[-1]["dirichlet_prefix"] for _, log, _ in runs])
        dc = np.mean([log[-1]["dirichlet_contain"] for _, log, _ in runs])
        unfinished = sum(not done for _, _, done in runs)
        tag = f" ({unfinished} running)" if unfinished else ""
        print(f"{name + tag:<26}{len(runs):>6}{finals.mean():>11.4f} ± {sd:.4f}{delta:>+11.4f}"
              f"{reach_s:>13}{dp:>12.3f}{dc:>13.3f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, (name, runs) in enumerate(sorted(groups.items())):
            color = f"C{i}"
            for j, (_, log, _) in enumerate(runs):
                t = [r["tokens"] / 1e6 for r in log]
                a1.plot(t, [r["val_loss"] for r in log], color=color, alpha=0.8, label=name if j == 0 else None)
                a2.plot(t, [r["dirichlet_prefix"] for r in log], color=color, alpha=0.8, label=name if j == 0 else None)
        a1.set(xlabel="tokens (M)", ylabel="validation loss", title="Validation loss")
        lo = min(log[-1]["val_loss"] for runs in groups.values() for _, log, _ in runs)
        a1.set_ylim(lo - 0.05, lo + 1.0)
        a2.set(xlabel="tokens (M)", ylabel="Dirichlet ratio (prefix graph)",
               title="Embedding smoothness on the prefix graph (1 = none)")
        a2.axhline(1.0, color="gray", lw=0.8, ls="--")
        a1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
