"""Figures for paper/tokenix.tex, drawn from the result tables in README.md.

    python paper/make_figures.py        # writes paper/figs/*.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

OUT = Path(__file__).resolve().parent / "figs"
REAL, SHUF = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated, light surface)
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42,
})

SETS = ["FineWeb\n(in-domain)", "arXiv", "Python code", "PubMed", "Wiki DE", "Wiki IT", "Wiki RU"]
# Delta mean validation loss vs baseline (nats), cosine prefix penalty, lambda = 0.1.
SETTINGS = {
    "GPT-2 tokenizer, 51M\n(4 seeds)": ([-0.0002, -0.052, -0.068, -0.036, -0.025, -0.013, -0.007],
                                        [+0.020, -0.012, -0.002, -0.024, +0.021, +0.026, +0.010]),
    "GPT-NeoX tokenizer, 51M\n(2 seeds)": ([-0.0001, -0.031, -0.266, -0.019, +0.003, +0.013, -0.153],
                                           [+0.021, -0.000, -0.191, -0.032, +0.143, +0.117, +0.416]),
    "GPT-2 tokenizer, 124M\n(2 seeds)": ([+0.0009, -0.041, -0.096, -0.017, -0.027, -0.017, -0.010],
                                         [+0.017, -0.009, -0.082, -0.024, +0.018, +0.020, -0.003]),
}


def style(ax):
    ax.axvline(0, color=INK, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_ood():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    y = np.arange(len(SETS))[::-1]
    h = 0.36
    for ax, (title, (real, shuf)) in zip(axes, SETTINGS.items()):
        ax.barh(y + h / 2 + 0.01, real, h, color=REAL, label="tokenizer prefix graph", edgecolor="white", lw=0.8)
        ax.barh(y - h / 2 - 0.01, shuf, h, color=SHUF, label="shuffled graph (control)", edgecolor="white", lw=0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        lim = max(0.08, np.max(np.abs(real + shuf)) * 1.08)
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, symmetric=True))
        style(ax)
    axes[0].set_yticks(y, SETS)
    for ax in axes:
        ax.set_xlabel("Δ loss vs baseline (nats)\n← better        worse →", fontsize=8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.65, -0.32), ncol=2, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.30, wspace=0.16)
    fig.savefig(OUT / "ood_cosine.pdf")
    plt.close(fig)


def fig_lambda():
    lam = np.array([0.03, 0.1, 0.3, 1.0, 3.0])
    real = np.array([0.012, 0.006, 0.003, 0.052, 0.162])
    shuf = np.array([0.019, 0.027, 0.038, 0.115, np.nan])
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(lam, real, "-o", color=REAL, lw=2, ms=5, mec="white", mew=1.2)
    ax.plot(lam, shuf, "-o", color=SHUF, lw=2, ms=5, mec="white", mew=1.2)
    ax.text(0.3, 0.045, "shuffled graph", color=INK, ha="center", va="bottom", fontsize=8)
    ax.text(0.3, -0.004, "prefix graph", color=INK, ha="center", va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(lam, ["0.03", "0.1", "0.3", "1", "3"])
    ax.set_xlabel("penalty weight λ")
    ax.set_ylabel("Δ val. loss vs baseline (nats)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0.022, 4.5)
    ax.set_ylim(-0.03, 0.175)
    fig.tight_layout()
    fig.savefig(OUT / "lambda_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ood()
    fig_lambda()
    print("wrote", *sorted(p.name for p in OUT.glob("*.pdf")))
