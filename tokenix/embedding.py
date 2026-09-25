"""The embedding matrix as rotation x stretch (the SVD / polar lens).

For ``E`` of shape ``(V, d)`` with thin SVD ``E = U diag(s) Vt``:

* ``U``  -- how token space is laid out (V-side "rotation"),
* ``s``  -- the stretch: the embedding's singular spectrum,
* ``Vt`` -- the rotation of model space, which downstream weights can absorb.

The polar form ``E = Q P`` with ``Q = U Vt`` (orthonormal columns) and
``P = Vt.T diag(s) Vt`` (symmetric PSD) separates the two exactly.
"""

from __future__ import annotations

import numpy as np


def svd(E: np.ndarray):
    return np.linalg.svd(E, full_matrices=False)


def polar(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``E = Q @ P`` with ``Q`` an isometry (rotation) and ``P`` PSD (stretch)."""
    U, s, Vt = svd(E)
    return U @ Vt, (Vt.T * s) @ Vt


def effective_rank(s: np.ndarray) -> float:
    """Roy & Vetterli (2007): ``exp`` of the entropy of the normalised spectrum."""
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(s: np.ndarray) -> float:
    return float((s**2).sum() / s[0] ** 2)


def powerlaw_exponent(s: np.ndarray, skip: int = 1) -> float:
    """Slope ``alpha`` of the least-squares fit ``s_i ~ i^-alpha`` (log-log)."""
    i = np.arange(1, len(s) + 1)[skip:]
    y = s[skip:]
    keep = y > 0
    return float(-np.polyfit(np.log(i[keep]), np.log(y[keep]), 1)[0])


def anisotropy(E: np.ndarray) -> float:
    """Mean pairwise cosine similarity of the (non-zero) rows, in O(Vd)."""
    n = np.linalg.norm(E, axis=1)
    X = E[n > 0] / n[n > 0, None]
    m = len(X)
    total = np.linalg.norm(X.sum(axis=0)) ** 2
    return float((total - m) / (m * (m - 1)))


def profile(E: np.ndarray) -> dict:
    """Summary statistics of an embedding matrix's singular spectrum."""
    s = svd(E)[1]
    return {
        "shape": E.shape,
        "singular_values": s,
        "effective_rank": effective_rank(s),
        "stable_rank": stable_rank(s),
        "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
        "powerlaw_alpha": powerlaw_exponent(s),
        "anisotropy": anisotropy(E),
        "mean_row_norm": float(np.linalg.norm(E, axis=1).mean()),
    }
