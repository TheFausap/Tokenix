"""Where the two lenses meet: how an embedding matrix sits on a tokenizer graph.

The columns of ``E`` are signals on the vocabulary graph.  Their smoothness on
the graph, and the overlap of ``E``'s left singular subspace with the graph's
low-frequency Laplacian modes, quantify whether the "rotation" part of ``E``
respects the tokenizer's combinatorial structure.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .spectral import Spectrum


def dirichlet_energy(E: np.ndarray, L: sp.spmatrix) -> float:
    """Rayleigh quotient ``tr(E^T L E) / tr(E^T E)``.

    For the normalised Laplacian it lies in ``[0, 2]``; small means neighbouring
    tokens get similar embeddings.
    """
    return float(np.sum(E * (L @ E)) / np.sum(E * E))


def frequency_profile(E: np.ndarray, spec: Spectrum) -> np.ndarray:
    """Fraction of ``||E||_F^2`` carried by each Laplacian eigenmode (ascending frequency)."""
    c = spec.eigenvectors.T @ E
    e = (c**2).sum(axis=1)
    return e / e.sum()


def low_frequency_fraction(E: np.ndarray, spec: Spectrum, k: int) -> float:
    """Energy share of the ``k`` smoothest modes. A random ``E`` gives about ``k/V``."""
    return float(frequency_profile(E, spec)[:k].sum())


def subspace_alignment(E: np.ndarray, spec: Spectrum, k: int, skip: int = 0) -> float:
    """Mean squared cosine of principal angles between top-``k`` left singular
    vectors of ``E`` and the ``k`` smoothest Laplacian eigenvectors (after
    skipping ``skip`` trivial ones).  1 = identical subspaces, ``~k/V`` = chance.
    """
    U = np.linalg.svd(E, full_matrices=False)[0][:, :k]
    G = spec.eigenvectors[:, skip : skip + k]
    cos = np.linalg.svd(U.T @ G, compute_uv=False)
    return float(np.mean(cos**2))


def stratified_permutation(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random permutation that only swaps indices sharing the same group label."""
    perm = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = rng.permutation(idx)
    return perm


def permutation_test(
    stat, E: np.ndarray, n: int, rng: np.random.Generator, groups: np.ndarray | None = None
) -> dict:
    """Compare ``stat(E)`` to ``stat(E[perm])`` over ``n`` random row permutations.

    Permuting the rows of ``E`` is the same as relabelling the graph's vertices:
    the spectrum is unchanged, only the token <-> vertex correspondence is lost.
    With ``groups`` the permutation is stratified (e.g. by token length), which
    controls for whatever the grouping explains.
    """
    obs = stat(E)
    null = np.array([
        stat(E[rng.permutation(len(E)) if groups is None else stratified_permutation(groups, rng)])
        for _ in range(n)
    ])
    sd = null.std(ddof=1) if n > 1 else np.nan
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_std": float(sd),
            "ratio": float(obs / null.mean()), "z": float((obs - null.mean()) / sd)}
