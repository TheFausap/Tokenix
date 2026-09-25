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
