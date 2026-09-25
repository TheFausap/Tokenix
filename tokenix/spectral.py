"""Spectral graph theory on tokenizer graphs (the "matrices are graphs" lens)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.linalg as la


def laplacian(W: sp.spmatrix, normalized: bool = True) -> sp.csr_matrix:
    """Combinatorial ``D - W`` or symmetric normalised ``I - D^-1/2 W D^-1/2``.

    Isolated vertices get a zero row/column in both cases.
    """
    W = sp.csr_matrix(W, dtype=float)
    deg = np.asarray(W.sum(axis=1)).ravel()
    if not normalized:
        return (sp.diags(deg) - W).tocsr()
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = deg[nz] ** -0.5
    Dm = sp.diags(inv)
    return (sp.diags(nz.astype(float)) - Dm @ W @ Dm).tocsr()


@dataclass
class Spectrum:
    eigenvalues: np.ndarray  # ascending
    eigenvectors: np.ndarray  # columns, orthonormal
    n_components: int

    @property
    def algebraic_connectivity(self) -> float:
        """First non-trivial eigenvalue (Fiedler value) past the null space."""
        k = self.n_components
        return float(self.eigenvalues[k]) if k < len(self.eigenvalues) else 0.0

    @property
    def fiedler_vector(self) -> np.ndarray:
        return self.eigenvectors[:, self.n_components]

    def heat_trace(self, t: float) -> float:
        """``tr exp(-tL)`` -- a multiscale spectral signature of the graph."""
        return float(np.exp(-t * self.eigenvalues).sum())

    def spectral_density(self, bins: int = 50, range_: tuple[float, float] | None = None):
        return np.histogram(self.eigenvalues, bins=bins, range=range_, density=True)


def spectrum(W: sp.spmatrix, normalized: bool = True, k: int | None = None) -> Spectrum:
    """Laplacian eigendecomposition.

    Dense ``eigh`` when ``k`` is ``None`` (fine up to a few thousand vertices);
    otherwise the ``k`` smallest eigenpairs via shift-invert Lanczos.
    """
    L = laplacian(W, normalized)
    n_comp = csgraph.connected_components(W, directed=False)[0]
    if k is None or k >= L.shape[0] - 1:
        vals, vecs = la.eigh(L.toarray())
    else:
        vals, vecs = sp.linalg.eigsh(L, k=k, sigma=-1e-3, which="LM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    return Spectrum(np.clip(vals, 0, None), vecs, int(n_comp))


def graph_fourier(signal: np.ndarray, spec: Spectrum) -> np.ndarray:
    """Coefficients of a vertex signal (or matrix of signals) in the Laplacian eigenbasis."""
    return spec.eigenvectors.T @ signal


def largest_component(W: sp.spmatrix) -> np.ndarray:
    """Sorted vertex ids of the largest connected component."""
    _, labels = csgraph.connected_components(W, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())
