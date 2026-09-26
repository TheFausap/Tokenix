"""Graph-smoothness penalty on an embedding matrix, in PyTorch.

For a symmetric adjacency ``W`` with degrees ``d`` and the normalised Laplacian
``L = I - D^-1/2 W D^-1/2``,

    tr(E^T L E) = sum_{i<j} w_ij || e_i / sqrt(d_i) - e_j / sqrt(d_j) ||^2 ,

so the penalty only needs the edge list.  :class:`GraphPenalty` returns the
Rayleigh quotient ``tr(E^T L E) / ||E||_F^2`` (the same quantity as
:func:`tokenix.alignment.dirichlet_energy`), which cannot be lowered by simply
shrinking ``E``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class GraphPenalty(torch.nn.Module):
    """Graph-smoothness penalty on the rows of an embedding matrix.

    ``mode="rayleigh"``: ``tr(E^T L E) / ||E||_F^2``.  Scale-invariant as a
    whole, which leaves a loophole: rows the language-model loss never touches
    (tokens absent from training) can lower the quotient by growing large
    together while staying mutually smooth.  In practice clusters of unseen
    tokens (braces, ``\\r``) inflated ~15x and got extreme logits out of domain.

    ``mode="cosine"``: the same Laplacian energy on unit-normalised rows,
    ``sum_ij a_ij ||e_i/|e_i| - e_j/|e_j|||^2`` with ``a_ij = w_ij / sqrt(d_i d_j)``
    normalised to sum to 1.  Row norms get no gradient from it at all.

    In Rayleigh mode the denominator runs over vertices that still have an edge,
    so rows outside the graph cannot lower the quotient by growing.

    ``keep`` (boolean, per token) drops every edge touching a token outside it,
    e.g. tokens seen fewer than N times in training.  It is applied in token
    space, after the optional relabelling ``perm`` (the shuffled-graph control),
    so real and shuffled graphs avoid the same tokens.
    """

    def __init__(self, W: sp.spmatrix, perm: np.ndarray | None = None, mode: str = "rayleigh",
                 keep: np.ndarray | None = None):
        super().__init__()
        if mode not in ("rayleigh", "cosine"):
            raise ValueError(f"unknown mode {mode!r}")
        W = sp.triu(sp.csr_matrix(W, dtype=np.float64), k=1).tocoo()
        rows, cols, w = W.row, W.col, W.data
        n = max(W.shape)
        if perm is not None:
            rows, cols = perm[rows], perm[cols]
        if keep is not None:
            ok = keep[rows] & keep[cols]
            rows, cols, w = rows[ok], cols[ok], w[ok]
        deg = np.bincount(rows, weights=w, minlength=n) + np.bincount(cols, weights=w, minlength=n)
        scale = np.zeros(n)
        scale[deg > 0] = deg[deg > 0] ** -0.5
        if mode == "cosine":  # fold the degree normalisation into the edge weights
            w = w * scale[rows] * scale[cols]
            w = w / w.sum()
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        # Rayleigh denominator over graph vertices only: a row outside every edge (masked by
        # ``keep``, or isolated) would otherwise lower the quotient just by growing.
        self.register_buffer("in_graph", torch.as_tensor(deg > 0))
        self.n, self.mode, self.n_edges = n, mode, len(w)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        E = E[: self.n].float()
        if self.mode == "cosine":
            X = E / E.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return (self.w * (X[self.rows] - X[self.cols]).pow(2).sum(dim=1)).sum()
        X = E * self.scale[:, None]
        diff = X[self.rows] - X[self.cols]
        return (self.w * diff.pow(2).sum(dim=1)).sum() / E[self.in_graph].pow(2).sum()
