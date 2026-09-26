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
    def __init__(self, W: sp.spmatrix, perm: np.ndarray | None = None):
        """``perm`` relabels the graph's vertices (the shuffled-graph control)."""
        super().__init__()
        W = sp.triu(sp.csr_matrix(W, dtype=np.float64), k=1).tocoo()
        rows, cols, w = W.row, W.col, W.data
        n = max(W.shape)
        deg = np.bincount(rows, weights=w, minlength=n) + np.bincount(cols, weights=w, minlength=n)
        scale = np.zeros(n)
        scale[deg > 0] = deg[deg > 0] ** -0.5
        if perm is not None:
            rows, cols = perm[rows], perm[cols]
            scale = scale[np.argsort(perm)]  # scale follows its vertex to perm[vertex]
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.n = n

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        E = E[: self.n].float()
        X = E * self.scale[:, None]
        diff = X[self.rows] - X[self.cols]
        return (self.w * diff.pow(2).sum(dim=1)).sum() / E.pow(2).sum()
