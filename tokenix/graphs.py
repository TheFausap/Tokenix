"""Graphs a tokenizer induces on its own vocabulary.

All builders return a symmetric, non-negative ``scipy.sparse.csr_matrix`` of
shape ``(V, V)`` -- a weighted adjacency matrix -- so they plug directly into
:mod:`tokenix.spectral`.

* :func:`merge_graph`       -- undirected merge DAG: ``c`` joined to ``a`` and ``b``.
* :func:`containment_graph` -- Hasse diagram of the substring order on ``V``.
* :func:`transition_graph`  -- token co-occurrence (bigrams) in an encoded corpus.

The first two depend on the tokenizer alone; the third on tokenizer + data.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp

from .spec import TokenizerSpec


def _symmetric(rows, cols, vals, n: int) -> sp.csr_matrix:
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def merge_graph(spec: TokenizerSpec) -> sp.csr_matrix:
    rows, cols = [], []
    for a, b, c in spec.merges:
        rows += [c, c]
        cols += [a, b]
    return _symmetric(rows, cols, np.ones(len(rows)), len(spec)).sign()


def _substrings(s: bytes) -> Iterable[bytes]:
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if j - i < n:
                yield s[i:j]


def containment_graph(spec: TokenizerSpec, max_len: int = 32) -> sp.csr_matrix:
    """Hasse diagram (covering relation) of the substring poset on the vocabulary.

    ``t -- u`` iff ``t`` is a proper substring of ``u`` and no token ``v`` sits
    strictly between them.  Tokens longer than ``max_len`` bytes are only
    checked for substrings up to that length.
    """
    index = spec.index
    rows, cols = [], []
    for u, s in enumerate(spec.tokens):
        below = {index[x] for x in _substrings(s[:max_len] if len(s) > max_len else s) if x in index}
        if len(s) > max_len:
            below |= {index[x] for x in _substrings(s[-max_len:]) if x in index}
        below.discard(u)
        # Keep the maximal elements of `below`: those not contained in another.
        for t in below:
            ts = spec.tokens[t]
            if not any(v != t and ts in spec.tokens[v] for v in below):
                rows.append(u)
                cols.append(t)
    return _symmetric(rows, cols, np.ones(len(rows)), len(spec)).sign()


def transition_graph(
    ids: Iterable[int] | np.ndarray, vocab_size: int, weighting: str = "count"
) -> sp.csr_matrix:
    """Symmetrised bigram graph of an encoded corpus.

    ``weighting``: ``"count"`` (raw counts), ``"log"`` (``log1p`` counts) or
    ``"ppmi"`` (positive pointwise mutual information).
    """
    C = bigram_counts(ids, vocab_size)
    W = (C + C.T).tocsr().astype(float)
    W.setdiag(0)
    W.eliminate_zeros()
    if weighting == "count":
        return W
    if weighting == "log":
        W.data = np.log1p(W.data)
        return W
    if weighting == "ppmi":
        total = W.sum()
        deg = np.asarray(W.sum(axis=1)).ravel()
        W = W.tocoo()
        pmi = np.log(W.data * total / (deg[W.row] * deg[W.col]))
        keep = pmi > 0
        return sp.coo_matrix((pmi[keep], (W.row[keep], W.col[keep])), shape=W.shape).tocsr()
    raise ValueError(f"unknown weighting {weighting!r}")


def bigram_counts(ids: Iterable[int] | np.ndarray, vocab_size: int) -> sp.csr_matrix:
    """Directed bigram count matrix ``C[i, j] = #(i followed by j)``."""
    x = np.asarray(ids if isinstance(ids, np.ndarray) else list(ids), dtype=np.int64)
    C = sp.coo_matrix((np.ones(len(x) - 1), (x[:-1], x[1:])), shape=(vocab_size, vocab_size))
    return C.tocsr()
