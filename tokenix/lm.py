"""A minimal tied-embedding bigram language model, trained full-batch in NumPy.

    logits(i -> j) = e_i^T A e_j + b_j,     E in R^{V x d},  A in R^{d x d}

The embedding matrix ``E`` is used on both sides, so its spectrum is the only
place token geometry lives: a controlled testbed for spectral constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp


def _dense(C) -> np.ndarray:
    return C.toarray() if sp.issparse(C) else np.asarray(C, dtype=float)


def _log_softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


@dataclass
class BigramLM:
    E: np.ndarray
    A: np.ndarray
    b: np.ndarray
    history: list[dict] = field(default_factory=list)

    @classmethod
    def init(cls, E: np.ndarray, rng: np.random.Generator) -> "BigramLM":
        V, d = E.shape
        A = np.eye(d) + 0.01 * rng.standard_normal((d, d))
        return cls(E.copy(), A, np.zeros(V))

    def logits(self) -> np.ndarray:
        return self.E @ self.A @ self.E.T + self.b

    def nll(self, C) -> float:
        """Mean negative log-likelihood (nats/token) of bigram counts ``C``."""
        C = _dense(C)
        return float(-(C * _log_softmax(self.logits())).sum() / C.sum())

    def grads(self, C: np.ndarray):
        logp = _log_softmax(self.logits())
        N = C.sum()
        G = (C.sum(axis=1, keepdims=True) * np.exp(logp) - C) / N
        loss = float(-(C * logp).sum() / N)
        gE = G @ (self.E @ self.A.T) + G.T @ (self.E @ self.A)
        gA = self.E.T @ G @ self.E
        gb = G.sum(axis=0)
        return loss, gE, gA, gb

    def fit(
        self,
        C_train,
        C_test=None,
        steps: int = 300,
        lr: float = 0.02,
        reg_grad: Callable[[np.ndarray], np.ndarray] | None = None,
        project: Callable[[np.ndarray], np.ndarray] | None = None,
        log_every: int = 50,
    ) -> "BigramLM":
        """Adam on the full bigram table.

        ``reg_grad(E)`` is added to the embedding gradient; ``project(E)`` is
        applied after each step (projected gradient descent).
        """
        C = _dense(C_train)
        params = [self.E, self.A, self.b]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            loss, *gs = self.grads(C)
            if reg_grad is not None:
                gs[0] = gs[0] + reg_grad(self.E)
            for p, g, mi, vi in zip(params, gs, m, v):
                mi *= b1
                mi += (1 - b1) * g
                vi *= b2
                vi += (1 - b2) * g * g
                p -= lr * (mi / (1 - b1**t)) / (np.sqrt(vi / (1 - b2**t)) + eps)
            if project is not None:
                self.E[...] = project(self.E)
            if t % log_every == 0 or t == steps:
                rec = {"step": t, "train_nll": loss}
                if C_test is not None:
                    rec["test_nll"] = self.nll(C_test)
                self.history.append(rec)
        return self
