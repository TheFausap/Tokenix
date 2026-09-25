"""Ways to constrain the embedding matrix's spectrum.

The factorisation ``E = U diag(s) Vt`` suggests three independent knobs:

* ``U`` -- *where* tokens go: can be drawn from a tokenizer graph
  (:func:`graph_spectral_init`) or pulled towards it (:func:`dirichlet_grad`);
* ``s`` -- *how much* each direction is stretched: can be fixed to a target
  profile (:func:`project_spectrum`);
* ``Vt`` -- a free rotation, absorbed by the next layer.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .spectral import Spectrum


def powerlaw_spectrum(d: int, alpha: float = 0.5) -> np.ndarray:
    return np.arange(1, d + 1, dtype=float) ** -alpha


def flat_spectrum(d: int) -> np.ndarray:
    return np.ones(d)


def _rescale(E: np.ndarray, fro: float | None) -> np.ndarray:
    return E if fro is None else E * (fro / np.linalg.norm(E))


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def project_spectrum(E: np.ndarray, target: np.ndarray, fro: float | None = None) -> np.ndarray:
    """Keep ``E``'s rotations, replace its stretch by ``target`` (up to scale).

    If ``fro`` is ``None`` the Frobenius norm of ``E`` is preserved.
    """
    U, _, Vt = np.linalg.svd(E, full_matrices=False)
    out = (U * target) @ Vt
    return _rescale(out, np.linalg.norm(E) if fro is None else fro)


def graph_spectral_init(
    spec: Spectrum,
    d: int,
    rng: np.random.Generator,
    stretch: np.ndarray | None = None,
    skip: int = 1,
    fro: float | None = None,
) -> np.ndarray:
    """Embedding whose left singular vectors are the ``d`` smoothest non-trivial
    Laplacian eigenvectors (a Laplacian eigenmap), with singular values
    ``stretch`` (default flat) and a random right rotation.
    """
    U = spec.eigenvectors[:, skip : skip + d]
    s = flat_spectrum(d) if stretch is None else stretch
    E = (U * s) @ random_rotation(d, rng)
    return _rescale(E, fro)


def dirichlet_grad(E: np.ndarray, L: sp.spmatrix) -> np.ndarray:
    """Gradient of ``tr(E^T L E)`` -- a graph-smoothness regulariser."""
    return 2.0 * (L @ E)
