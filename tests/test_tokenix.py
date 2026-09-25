import json

import numpy as np
import pytest

from tokenix import Encoder, containment_graph, merge_graph, spectrum, train_bpe, transition_graph
from tokenix import alignment, constraints, embedding
from tokenix.io import load_hf_tokenizer_json
from tokenix.lm import BigramLM
from tokenix.spectral import laplacian

TEXT = "the cat sat on the mat. the hat is on the cat. that cat, the fat cat, sat." * 20


@pytest.fixture(scope="module")
def tok():
    return train_bpe(TEXT, 30)


def test_bpe_roundtrip_and_compression(tok):
    enc = Encoder(tok)
    ids = enc.encode(TEXT)
    assert enc.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode()) / 2
    assert all(tok.tokens[a] + tok.tokens[b] == tok.tokens[c] for a, b, c in tok.merges)
    assert max(tok.depth()) >= 2


def test_merge_and_containment_graphs(tok):
    for W in (merge_graph(tok), containment_graph(tok)):
        assert W.shape == (len(tok), len(tok))
        assert (W != W.T).nnz == 0
        assert W.diagonal().sum() == 0
    # Every merged token contains its parents, so it covers something shorter.
    Wc = containment_graph(tok).tolil()
    for _, _, c in tok.merges:
        assert any(len(tok.tokens[j]) < len(tok.tokens[c]) for j in Wc.rows[c])


def test_containment_is_covering_relation():
    from tokenix.spec import TokenizerSpec

    spec = TokenizerSpec([b"a", b"b", b"ab", b"abb", b"bb"], [(0, 1, 2), (2, 1, 3), (1, 1, 4)])
    W = containment_graph(spec).toarray()
    assert W[3, 2] and W[3, 4]  # ab < abb, bb < abb are covers
    assert not W[3, 0]  # a < ab < abb: not a cover
    assert W[2, 0] and W[2, 1] and W[4, 1]


def test_spectrum_basics(tok):
    W = transition_graph(Encoder(tok).encode(TEXT), len(tok))
    s = spectrum(W)
    assert np.all(np.diff(s.eigenvalues) >= -1e-10)
    assert s.eigenvalues.max() <= 2 + 1e-8
    assert np.sum(s.eigenvalues < 1e-8) == s.n_components
    np.testing.assert_allclose(s.eigenvectors.T @ s.eigenvectors, np.eye(len(tok)), atol=1e-8)


def test_spectrum_path_graph():
    import scipy.sparse as sp

    n = 10
    W = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1])
    s = spectrum(W, normalized=False)
    expected = 2 - 2 * np.cos(np.pi * np.arange(n) / n)
    np.testing.assert_allclose(s.eigenvalues, expected, atol=1e-10)
    assert s.n_components == 1


def test_polar_and_profile():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((50, 8))
    Q, P = embedding.polar(E)
    np.testing.assert_allclose(Q @ P, E, atol=1e-10)
    np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-10)
    assert np.all(np.linalg.eigvalsh(P) > 0)
    prof = embedding.profile(E)
    assert 1 <= prof["effective_rank"] <= 8
    # anisotropy matches the brute-force mean cosine
    X = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = X @ X.T
    brute = (S.sum() - 50) / (50 * 49)
    assert abs(prof["anisotropy"] - brute) < 1e-10


def test_constraints(tok):
    rng = np.random.default_rng(0)
    d = 6
    E = rng.standard_normal((len(tok), d))
    target = constraints.powerlaw_spectrum(d, 1.0)
    P = constraints.project_spectrum(E, target)
    s = np.linalg.svd(P, compute_uv=False)
    np.testing.assert_allclose(s / s[0], target / target[0], atol=1e-10)
    assert np.isclose(np.linalg.norm(P), np.linalg.norm(E))

    sp_ = spectrum(containment_graph(tok))
    G = constraints.graph_spectral_init(sp_, d, rng, skip=sp_.n_components)
    L = laplacian(containment_graph(tok))
    # A Laplacian eigenmap is much smoother on the graph than a random matrix.
    assert alignment.dirichlet_energy(G, L) < alignment.dirichlet_energy(E, L)
    assert alignment.subspace_alignment(G, sp_, d, skip=sp_.n_components) > 0.99
    assert np.isclose(alignment.frequency_profile(E, sp_).sum(), 1.0)


def test_bigram_lm_gradients_and_training(tok):
    rng = np.random.default_rng(0)
    V, d = len(tok), 4
    ids = Encoder(tok).encode(TEXT)
    from tokenix import bigram_counts

    C = bigram_counts(ids, V).toarray()
    m = BigramLM.init(0.1 * rng.standard_normal((V, d)), rng)
    loss, gE, gA, gb = m.grads(C)
    # finite-difference check on a few coordinates
    for arr, g in ((m.E, gE), (m.A, gA), (m.b, gb)):
        idx = tuple(rng.integers(0, s) for s in arr.shape)
        h = 1e-6
        arr[idx] += h
        lp = m.nll(C)
        arr[idx] -= 2 * h
        lm = m.nll(C)
        arr[idx] += h
        assert abs((lp - lm) / (2 * h) - g[idx]) < 1e-5
    before = m.nll(C)
    m.fit(C, steps=50, lr=0.05)
    assert m.nll(C) < before


def test_hf_loader(tmp_path):
    data = {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "Ġ": 2, "ab": 3, "Ġab": 4},
            "merges": ["a b", ["Ġ", "ab"]],
        }
    }
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(data))
    spec = load_hf_tokenizer_json(p)
    assert spec.tokens[2] == b" " and spec.tokens[4] == b" ab"
    assert spec.merges == [(0, 1, 3), (2, 3, 4)]


def test_safetensors_reader(tmp_path):
    import struct

    from tokenix.hub import SafetensorsSource

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([1.5, -2.0], dtype=np.float32)
    b_bf16 = (b.view(np.uint32) >> 16).astype(np.uint16)
    header = {
        "__metadata__": {"format": "pt"},
        "wte.weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]},
        "x": {"dtype": "BF16", "shape": [2], "data_offsets": [48, 52]},
    }
    h = json.dumps(header).encode()
    p = tmp_path / "model.safetensors"
    p.write_bytes(struct.pack("<Q", len(h)) + h + a.tobytes() + b_bf16.tobytes())
    src = SafetensorsSource(p)
    assert set(src.names()) == {"wte.weight", "x"}
    np.testing.assert_array_equal(src.load("wte.weight"), a)
    np.testing.assert_array_equal(src.load("x"), b)


def test_permutation_controls(tok):
    from tokenix.spectral import largest_component

    rng = np.random.default_rng(0)
    groups = rng.integers(0, 3, 40)
    perm = alignment.stratified_permutation(groups, rng)
    assert sorted(perm) == list(range(40))
    assert np.all(groups[perm] == groups)

    W = containment_graph(tok)
    lcc = largest_component(W)
    L = laplacian(W[lcc][:, lcc])
    sp_ = spectrum(W[lcc][:, lcc])

    def stat(M):
        return alignment.dirichlet_energy(M, L)

    smooth = sp_.eigenvectors[:, 1:4]
    noise = rng.standard_normal((len(lcc), 3))
    assert alignment.permutation_test(stat, smooth, 20, rng)["z"] < -3
    assert abs(alignment.permutation_test(stat, noise, 20, rng)["z"]) < 4


def test_sparse_spectrum_matches_dense(tok):
    from tokenix.spectral import largest_component

    W = containment_graph(tok)
    lcc = largest_component(W)
    W = W[lcc][:, lcc]
    dense = spectrum(W).eigenvalues[:5]
    sparse = spectrum(W, k=5).eigenvalues
    np.testing.assert_allclose(sparse, dense, atol=1e-8)
