"""H1/H2 on pretrained models: does a real embedding matrix respect its tokenizer's graphs?

For each model we load the tokenizer (``tokenizer.json``) and only the embedding
tensor(s) of the checkpoint, build the merge DAG and substring poset, and on the
largest connected component of each graph measure:

* **Dirichlet ratio**: Dirichlet energy of ``E`` on the graph divided by its mean
  under random vertex relabelling. ``< 1`` means graph neighbours have more
  similar embeddings than chance.
* **Subspace alignment**: overlap between the top-k left singular vectors of ``E``
  and the k smoothest Laplacian modes, again against relabelled controls.

Two nulls are used: a uniform relabelling, and one stratified by token byte
length, which removes whatever token length alone explains (short tokens are
the hubs of both graphs).  Everything is also reported for the mean-centred
``E``, because the common mean direction dominates raw GPT-2 geometry.

Usage:
    python experiments/pretrained_probe.py                       # GPT-2 small..xl
    python experiments/pretrained_probe.py --models openai-community/gpt2
    python experiments/pretrained_probe.py --selftest            # offline, local BPE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenix import alignment, embedding  # noqa: E402
from tokenix.graphs import containment_graph, merge_graph  # noqa: E402
from tokenix.spec import TokenizerSpec  # noqa: E402
from tokenix.spectral import largest_component, laplacian, spectrum  # noqa: E402

GPT2 = ["openai-community/gpt2", "openai-community/gpt2-medium",
        "openai-community/gpt2-large", "openai-community/gpt2-xl"]


def graph_report(spec: TokenizerSpec, k: int) -> dict:
    out = {}
    for name, W in (("merge", merge_graph(spec)), ("contain", containment_graph(spec))):
        lcc = largest_component(W)
        Wl = W[lcc][:, lcc]
        S = spectrum(Wl, k=None if len(lcc) <= 3000 else k + 1)
        out[name] = {"lcc": lcc, "L": laplacian(Wl), "S": S,
                     "summary": {"vertices": len(lcc), "edges": int(Wl.nnz // 2),
                                 "lambda_2": S.algebraic_connectivity,
                                 "lowest_eigenvalues": S.eigenvalues[: 8].tolist()}}
    return out


def probe_matrix(E: np.ndarray, spec: TokenizerSpec, graphs: dict, k: int, n_perm: int,
                 seed: int) -> dict:
    lengths = np.minimum(np.array([len(t) for t in spec.tokens]), 12)
    res = {}
    for centred in (False, True):
        X = E - E.mean(axis=0) if centred else E
        prof = embedding.profile(X)
        entry = {"effective_rank": prof["effective_rank"], "powerlaw_alpha": prof["powerlaw_alpha"],
                 "anisotropy": prof["anisotropy"]}
        for gname, g in graphs.items():
            Xg = X[g["lcc"]]
            L, S = g["L"], g["S"]
            groups = lengths[g["lcc"]]
            kk = min(k, Xg.shape[1], S.eigenvectors.shape[1] - 1)
            U = np.linalg.svd(Xg, full_matrices=False)[0][:, :kk]
            G = S.eigenvectors[:, 1 : 1 + kk]

            def align(Urows, G=G):
                return float(np.mean(np.linalg.svd(Urows.T @ G, compute_uv=False) ** 2))

            def dirichlet(M, L=L):
                return alignment.dirichlet_energy(M, L)

            for null, grp in (("uniform", None), ("length", groups)):
                rng = np.random.default_rng(seed)
                entry[f"{gname}/dirichlet/{null}"] = alignment.permutation_test(dirichlet, Xg, n_perm, rng, grp)
                rng = np.random.default_rng(seed)
                entry[f"{gname}/align@{kk}/{null}"] = alignment.permutation_test(align, U, n_perm, rng, grp)
        res["centred" if centred else "raw"] = entry
    return res


def print_matrix_report(label: str, rep: dict) -> None:
    for mode, entry in rep.items():
        print(f"  [{label}, {mode}] eff.rank={entry['effective_rank']:.1f}  "
              f"alpha={entry['powerlaw_alpha']:.2f}  anisotropy={entry['anisotropy']:.3f}")
        for key, t in entry.items():
            if isinstance(t, dict):
                print(f"    {key:<26} obs={t['observed']:.4f}  null={t['null_mean']:.4f}"
                      f"±{t['null_std']:.4f}  ratio={t['ratio']:.3f}  z={t['z']:+.1f}")


def run(label: str, spec: TokenizerSpec, matrices: dict, args) -> dict:
    t0 = time.time()
    graphs = graph_report(spec, args.k)
    print(f"\n=== {label}: V={len(spec)}  ({time.time() - t0:.0f}s graphs+spectra)")
    for g, v in graphs.items():
        s = v["summary"]
        print(f"  {g:<8} LCC={s['vertices']}  edges={s['edges']}  lambda_2={s['lambda_2']:.5f}")
    out = {"graphs": {g: v["summary"] for g, v in graphs.items()}, "matrices": {}}
    for mname, E in matrices.items():
        if E.shape[0] < len(spec):
            raise ValueError(f"{mname} has {E.shape[0]} rows < vocab {len(spec)}")
        E = E[: len(spec)]  # drop padding rows beyond the vocabulary
        rep = probe_matrix(E, spec, graphs, args.k, args.perms, args.seed)
        print_matrix_report(mname, rep)
        out["matrices"][mname] = rep
    return out


def selftest(args) -> dict:
    """Offline run: our BPE + a briefly trained bigram-LM embedding."""
    from tokenix import Encoder, bigram_counts, train_bpe
    from tokenix.corpus import stdlib_docstrings
    from tokenix.lm import BigramLM

    text = stdlib_docstrings()
    spec = train_bpe(text, 768)
    ids = Encoder(spec).encode(text)
    rng = np.random.default_rng(0)
    m = BigramLM.init(0.1 * rng.standard_normal((len(spec), 32)), rng)
    m.fit(bigram_counts(ids, len(spec)), steps=200, lr=0.02, log_every=200)
    return {"selftest": run("selftest (local BPE, bigram LM)", spec, {"E": m.E}, args)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=GPT2)
    ap.add_argument("--k", type=int, default=64, help="subspace dimension / eigenpairs")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/pretrained_probe.json"))
    args = ap.parse_args()

    if args.selftest:
        results = selftest(args)
    else:
        from tokenix.hub import fetch_file, load_embeddings
        from tokenix.io import load_hf_tokenizer_json

        results, spec_cache = {}, {}
        for repo in args.models:
            tok_path = fetch_file(repo, "tokenizer.json")
            key = tok_path.read_bytes()
            spec = spec_cache.setdefault(key, load_hf_tokenizer_json(tok_path))
            E_in, E_out = load_embeddings(repo)
            mats = {"E_in": E_in} if E_out is None else {"E_in": E_in, "E_out": E_out}
            results[repo] = run(repo, spec, mats, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
