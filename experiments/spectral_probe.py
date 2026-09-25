"""First probe: do tokenizer-graph spectra matter for the embedding matrix?

1. Train a byte-level BPE tokenizer on an offline corpus (stdlib docstrings).
2. Build its merge, containment and transition graphs; report their spectra.
3. Train a tied-embedding bigram LM under different embedding conditions:
     random            -- Gaussian init, unconstrained
     proj:flat         -- random init, stretch projected to a flat spectrum each step
     proj:powerlaw     -- random init, stretch projected to s_i ~ i^-0.5
     eigmap:merge      -- init from Laplacian eigenmap of the merge DAG
     eigmap:contain    -- init from Laplacian eigenmap of the substring poset
     eigmap:transition -- init from Laplacian eigenmap of the PPMI bigram graph
     dirichlet:contain -- random init + tr(E^T L E) penalty on the substring poset
     dirichlet:shuffled-- same penalty on a vertex-permuted copy of the poset
                          (identical spectrum, meaningless structure: the control)
4. Report held-out NLL and the learned embedding's spectral profile / alignment.

Usage:  python experiments/spectral_probe.py [--merges 768] [--dim 32] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenix import Encoder, bigram_counts, containment_graph, merge_graph, spectrum, train_bpe  # noqa: E402
from tokenix import alignment, constraints, embedding  # noqa: E402
from tokenix.corpus import stdlib_docstrings  # noqa: E402
from tokenix.graphs import transition_graph  # noqa: E402
from tokenix.spectral import laplacian  # noqa: E402
from tokenix.lm import BigramLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=768)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--dirichlet", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=Path("results/spectral_probe.json"))
    args = ap.parse_args()

    t0 = time.time()
    text = stdlib_docstrings()
    docs = text.split("\n\n")
    train_text = "\n\n".join(d for i, d in enumerate(docs) if i % 10)
    test_text = "\n\n".join(d for i, d in enumerate(docs) if not i % 10)

    tok = train_bpe(train_text, args.merges, alphabet=text.encode("utf-8"))
    enc = Encoder(tok)
    train_ids, test_ids = enc.encode(train_text), enc.encode(test_text)
    V, d = len(tok), args.dim
    C_train, C_test = bigram_counts(train_ids, V), bigram_counts(test_ids, V)
    print(f"vocab={V}  train_tokens={len(train_ids)}  test_tokens={len(test_ids)}  ({time.time()-t0:.1f}s)")

    graphs = {
        "merge": merge_graph(tok),
        "contain": containment_graph(tok),
        "transition": transition_graph(train_ids, V, weighting="ppmi"),
    }
    spectra = {k: spectrum(W) for k, W in graphs.items()}
    graph_report = {}
    print("\n== tokenizer graph spectra (normalised Laplacian) ==")
    print(f"{'graph':<11}{'edges':>8}{'comps':>7}{'lambda_2':>10}{'heat tr(t=1)/V':>16}")
    for k, s in spectra.items():
        rep = {
            "edges": int(graphs[k].nnz // 2),
            "components": s.n_components,
            "algebraic_connectivity": s.algebraic_connectivity,
            "heat_trace_t1_over_V": s.heat_trace(1.0) / V,
        }
        graph_report[k] = rep
        print(f"{k:<11}{rep['edges']:>8}{rep['components']:>7}{rep['algebraic_connectivity']:>10.4f}"
              f"{rep['heat_trace_t1_over_V']:>16.4f}")

    Ls = spectra
    L_contain = laplacian(graphs["contain"])
    perm = np.random.default_rng(12345).permutation(V)
    L_shuffled = laplacian(graphs["contain"][perm][:, perm])
    init_std = 0.1
    fro = init_std * np.sqrt(V * d)

    def make(cond: str, rng: np.random.Generator):
        kw: dict = {}
        if cond.startswith("eigmap:"):
            g = cond.split(":")[1]
            s = Ls[g]
            E0 = constraints.graph_spectral_init(s, d, rng, skip=s.n_components, fro=fro)
        else:
            E0 = init_std * rng.standard_normal((V, d))
        if cond == "proj:flat":
            kw["project"] = lambda E: constraints.project_spectrum(E, constraints.flat_spectrum(d))
        if cond == "proj:powerlaw":
            kw["project"] = lambda E: constraints.project_spectrum(E, constraints.powerlaw_spectrum(d, 0.5))
        if cond == "dirichlet:contain":
            kw["reg_grad"] = lambda E: args.dirichlet * constraints.dirichlet_grad(E, L_contain)
        if cond == "dirichlet:shuffled":
            kw["reg_grad"] = lambda E: args.dirichlet * constraints.dirichlet_grad(E, L_shuffled)
        return E0, kw

    conds = ["random", "proj:flat", "proj:powerlaw", "eigmap:merge", "eigmap:contain",
             "eigmap:transition", "dirichlet:contain", "dirichlet:shuffled"]
    results: dict = {}
    print(f"\n== bigram LM, d={d}, {args.steps} steps, {args.seeds} seeds (test NLL in nats) ==")
    hdr = (f"{'condition':<19}{'test NLL':>14}{'train':>8}{'eff.rank':>9}{'alpha':>7}{'aniso':>7}"
           f"{'Dir(cont)':>10}{'align:cont':>11}{'align:trans':>12}")
    print(hdr)
    k_align = d
    for cond in conds:
        rows = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            E0, kw = make(cond, rng)
            model = BigramLM.init(E0, rng).fit(C_train, C_test, steps=args.steps, lr=args.lr,
                                               log_every=args.steps, **kw)
            prof = embedding.profile(model.E)
            rows.append({
                "test_nll": model.history[-1]["test_nll"],
                "train_nll": model.history[-1]["train_nll"],
                "effective_rank": prof["effective_rank"],
                "powerlaw_alpha": prof["powerlaw_alpha"],
                "anisotropy": prof["anisotropy"],
                "dirichlet_contain": alignment.dirichlet_energy(model.E, L_contain),
                "align_contain": alignment.subspace_alignment(model.E, Ls["contain"], k_align,
                                                              skip=Ls["contain"].n_components),
                "align_transition": alignment.subspace_alignment(model.E, Ls["transition"], k_align,
                                                                 skip=Ls["transition"].n_components),
                "singular_values": prof["singular_values"].tolist(),
            })
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
               for k in rows[0] if k != "singular_values"}
        results[cond] = {"runs": rows, "mean_std": agg}
        m = {k: v[0] for k, v in agg.items()}
        print(f"{cond:<19}{m['test_nll']:>8.4f}±{agg['test_nll'][1]:<5.3f}{m['train_nll']:>8.4f}{m['effective_rank']:>9.2f}"
              f"{m['powerlaw_alpha']:>7.2f}{m['anisotropy']:>7.3f}{m['dirichlet_contain']:>10.3f}"
              f"{m['align_contain']:>11.3f}{m['align_transition']:>12.3f}")
    print(f"(chance level for alignment ~ k/V = {k_align / V:.3f}; total {time.time()-t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                                    "vocab_size": V, "graphs": graph_report, "results": results},
                                   indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
