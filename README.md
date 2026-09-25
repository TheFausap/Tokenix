# Tokenix

**Tokenizers as graphs, embedding matrices as rotation × stretch.**

Tokenix is a research toolkit built on two lenses that are rarely applied to
tokenization:

1. **Matrices are graphs.** A tokenizer is a map from the free monoid over
   bytes onto words over a finite alphabet `V`. That alphabet is an
   order-theoretic and combinatorial object. BPE merges form a DAG, tokens are
   ordered by the substring relation, and a corpus induces a co-occurrence graph
   on them. Spectral graph theory (Laplacian eigenvalues and eigenvectors) gives
   these structures a language.
2. **Every matrix is a rotation and a stretch.** The embedding matrix
   `E ∈ R^{V×d}` factors as `E = U Σ Vᵀ`, or in polar form as `E = Q P`. It is
   the one weight matrix of a language model whose spectrum nobody constrains,
   even though its rows are indexed by the vertices of the graphs above.

The meeting point is this: **the columns of `E` are signals on the tokenizer's
graphs.** We can ask whether the "rotation" `U` respects the tokenizer's
combinatorial structure, and whether the "stretch" `Σ` should be shaped
deliberately.

## Research questions

| # | Question | Tokenix instrument |
|---|----------|--------------------|
| H1 | Do tokenizer graphs (merge DAG, substring poset) have characteristic spectra, and do these spectra differ across tokenizers, vocabulary sizes and languages? | `graphs`, `spectral.Spectrum` (λ₂, heat trace, spectral density) |
| H2 | Do *learned* embeddings align with the smooth Laplacian modes of the tokenizer's own graphs, beyond what corpus statistics explain? | `alignment.subspace_alignment`, `alignment.frequency_profile`, `alignment.dirichlet_energy` |
| H3 | Does constraining the embedding's stretch `Σ` (flat or power-law) help? | `constraints.project_spectrum` |
| H4 | Does constraining its rotation `U` toward the tokenizer graph help, through a Laplacian-eigenmap init or a Dirichlet-energy penalty? | `constraints.graph_spectral_init`, `constraints.dirichlet_grad` |

## Layout

```
tokenix/
  spec.py         TokenizerSpec: vocabulary + merge rules (the combinatorial object)
  bpe.py          dependency-free byte-level BPE trainer / encoder
  io.py           load Hugging Face tokenizer.json (BPE) files
  graphs.py       merge_graph, containment_graph (substring Hasse diagram), transition_graph
  spectral.py     Laplacians, eigendecomposition, Fiedler value/vector, heat trace
  embedding.py    SVD / polar decomposition, effective & stable rank, power-law α, anisotropy
  alignment.py    Dirichlet energy, graph-frequency profile, subspace alignment
  constraints.py  spectrum projection, Laplacian-eigenmap init, Dirichlet gradient
  lm.py           tied-embedding bigram LM (NumPy, full-batch Adam) as a controlled testbed
  corpus.py       offline corpus: Python stdlib docstrings
experiments/
  spectral_probe.py   first end-to-end probe of H1–H4
tests/
```

## Quick start

```bash
pip install -e ".[dev]"
pytest
python experiments/spectral_probe.py          # ~2.5 min on CPU
```

Analyse a real tokenizer and embedding matrix:

```python
import numpy as np
from tokenix import containment_graph, spectrum
from tokenix.io import load_hf_tokenizer_json
from tokenix import alignment, embedding

tok = load_hf_tokenizer_json("tokenizer.json")      # e.g. GPT-2
S = spectrum(containment_graph(tok), k=256)          # sparse: 256 smoothest modes
E = np.load("wte.npy")                               # the model's (V, d) embedding
print(embedding.profile(E)["effective_rank"])
print(alignment.subspace_alignment(E, S, k=64, skip=S.n_components))
```

## First results (`experiments/spectral_probe.py`)

Setup: a BPE tokenizer with 768 merges trained on ~480 KB of stdlib docstrings,
giving `V = 870`. A tied bigram LM with `d = 32` is trained for 400 Adam steps,
3 seeds, and evaluated on held-out docstrings (every 10th paragraph).

**Graph spectra (normalised Laplacian)**

| graph | edges | components | λ₂ |
|-------|------:|-----------:|---:|
| merge DAG | 1514 | 33 | 0.022 |
| substring poset (Hasse) | 1558 | 33 | 0.014 |
| PPMI bigram graph | 22212 | 7 | 0.168 |

The two tokenizer-intrinsic graphs are sparse and nearly tree-like, with small
spectral gaps. Most of the 33 components are rare bytes that are never merged.

**Embedding conditions** (test NLL in nats, mean ± std over 3 seeds; alignment is
the mean cos² of the principal angles for k = 32, chance ≈ 0.037)

| condition | test NLL | eff. rank | α | anisotropy | align: poset | align: bigram |
|-----------|---------:|---------:|--:|-----------:|------------:|-------------:|
| random | 4.076 ± .002 | 30.96 | 0.25 | 0.124 | 0.068 | 0.209 |
| proj: flat Σ | 4.067 ± .002 | 32.00 | 0.00 | 0.037 | 0.066 | 0.211 |
| proj: power-law Σ (α = 0.5) | 4.080 ± .004 | 28.32 | 0.50 | 0.243 | 0.070 | 0.219 |
| eigenmap init: merge DAG | 4.069 ± .002 | 30.97 | 0.25 | 0.122 | 0.070 | 0.209 |
| eigenmap init: substring poset | 4.079 ± .002 | 30.97 | 0.25 | 0.125 | 0.069 | 0.212 |
| eigenmap init: bigram graph | 4.073 ± .003 | 30.96 | 0.26 | 0.110 | 0.065 | 0.236 |
| **Dirichlet penalty: substring poset** | **4.035 ± .005** | 30.35 | 0.30 | 0.169 | 0.108 | 0.234 |
| Dirichlet penalty: *shuffled* poset (control) | 4.043 ± .004 | 30.75 | 0.23 | 0.168 | 0.079 | 0.216 |

Reading these results with caution (a toy model on a small corpus):

* **H2:** freely learned embeddings align with the corpus bigram graph (0.21,
  about 6× chance). They barely align with the tokenizer's substring poset
  (0.07, about 2× chance). Left alone, the embedding does not discover the
  tokenizer's order structure.
* **H3:** a flat stretch (isotropic `Σ`) gives a small, consistent gain and cuts
  anisotropy by 3×. A power-law stretch slightly hurts.
* **H4:** initialisation effects wash out under Adam. The Dirichlet penalty on
  the substring poset gives the best test NLL, but **most of that gain is
  generic regularisation**: the label-shuffled control, which has an identical
  spectrum but meaningless structure, recovers about 80% of it. The
  structure-specific part is about 0.008 nats, roughly 2σ. This is a lead worth
  scaling up, not a result yet.

## Next steps

* Run H1/H2 on real tokenizers and models (GPT-2, Llama-3, Mistral): compare
  the spectra of the merge DAG and the substring poset across vocabulary sizes.
  Measure how pretrained `wte` matrices align with them.
* Test on a transformer rather than a bigram model, where input and output
  geometry can differ (tied vs. untied embeddings).
* Treat `Σ` as a hyperparameter family: learnable-but-regularised spectra and
  spectral normalisation of `E` only.
* Build graph-aware tokenizers, choosing merges to shape the vocabulary graph's
  spectrum (for example, maximising λ₂).
* Scale-matched controls for every structural claim: shuffled graphs,
  degree-preserving rewiring, and plain L2.
