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
  hub.py          stdlib-only Hub access; range-reads a single tensor out of safetensors
  graphs.py       merge_graph, containment_graph (substring Hasse diagram), transition_graph
  spectral.py     Laplacians, eigendecomposition, Fiedler value/vector, heat trace
  embedding.py    SVD / polar decomposition, effective & stable rank, power-law α, anisotropy
  alignment.py    Dirichlet energy, graph-frequency profile, subspace alignment
  constraints.py  spectrum projection, Laplacian-eigenmap init, Dirichlet gradient
  lm.py           tied-embedding bigram LM (NumPy, full-batch Adam) as a controlled testbed
  corpus.py       offline corpus: Python stdlib docstrings
experiments/
  spectral_probe.py   first end-to-end probe of H1–H4 (toy BPE + bigram LM)
  pretrained_probe.py H1/H2 on pretrained models (GPT-2 small → XL by default)
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

## Pretrained models (`experiments/pretrained_probe.py`)

The script starts with the GPT-2 family: small, medium, large and XL. All four
share one tokenizer and tie their input and output embeddings, so model scale
varies while the graph stays fixed. Llama-style models with untied embeddings
are the next step; the script then probes `E_in` and `E_out` separately.

The script downloads `tokenizer.json` and **only the embedding tensor(s)**,
using HTTP range requests into `model.safetensors`. It does not download the
full checkpoint. Files are cached in `~/.cache/tokenix`, which can be
overridden with `TOKENIX_CACHE`. Set `HF_TOKEN` for gated repositories.

For the merge DAG and the substring poset, restricted to the largest connected
component, it reports:

* the **Dirichlet ratio**: the Dirichlet energy of `E` divided by its mean under
  random vertex relabelling. A ratio below 1 means graph neighbours have more
  similar embeddings than chance;
* the **subspace alignment** of the top-k singular vectors with the k smoothest
  Laplacian modes, against the same relabelling null.

There are two nulls. The *uniform* null relabels vertices freely. The *length*
null only swaps tokens of equal byte length, which rules out token length as
the explanation. Every figure is also computed for the mean-centred `E`.
Calibration: a random Gaussian `E` on an 11k-token vocabulary gives z ≈ 0.

```bash
python experiments/pretrained_probe.py --selftest          # offline sanity run
python experiments/pretrained_probe.py                     # needs huggingface.co access
python experiments/pretrained_probe.py --models meta-llama/Llama-3.2-1B   # HF_TOKEN
```

## GPT-2 results (H1/H2)

These results come from `python experiments/pretrained_probe.py`: 20 relabellings
per null, k = 64. Each model takes about 5 minutes on CPU, and the graph
spectra are computed once and shared.

**The tokenizer graphs.** GPT-2 has 50,000 merges over 50,257 tokens. Each
graph's largest connected component holds all but ~67 tokens. Both graphs are
very weakly connected: λ₂ = 0.0016 for the merge DAG and 0.0012 for the
substring poset, both nearly tree-like with byte-token hubs.

**Embeddings respect both graphs, by a constant amount at every model size.**
The table uses the mean-centred `E` and the length-stratified null.

| model | params | eff. rank | anisotropy (raw) | Dirichlet ratio: merge | Dirichlet ratio: poset | alignment@64: poset (null) |
|-------|-------:|---------:|----------------:|-----------------------:|-----------------------:|---------------------------:|
| gpt2 | 124M | 714 / 768 | 0.268 | 0.873 | 0.779 | 0.044 (0.005) |
| gpt2-medium | 355M | 967 / 1024 | 0.308 | 0.873 | 0.777 | 0.048 (0.005) |
| gpt2-large | 774M | 1183 / 1280 | 0.082 | 0.869 | 0.773 | 0.048 (0.005) |
| gpt2-xl | 1.5B | 1490 / 1600 | 0.079 | 0.874 | 0.780 | 0.049 (0.005) |

Mean cosine similarity of centred embeddings across each kind of graph edge.
The substring-poset edges are split by how the shorter token sits in the
longer one. Length-matched random pairs give ≈ 0.00–0.01 for every row.

| edge kind | edges | gpt2 | medium | large | xl |
|-----------|------:|-----:|-------:|------:|---:|
| space variant (`the` → ` the`) | 8,533 | 0.549 | 0.559 | 0.531 | 0.525 |
| prefix (`under` → `underst`) | 49,996 | 0.288 | 0.295 | 0.316 | 0.305 |
| suffix (`ing` → `ting`) | 41,122 | 0.144 | 0.136 | 0.134 | 0.127 |
| infix | 11,219 | 0.014 | 0.015 | 0.013 | 0.013 |
| merge DAG (parent–child) | 99,756 | 0.171 | 0.168 | 0.178 | 0.171 |

What these results show:

* **H2 holds locally.** Across every model, neighbouring tokens in the
  substring poset have ~22% less embedding variation than random pairs of the
  same length (Dirichlet ratio ≈ 0.78). The merge DAG gives ≈ 0.87, so the
  poset, which is *order* structure, is the better description of embedding
  geometry than the merge *history*. The z-scores are in the hundreds, but
  with 50k vertices the effect sizes above are what matter.
* **The effect is scale-invariant.** The ratio barely moves over a 12× range of
  parameters, while effective rank doubles and raw anisotropy drops ~4×. The
  tokenizer's order structure takes up a fixed share of the embedding's
  geometry, independent of how much else the model packs in.
* **Only word-boundary structure is used.** Space-variant and prefix edges
  carry the signal, suffix edges carry about half as much, and infix edges are
  at chance. The model encodes where a token *starts* far more than what it
  *contains*. The substring poset is too coarse a graph: a poset of prefixes
  (the trie) may be the natural object instead.
* **Global structure is weak.** The top-64 singular directions overlap the 64
  smoothest Laplacian modes at only ~0.05, about 10× the length null but small
  in absolute terms. The dominant stretch directions of `E` are not the
  graph's global modes. The structure lives in local neighbourhoods, which is
  exactly what a Dirichlet-type constraint acts on.

Caveats: token frequency is not controlled; under-trained "glitch" tokens could
shift the numbers; and all four models share one tokenizer and one training
corpus.

## Pythia results: untied input and output embeddings

Pythia (EleutherAI) uses the GPT-NeoX tokenizer, which is also a byte-level BPE
with 50k tokens. It keeps **separate** input (`embed_in`) and output
(`embed_out`) matrices, and all sizes share one tokenizer and one training set.
The run is `python experiments/pretrained_probe.py --models EleutherAI/pythia-160m ...`.
All figures use the mean-centred embeddings and the length-stratified null.

| model | poset ratio: in / out | merge ratio: in / out | prefix cos: in / out | suffix cos: in / out | space-variant cos: in / out |
|-------|------------------------|-----------------------|----------------------|----------------------|-----------------------------|
| pythia-160m | 0.788 / **0.706** | 0.893 / 0.811 | 0.21 / 0.38 | 0.17 / 0.16 | 0.56 / 0.49 |
| pythia-410m | 0.778 / 0.770 | 0.889 / 0.882 | 0.22 / 0.30 | 0.18 / 0.11 | 0.58 / 0.58 |
| pythia-1.4b | 0.783 / 0.777 | 0.890 / 0.890 | 0.21 / 0.28 | 0.18 / 0.11 | 0.55 / 0.59 |
| pythia-2.8b | 0.782 / 0.777 | 0.889 / 0.892 | 0.21 / 0.27 | 0.18 / 0.10 | 0.54 / 0.61 |

Infix edges stay at chance (≈ 0.01–0.02) and the length-matched nulls are
≤ 0.02 throughout.

* **The ~0.78 poset ratio holds across 8 models.** It appears for both
  tokenizers, for tied (GPT-2) and untied (Pythia) embeddings, and from 124M to
  2.8B parameters. Only Pythia-160m's output matrix is noticeably smoother.
* **Input and output embeddings share the ratio but not its make-up.** Output
  embeddings are more prefix-driven (0.27–0.30 vs 0.21) and less suffix-driven
  (0.10–0.11 vs 0.18). One hypothesis, not yet tested: the output matrix scores
  candidate next tokens, and tokens that share a beginning compete for the same
  continuation. The input matrix encodes content, where morphology at the end
  of a word (`-ing`, `-ed`) matters.

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

* Run H1/H2 on Llama 3 (128k vocabulary; 3.1-8B has untied embeddings) and
  other tokenizers and vocabulary sizes.
* Track the ratio over Pythia's training checkpoints to see when the structure
  appears.
* Add a prefix-trie graph and a frequency-stratified null, since word-boundary
  edges carry the GPT-2 signal.
* Test on a transformer rather than a bigram model, where input and output
  geometry can differ (tied vs. untied embeddings).
* Treat `Σ` as a hyperparameter family: learnable-but-regularised spectra and
  spectral normalisation of `E` only.
* Build graph-aware tokenizers, choosing merges to shape the vocabulary graph's
  spectrum (for example, maximising λ₂).
* Scale-matched controls for every structural claim: shuffled graphs,
  degree-preserving rewiring, and plain L2.
