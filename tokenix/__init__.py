"""Tokenix: tokenizers as graphs, embedding matrices as rotation x stretch."""

from .spec import TokenizerSpec
from .bpe import Encoder, train_bpe
from .graphs import bigram_counts, containment_graph, merge_graph, transition_graph
from .spectral import Spectrum, laplacian, spectrum

__all__ = [
    "TokenizerSpec",
    "Encoder",
    "train_bpe",
    "bigram_counts",
    "containment_graph",
    "merge_graph",
    "transition_graph",
    "Spectrum",
    "laplacian",
    "spectrum",
]

__version__ = "0.1.0"
