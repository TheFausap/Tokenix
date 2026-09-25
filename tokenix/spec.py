"""A tokenizer seen as a combinatorial object.

A (BPE-style) tokenizer is a finite alphabet ``V`` of strings together with a
set of *merge rules* ``(a, b) -> c`` with ``c = a + b``.  Encoding is a map from
the free monoid over bytes onto words over ``V``; the vocabulary itself carries
two natural orders:

* the **merge order** -- the DAG in which ``c`` has parents ``a`` and ``b``;
* the **containment order** -- the poset of tokens under the substring relation.

:class:`TokenizerSpec` is the minimal structure every analysis in this package
consumes, so learned tokenizers (:mod:`tokenix.bpe`) and external ones
(:mod:`tokenix.io`) can be studied with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenizerSpec:
    tokens: list[bytes]
    merges: list[tuple[int, int, int]] = field(default_factory=list)
    """Merge rules ``(left_id, right_id, merged_id)`` in priority order."""

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.tokens)}
        if len(self.index) != len(self.tokens):
            raise ValueError("duplicate tokens in vocabulary")
        for a, b, c in self.merges:
            if self.tokens[a] + self.tokens[b] != self.tokens[c]:
                raise ValueError(f"merge {a}+{b}->{c} is not a concatenation")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def base_ids(self) -> list[int]:
        """Tokens that are not produced by any merge (the base alphabet)."""
        produced = {c for _, _, c in self.merges}
        return [i for i in range(len(self.tokens)) if i not in produced]

    def depth(self) -> list[int]:
        """Height of each token in the merge DAG (base tokens have depth 0)."""
        d = [0] * len(self.tokens)
        for a, b, c in self.merges:
            d[c] = max(d[c], 1 + max(d[a], d[b]))
        return d
