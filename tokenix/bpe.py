"""A small, dependency-free byte-level BPE trainer and encoder.

It exists so experiments can control the tokenizer end to end (vocabulary size,
corpus, merge order) without pulling in an external tokenizer library.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .spec import TokenizerSpec

# GPT-2 style pre-tokenization, simplified: words keep their leading space.
_PRETOKEN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def pretokenize(text: str) -> list[bytes]:
    return [m.group().encode("utf-8") for m in _PRETOKEN.finditer(text)]


def _pairs(word: tuple[int, ...]) -> Counter:
    return Counter(zip(word, word[1:]))


def train_bpe(text: str, num_merges: int, alphabet: bytes | None = None) -> TokenizerSpec:
    """Learn ``num_merges`` BPE merges on ``text``.

    The base alphabet is the set of bytes occurring in ``alphabet`` (default:
    in ``text``), so the vocabulary has no unused byte tokens.
    """
    raw = text.encode("utf-8") if alphabet is None else alphabet
    tokens = [bytes([b]) for b in sorted(set(raw))]
    index = {t: i for i, t in enumerate(tokens)}

    counts = Counter(pretokenize(text))
    words = [tuple(index[bytes([b])] for b in w) for w in counts]
    freqs = list(counts.values())

    pair_counts: Counter = Counter()
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        for p, n in _pairs(w).items():
            pair_counts[p] += n * freqs[wi]
            where[p].add(wi)

    merges: list[tuple[int, int, int]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        # Deterministic tie-break on the pair ids.
        (a, b), n = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if n <= 0:
            break
        c = len(tokens)
        tokens.append(tokens[a] + tokens[b])
        merges.append((a, b, c))

        for wi in list(where[(a, b)]):
            old = words[wi]
            new, i = [], 0
            while i < len(old):
                if i + 1 < len(old) and old[i] == a and old[i + 1] == b:
                    new.append(c)
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            new = tuple(new)
            for p, k in _pairs(old).items():
                pair_counts[p] -= k * freqs[wi]
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                where[p].discard(wi)
            for p, k in _pairs(new).items():
                pair_counts[p] += k * freqs[wi]
                where[p].add(wi)
            words[wi] = new
        pair_counts.pop((a, b), None)
        where.pop((a, b), None)

    return TokenizerSpec(tokens, merges)


class Encoder:
    """Greedy rank-ordered BPE encoding with a per-word cache."""

    def __init__(self, spec: TokenizerSpec):
        self.spec = spec
        self.rank = {(a, b): r for r, (a, b, _) in enumerate(spec.merges)}
        self.result = {(a, b): c for a, b, c in spec.merges}
        self._cache: dict[bytes, list[int]] = {}

    def encode_word(self, word: bytes) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        try:
            ids = [self.spec.index[bytes([b])] for b in word]
        except KeyError as e:
            raise ValueError(f"byte {e} not in the tokenizer alphabet") from None
        while len(ids) > 1:
            ranked = [(self.rank.get(p, 1 << 62), i) for i, p in enumerate(zip(ids, ids[1:]))]
            r, i = min(ranked)
            if r == 1 << 62:
                break
            ids[i : i + 2] = [self.result[(ids[i], ids[i + 1])]]
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for w in pretokenize(text):
            out.extend(self.encode_word(w))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.spec.tokens[i] for i in ids).decode("utf-8", errors="replace")
