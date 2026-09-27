"""Load external tokenizers into a :class:`TokenizerSpec`.

Supports Hugging Face ``tokenizer.json`` files with a BPE model (GPT-2, Llama 3,
Mistral, ... style).  Only ``json`` from the standard library is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .spec import TokenizerSpec


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode`` table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_hf_tokenizer_json(path: str | Path) -> TokenizerSpec:
    """Read a Hugging Face ``tokenizer.json`` BPE model.

    Token strings are converted back to raw bytes when the tokenizer uses the
    GPT-2 byte-level alphabet; otherwise they are UTF-8 encoded (``▁`` kept).
    Merges whose result is not in the vocabulary are skipped, and token ids are
    those of the file.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = data["model"]
    if model.get("type", "BPE") != "BPE":
        raise ValueError(f"unsupported tokenizer model: {model.get('type')}")
    vocab: dict[str, int] = model["vocab"]

    decoder = _byte_decoder()
    byte_level = all(ch in decoder for tok in vocab for ch in tok)

    def to_bytes(tok: str) -> bytes:
        return bytes(decoder[ch] for ch in tok) if byte_level else tok.encode("utf-8")

    # Added tokens live outside the BPE vocab (e.g. GPT-NeoX's whitespace runs at
    # ids 50254+); their ``content`` is plain text, not the byte-level alphabet.
    added = {t["id"]: t["content"].encode("utf-8") for t in data.get("added_tokens", [])
             if t["id"] not in vocab.values()}
    size = max([*vocab.values(), *added]) + 1
    tokens: list[bytes | None] = [None] * size
    for tok, i in vocab.items():
        tokens[i] = to_bytes(tok)
    first = {t: i for i, t in enumerate(tokens) if t is not None}
    aliases: dict[int, int] = {}
    for i, b in sorted(added.items()):
        if b in first:  # duplicate string: keep a placeholder and record the alias
            aliases[i] = first[b]
            tokens[i] = b"<alias:" + b + b">"
        else:
            tokens[i] = first[b] = b
    # Fill id holes (reserved / special slots) with unique placeholders.
    synthetic = set(aliases)
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"<unused_{i}>".encode()
            synthetic.add(i)

    merges = []
    for m in model.get("merges", []):
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        c = a + b
        if a in vocab and b in vocab and c in vocab:
            merges.append((vocab[a], vocab[b], vocab[c]))
    return TokenizerSpec(tokens, merges, aliases, synthetic)  # type: ignore[arg-type]
