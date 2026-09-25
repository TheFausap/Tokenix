"""Fetch tokenizers and single tensors from the Hugging Face Hub, stdlib only.

``load_tensor`` reads the safetensors header with an HTTP range request and then
downloads *only* the requested tensor's bytes, so pulling the embedding matrix
of a multi-GB checkpoint costs just ``V x d`` values.  Results are cached as
``.npy`` under ``cache_dir`` (default ``~/.cache/tokenix``).
"""

from __future__ import annotations

import http.client
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"
CACHE = Path(os.environ.get("TOKENIX_CACHE", Path.home() / ".cache" / "tokenix"))

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64}


def _request(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "tokenix"}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


CHUNK = 32 << 20  # bytes per range request
RETRIES = 4


def _read_range(url: str, start: int, stop: int) -> bytes:
    """Bytes ``[start, stop)`` in chunks, retrying truncated or failed transfers."""
    parts = []
    for a in range(start, stop, CHUNK):
        b = min(a + CHUNK, stop)
        for attempt in range(RETRIES + 1):
            try:
                data = _request(url, a, b - 1)
                if len(data) != b - a:
                    raise OSError(f"short read: {len(data)} of {b - a} bytes")
                break
            except (OSError, http.client.IncompleteRead):  # resets, timeouts, truncation
                if attempt == RETRIES:
                    raise
                time.sleep(2 ** (attempt + 1))
        parts.append(data)
    return b"".join(parts)


def _url(repo: str, filename: str, revision: str = "main") -> str:
    return f"{HF}/{repo}/resolve/{revision}/{filename}"


def fetch_file(repo: str, filename: str, cache_dir: Path = CACHE) -> Path:
    """Download a (small) file from a Hub repo into the cache, return its path."""
    path = cache_dir / repo.replace("/", "__") / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_request(_url(repo, filename)))
    return path


class SafetensorsSource:
    """A safetensors file, local or remote, read lazily by byte ranges."""

    def __init__(self, location: str | Path):
        self.location = str(location)
        self.remote = self.location.startswith(("http://", "https://"))
        (n,) = struct.unpack("<Q", self._read(0, 8))
        self.header = json.loads(self._read(8, 8 + n))
        self.header.pop("__metadata__", None)
        self.offset = 8 + n

    def _read(self, start: int, stop: int) -> bytes:
        if self.remote:
            return _read_range(self.location, start, stop)
        with open(self.location, "rb") as f:
            f.seek(start)
            return f.read(stop - start)

    def names(self) -> list[str]:
        return list(self.header)

    def load(self, name: str) -> np.ndarray:
        info = self.header[name]
        a, b = info["data_offsets"]
        raw = np.frombuffer(self._read(self.offset + a, self.offset + b), dtype=_DTYPES[info["dtype"]])
        if info["dtype"] == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32).reshape(info["shape"])


def safetensors_files(repo: str) -> list[str]:
    """The safetensors shard names of a Hub repo (sharded or single-file)."""
    try:
        index = json.loads(fetch_file(repo, "model.safetensors.index.json").read_text())
        return sorted(set(index["weight_map"].values()))
    except Exception:
        return ["model.safetensors"]


EMBEDDING_NAMES = ("wte.weight", "embed_tokens.weight", "word_embeddings.weight", "embed_in.weight")
UNEMBEDDING_NAMES = ("lm_head.weight", "embed_out.weight")


def load_tensor(repo: str, suffixes: tuple[str, ...], cache_dir: Path = CACHE) -> np.ndarray | None:
    """First tensor whose name ends with one of ``suffixes``; ``None`` if absent."""
    for filename in safetensors_files(repo):
        src = SafetensorsSource(_url(repo, filename))
        for name in src.names():
            if name.endswith(suffixes):
                path = cache_dir / repo.replace("/", "__") / f"{name}.npy"
                if path.exists():
                    return np.load(path)
                arr = src.load(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)
                return arr
    return None


def load_embeddings(repo: str, cache_dir: Path = CACHE) -> tuple[np.ndarray, np.ndarray | None]:
    """``(input_embedding, output_embedding_or_None_if_tied)``."""
    E = load_tensor(repo, EMBEDDING_NAMES, cache_dir)
    if E is None:
        raise KeyError(f"no embedding tensor found in {repo}")
    return E, load_tensor(repo, UNEMBEDDING_NAMES, cache_dir)
