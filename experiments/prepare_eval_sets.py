"""Out-of-domain validation sets, tokenized with the GPT-2 tokenizer.

    python experiments/prepare_eval_sets.py --out data/eval --tokens 2000000

Writes ``<out>/<name>.bin`` (uint16 ids, documents separated by
``<|endoftext|>``) for each set below.  The training data (FineWeb-Edu) is
filtered English web text, so tokens that are rare in training are far more
common here: code, other languages, specialist prose.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# name: (dataset, config, split, text field)
SETS = {
    "code_python": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    "wiki_it": ("wikimedia/wikipedia", "20231101.it", "train", "text"),
    "wiki_de": ("wikimedia/wikipedia", "20231101.de", "train", "text"),
    "wiki_ru": ("wikimedia/wikipedia", "20231101.ru", "train", "text"),
    "pubmed": ("ccdv/pubmed-summarization", "document", "validation", "article"),
    "arxiv": ("ccdv/arxiv-summarization", "document", "validation", "article"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/eval"))
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--tokenizer", default="openai-community/gpt2")
    args = ap.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.sets:
        path = args.out / f"{name}.bin"
        if path.exists():
            print(f"{name}: exists")
            continue
        ds, cfg, split, field = SETS[name]
        it = iter(load_dataset(ds, name=cfg, split=split, streaming=True))
        ids: list[int] = []
        while len(ids) < args.tokens:
            texts = [ex[field] for _, ex in zip(range(200), it)]
            if not texts:
                break
            for enc in tok.encode_batch(texts):
                ids.extend(enc.ids)
                ids.append(eot)
        arr = np.asarray(ids[: args.tokens], dtype=np.uint16)
        arr.tofile(path)
        print(f"{name}: {len(arr):,} tokens from {ds}" + (f" ({cfg})" if cfg else ""))


if __name__ == "__main__":
    main()
    # Streaming `datasets` iterators leave background threads that can abort the
    # interpreter at shutdown ("PyGILState_Release"); all files are written by now.
    sys.stdout.flush()
    os._exit(0)
