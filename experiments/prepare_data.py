"""Tokenize a slice of FineWeb-Edu with the GPT-2 tokenizer into uint16 .bin files.

    python experiments/prepare_data.py --out data/fineweb_gpt2 --train_tokens 500_000_000

Writes ``train.bin`` / ``val.bin`` (flat uint16 token ids, documents separated by
``<|endoftext|>``) and ``meta.json``.  Streams the dataset, so only the text
actually used is downloaded.  Needs ``pip install datasets tokenizers``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/fineweb_gpt2"))
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--tokenizer", default="openai-community/gpt2")
    ap.add_argument("--train_tokens", type=int, default=500_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--batch", type=int, default=1000, help="documents per encode_batch call")
    args = ap.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    assert tok.get_vocab_size() < 2**16

    args.out.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(args.dataset, name=args.config, split="train", streaming=True)

    # Validation first (the first documents of the stream), then training.
    targets = [("val.bin", args.val_tokens), ("train.bin", args.train_tokens)]
    it = iter(stream)
    exhausted = False
    for fname, budget in targets:
        path = args.out / fname
        written = 0
        with open(path, "wb") as f:
            while written < budget and not exhausted:
                texts = []
                for _ in range(args.batch):
                    try:
                        texts.append(next(it)["text"])
                    except StopIteration:
                        exhausted = True
                        break
                if not texts:
                    break
                ids = []
                for enc in tok.encode_batch(texts):
                    ids.extend(enc.ids)
                    ids.append(eot)
                arr = np.asarray(ids[: budget - written], dtype=np.uint16)
                f.write(arr.tobytes())
                written += len(arr)
                if fname == "train.bin" and written // 50_000_000 != (written - len(arr)) // 50_000_000:
                    print(f"  {written / 1e6:.0f}M tokens", flush=True)
        print(f"{fname}: {written:,} tokens")

    (args.out / "meta.json").write_text(json.dumps({
        "dataset": args.dataset, "config": args.config, "tokenizer": args.tokenizer,
        "vocab_size": tok.get_vocab_size(), "eot": eot, "dtype": "uint16",
    }, indent=1))


if __name__ == "__main__":
    main()
