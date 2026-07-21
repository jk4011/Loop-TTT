"""Tokenize WikiText-103 into flat uint16 bins for the loop-LM transfer test.

Writes train.bin / val.bin (GPT-2 BPE ids, uint16) to --out (lustre).
"""
import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
args = p.parse_args()

tok = GPT2TokenizerFast.from_pretrained("gpt2")
ds = load_dataset("wikitext", "wikitext-103-raw-v1")

for split, name in [("train", "train"), ("validation", "val")]:
    ids = []
    batch = []
    for line in ds[split]["text"]:
        batch.append(line)
        if len(batch) == 2000:
            for out in tok("".join(batch), add_special_tokens=False)["input_ids"],:
                ids.extend(out)
            batch = []
    if batch:
        ids.extend(tok("".join(batch), add_special_tokens=False)["input_ids"])
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(f"{args.out}/{name}.bin")
    print(f"{name}: {len(arr)/1e6:.1f}M tokens -> {args.out}/{name}.bin", flush=True)
