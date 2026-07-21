"""Stream-tokenize a slice of fineweb-edu (sample-10BT) into uint16 bins.

Writes train.bin (~target_tokens) and val.bin (~5M tokens) to --out.
CPU-only; safe to run alongside GPU training.
"""
import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--target_tokens", type=float, default=2.0e9)
p.add_argument("--val_tokens", type=float, default=5e6)
args = p.parse_args()

tok = GPT2TokenizerFast.from_pretrained("gpt2")
eot = tok.eos_token_id
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                  split="train", streaming=True)

train_f = open(f"{args.out}/train.bin", "wb")
val_f = open(f"{args.out}/val.bin", "wb")
n_train = n_val = 0
buf = []
for i, ex in enumerate(ds):
    buf.append(ex["text"])
    if len(buf) == 500:
        for ids in tok(buf, add_special_tokens=False)["input_ids"]:
            ids.append(eot)
            arr = np.array(ids, dtype=np.uint16)
            if n_val < args.val_tokens:
                arr.tofile(val_f); n_val += len(arr)
            else:
                arr.tofile(train_f); n_train += len(arr)
        buf = []
        if n_train >= args.target_tokens:
            break
        if (i + 1) % 50000 == 0:
            print(f"{i+1} docs, train {n_train/1e9:.2f}B val {n_val/1e6:.1f}M", flush=True)
train_f.close(); val_f.close()
print(f"DONE train {n_train/1e9:.3f}B tokens, val {n_val/1e6:.1f}M tokens", flush=True)
