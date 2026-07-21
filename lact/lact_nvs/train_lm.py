"""Minimal LM trainer for the loop-TTT task-agnostic transfer test.

Single GPU, bf16, AdamW with the same optzone grouping as train.py.
Reports final validation loss/ppl to outputs/<exp>/eval_lm.json.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.optimization import get_cosine_schedule_with_warmup

from loop_lm import LoopLM

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--expname", required=True)
p.add_argument("--data_dir", required=True, help="dir with train.bin / val.bin (uint16)")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--seq_len", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--weight_decay", type=float, default=0.05)
p.add_argument("--loop_param_lr_mult", type=float, default=0.0)
p.add_argument("--seed", type=int, default=95)
p.add_argument("--log_every", type=int, default=100)
p.add_argument("--val_every", type=int, default=2500)
p.add_argument("--val_batches", type=int, default=64)
args = p.parse_args()

ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
rank = 0
if ddp:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.manual_seed(args.seed + rank)
np.random.seed(args.seed + rank)
random.seed(args.seed + rank)
out_dir = f"outputs/{args.expname}"
os.makedirs(out_dir, exist_ok=True)

cfg = omegaconf.OmegaConf.load(args.config)
model = LoopLM(**cfg).cuda()
n_params = sum(p_.numel() for p_ in model.parameters())
print(f"params: {n_params/1e6:.1f}M", flush=True)

# ---- optimizer with optzone grouping (same as train.py) ----
LOOP_PARAM_KEYS = ("loop_film", "branch_gate", "state_gate", "loop_rho",
                   "loop_gate_bias", "loop_rot", "loop_temp", "qkv_a", "qkv_b",
                   "branch_shift", "state_shift")
def _is_loop_param(name):
    return args.loop_param_lr_mult > 0 and any(k in name for k in LOOP_PARAM_KEYS)
decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and not _is_loop_param(n)]
nodecay = [q for n, q in model.named_parameters() if q.dim() < 2 and not _is_loop_param(n)]
loopp = [q for n, q in model.named_parameters() if _is_loop_param(n)]
groups = [{"params": decay, "weight_decay": args.weight_decay},
          {"params": nodecay, "weight_decay": 0.0}]
if loopp:
    groups.append({"params": loopp, "weight_decay": 0.0,
                   "lr": args.lr * args.loop_param_lr_mult})
    print(f"optzone: {len(loopp)} per-loop params -> wd=0, lr x{args.loop_param_lr_mult}",
          flush=True)
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)

start_it = 0
ckpt_path = f"{out_dir}/ckpt.pth"
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); start_it = ck["it"]
    print(f"resumed from {ckpt_path} @ iter {start_it}", flush=True)
raw_model = model
if ddp:
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

train_data = np.memmap(f"{args.data_dir}/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap(f"{args.data_dir}/val.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(args.seed + rank * 1000)

def get_batch(data, bs, generator=None):
    ix = (generator if generator is not None else rng).integers(
        0, len(data) - args.seq_len - 1, size=bs)
    x = np.stack([data[i:i + args.seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + args.seq_len] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).cuda(non_blocking=True),
            torch.from_numpy(y).cuda(non_blocking=True))

@torch.no_grad()
def evaluate():
    model.eval()
    vr = np.random.default_rng(1234)  # fixed val windows for paired comparison
    losses = []
    for _ in range(args.val_batches):
        x, y = get_batch(val_data, args.bs, vr)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            logits = raw_model(x)
        losses.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                      y.view(-1)).item())
    model.train()
    return float(np.mean(losses))

model.train()
import time
t0 = time.time()
for it in range(start_it + 1, args.steps + 1):
    x, y = get_batch(train_data, args.bs)
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    if math.isfinite(gn):
        opt.step()
    sched.step()
    if it % args.log_every == 0 and rank == 0:
        ips = args.log_every / (time.time() - t0); t0 = time.time()
        print(f"Iter {it:06d} loss {loss.item():.4f} it/s {ips:.2f}", flush=True)
    if (it % args.val_every == 0 or it == args.steps) and rank == 0:
        vl = evaluate()
        print(f"Iter {it:06d} VAL loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True)
        torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "it": it}, ckpt_path)

if rank != 0:
    exit(0)
vl = evaluate()
result = {"expname": args.expname, "val_loss": vl, "ppl": math.exp(vl),
          "params_M": n_params / 1e6, "steps": args.steps,
          "tokens": args.steps * args.bs * args.seq_len}
with open(f"{out_dir}/eval_lm.json", "w") as f:
    json.dump(result, f, indent=1)
print("FINAL", json.dumps(result), flush=True)
torch.save({"model": raw_model.state_dict()}, f"{out_dir}/model_final.pth")
