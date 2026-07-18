"""Mechanism probe: per-(loop, layer) TTT fit diagnostics on a trained checkpoint.

For every TTT update segment it records, WITHOUT changing the computation:
  cos_pre  = mean cos(f_w_in(k), v)   how well the INCOMING fast weights already
                                      explain this pass's regression targets
  cos_post = mean cos(f_w_out(k), v)  ... after this pass's update
  dw       = relative fast-weight movement ||w_out - w_in|| / ||w_in|| (w1)
  x_rms    = RMS of the block-input tokens (loop drift/divergence check)

Usage:
  python probe_loop_fit.py --load outputs/<exp>/model_0030000.pth \
      --config config/<cfg>.yaml [--num_scenes 16] [--out probe.json]

Monkeypatches FastWeightGluMLPMultihead forwards at runtime (baseline files
stay untouched). Records are keyed by visit order = loop_idx * layers + block.
"""
import argparse
import json
import types

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from data_re10k import Re10KDataset
from model import LaCTLVSM
import lact_ttt

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--data_path", type=str, default="/tmp/re10k/test_index.json")
parser.add_argument("--num_scenes", type=int, default=16)
parser.add_argument("--num_input_views", type=int, default=8)
parser.add_argument("--num_target_views", type=int, default=4)
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--window", type=int, default=128)
parser.add_argument("--bs", type=int, default=2)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

model_config = omegaconf.OmegaConf.load(args.config)
model = LaCTLVSM(**model_config).cuda()
ckpt = torch.load(args.load, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

RECORDS = []  # one dict per TTT-layer call, in call order


def probe_forward(self, x, info={}, *fargs):
    qkv = F.silu(self.to_qkv(x), inplace=False)
    q, k, v = rearrange(qkv, "b l (qkv h d) -> qkv (b h) l d", qkv=3, h=self.num_heads)
    q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
    k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

    lr = self.lr_fc(x.float())
    lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
    lr0, lr1, lr2 = rearrange(lr, "b l (lrs h d) -> lrs (b h) l d", lrs=3, h=self.num_heads)

    if "w0" in info:
        w0, w1, w2 = info["w0"], info["w1"], info["w2"]
    else:
        w0 = self.w0.repeat(x.shape[0], 1, 1)
        w1 = self.w1.repeat(x.shape[0], 1, 1)
        w2 = self.w2.repeat(x.shape[0], 1, 1)

    rec = {"loop_idx": int(info.get("loop_idx", 0)), "x_rms": float(x.float().pow(2).mean().sqrt())}
    upd = [op for op in info["ttt_op_order"] if op.update]
    if upd:
        s, e = upd[0].start, upd[0].end
        ki, vi = k[:, s:e], v[:, s:e]
        f_in = (F.silu(ki @ w0, inplace=False) * (ki @ w2)) @ w1
        rec["cos_pre"] = float(F.cosine_similarity(f_in.float(), vi.float(), dim=-1).mean())

    out, w0o, w1o, w2o = lact_ttt.fast_weight_swish_glu_weight_norm_mini_batch_apply(
        w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
        muon_update_steps=self.muon_update_steps,
    )

    if upd:
        f_out = (F.silu(ki @ w0o, inplace=False) * (ki @ w2o)) @ w1o
        rec["cos_post"] = float(F.cosine_similarity(f_out.float(), vi.float(), dim=-1).mean())
        rec["dw1_rel"] = float((w1o - w1).norm() / (w1.norm() + 1e-8))
    RECORDS.append(rec)

    out = self.o_norm(out)
    out = rearrange(out, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0])
    return self.c_proj(out), {"w0": w0o, "w1": w1o, "w2": w2o}


n_patched = 0
for block in model.blocks:
    for mod in block.module_list:
        if isinstance(mod["f"], lact_ttt.FastWeightGluMLPMultihead):
            mod["f"].forward = types.MethodType(probe_forward, mod["f"])
            n_patched += 1
print(f"patched {n_patched} TTT layers; n_loops={getattr(model, 'n_loops', 1)}, "
      f"state_mode={getattr(model, 'ttt_state_mode', 'reset')}")

dataset = Re10KDataset(
    args.data_path, args.num_input_views + args.num_target_views,
    tuple(args.image_size), scene_pose_normalize=True, eval_mode=True,
    window=args.window, num_input_views=args.num_input_views,
    num_target_views=args.num_target_views, max_scenes=args.num_scenes,
)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)

n_in = args.num_input_views
agg = {}
with torch.no_grad():
    for data_dict in loader:
        data_dict = {kk: vv.cuda() for kk, vv in data_dict.items()}
        input_dd = {kk: vv[:, :n_in] for kk, vv in data_dict.items()}
        target_dd = {kk: vv[:, n_in:] for kk, vv in data_dict.items()}
        RECORDS.clear()
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            model(input_dd, target_dd)
        for visit, rec in enumerate(RECORDS):
            agg.setdefault(visit, []).append(rec)

summary = []
for visit in sorted(agg):
    recs = agg[visit]
    row = {"visit": visit, "loop_idx": recs[0]["loop_idx"]}
    for key in ("cos_pre", "cos_post", "dw1_rel", "x_rms"):
        vals = [r[key] for r in recs if key in r]
        if vals:
            row[key] = round(float(np.mean(vals)), 4)
    summary.append(row)
    print(row)

out_path = args.out or args.load.replace("model_0030000.pth", "probe_loop_fit.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=1)
print(f"saved -> {out_path}")
