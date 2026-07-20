"""Memory-overlap probe: are the per-loop fast-weight memories of a trained boost
(or naive/reset) model already complementary in FUNCTION space?

For each scene: capture every loop's converged (w0,w1,w2) per TTT layer plus the
final loop's post-LN input x to that layer; recompute the final loop's keys k
exactly as the layer does; evaluate every memory f_{W_l}(k_final); report
pairwise cosine overlap and the residual-chain norm decay.

Decides: if memories are already near-orthogonal -> diversity mechanisms
(multiboost/decorloss/...) have no headroom (explains wave-18 failures).

Usage: python probe_memory_overlap.py --ckpt outputs/r7_loop_l2x4_boost_s95/model_0030000.pth \
         --config config/loop_l2x4_boost_d256_p16.yaml --num_scenes 16
"""
import argparse
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import LaCTLVSM
from data_re10k import Re10KDataset
from lact_ttt_loop import LoopFastWeightGluMLPMultihead

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--config", required=True)
p.add_argument("--data_path", default="/tmp/re10k/test_index.json")
p.add_argument("--num_scenes", type=int, default=16)
p.add_argument("--bs", type=int, default=4)
args = p.parse_args()

model = LaCTLVSM(**omegaconf.OmegaConf.load(args.config)).cuda().eval()
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"] if "model" in ck else ck)

# wrap every TTT layer's forward to capture per-loop (x_in, state)
captures = {}  # layer_id -> list of (loop_idx, x_in, w0, w1, w2)
for name, mod in model.named_modules():
    if isinstance(mod, LoopFastWeightGluMLPMultihead):
        captures[name] = []
        orig = mod.forward

        def wrapped(x, info, _orig=orig, _name=name):
            out, state = _orig(x, info)
            captures[_name].append(
                (info.get("loop_idx", 0), x.detach(),
                 state["w0"].detach(), state["w1"].detach(), state["w2"].detach()))
            return out, state
        mod.forward = wrapped

ds = Re10KDataset(args.data_path, num_views=12, image_size=(256, 256),
                  scene_pose_normalize=True, window=128, eval_mode=True,
                  num_input_views=8, num_target_views=4, max_scenes=args.num_scenes)
loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=4)

n_loops = model.n_loops
overlap_acc = [[[] for _ in range(n_loops)] for _ in range(n_loops)]
resid_acc = [[] for _ in range(n_loops + 1)]

with torch.no_grad():
    for data in loader:
        data = {k: v.cuda() for k, v in data.items()}
        idd = {k: v[:, :8] for k, v in data.items()}
        tdd = {k: v[:, 8:] for k, v in data.items()}
        for c in captures.values():
            c.clear()
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            model(idd, tdd)
        for name, caps in captures.items():
            layer = dict(model.named_modules())[name]
            caps = sorted(caps, key=lambda t: t[0])
            # final loop's post-LN input -> recompute q,k,v exactly like the layer
            x_fin = caps[-1][1].float()
            qkv = layer.to_qkv(x_fin)
            qkv = F.silu(qkv, inplace=False)
            q, k, v = qkv.chunk(3, dim=-1)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            n_in = 8 * 256  # input tokens (update chunk)
            ki, vi = k[:, :n_in].float(), v[:, :n_in].float()
            # evaluate every loop's memory on the final keys
            preds = []
            for (li, _x, w0, w1, w2) in caps:
                pred = (F.silu(ki @ w0.float()) * (ki @ w2.float())) @ w1.float()
                preds.append(pred)
            for i in range(len(preds)):
                for j in range(len(preds)):
                    if i < j:
                        c = F.cosine_similarity(preds[i], preds[j], dim=-1).mean().item()
                        overlap_acc[i][j].append(c)
            # residual chain: ||v - sum_{l<=m} preds_l|| / ||v||
            run = torch.zeros_like(vi)
            resid_acc[0].append(1.0)
            for m, pr in enumerate(preds):
                run = run + pr
                resid_acc[m + 1].append(((vi - run).norm() / (vi.norm() + 1e-6)).item())

print(f"\n=== memory-overlap probe: {args.ckpt} ({args.num_scenes} scenes) ===")
print("pairwise cos( f_Wi(k_fin), f_Wj(k_fin) ), averaged over tokens/scenes/layers:")
for i in range(n_loops):
    row = " ".join(f"{np.mean(overlap_acc[i][j]):+.3f}" if i < j else "  --  "
                   for j in range(n_loops))
    print(f"  loop{i}: {row}")
print("residual-chain norm ||v - Σ_{l<=m} f_Wl(k)|| / ||v|| (m=0..n):")
print("  " + " ".join(f"{np.mean(r):.3f}" for r in resid_acc))
