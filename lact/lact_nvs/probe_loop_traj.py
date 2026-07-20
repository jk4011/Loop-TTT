"""Per-loop trajectory probe: PSNR/LPIPS of every loop pass' render, for one or
more checkpoints, over the standard 256-scene eval set.

Discriminates the +0.578 (L2x6) hypotheses (IDEAS_R3 agent-10):
  H1 re-paced trajectory: x6's loop-4 render < x4's final, gain accrues at loops 5-6.
  "better per-pass function": x6's loop-4 render already >= x4's final.
Also supports n_loops_override to render a model at extrapolated depth.

Usage:
  python probe_loop_traj.py \
    --ckpt outputs/r4_loop_l2x4_sup_s95/model_0030000.pth:config/loop_l2x4_d256_p16.yaml \
    --ckpt outputs/r7_loop_l2x6_sup_s95/model_0030000.pth:config/loop_l2x6_d256_p16.yaml \
    --num_scenes 128
"""
import argparse, numpy as np, torch
import omegaconf
from torch.utils.data import DataLoader
from model import LaCTLVSM
from data_re10k import Re10KDataset
import lpips as lpips_lib

p = argparse.ArgumentParser()
p.add_argument("--ckpt", action="append", required=True,
               help="path:config[:n_override] ; repeatable")
p.add_argument("--data_path", type=str, default="/tmp/re10k/test_index.json")
p.add_argument("--num_scenes", type=int, default=128)
p.add_argument("--num_input_views", type=int, default=8)
p.add_argument("--num_target_views", type=int, default=4)
p.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
p.add_argument("--window", type=int, default=128)
p.add_argument("--bs", type=int, default=8)
args = p.parse_args()

n_in, n_tg = args.num_input_views, args.num_target_views
dataset = Re10KDataset(args.data_path, num_views=n_in + n_tg,
                       image_size=tuple(args.image_size), scene_pose_normalize=True,
                       window=args.window, eval_mode=True, num_input_views=n_in,
                       num_target_views=n_tg, max_scenes=args.num_scenes)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)
lpips_model = lpips_lib.LPIPS(net="vgg").cuda().eval()


def load_model(path, cfg):
    m = LaCTLVSM(**omegaconf.OmegaConf.load(cfg)).cuda().eval()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m


for spec in args.ckpt:
    parts = spec.split(":")
    path, cfg = parts[0], parts[1]
    n_over = int(parts[2]) if len(parts) > 2 else None
    model = load_model(path, cfg)
    n_eff = n_over if n_over is not None else model.n_loops
    # accumulate per-loop psnr/lpips
    psnr_acc = [[] for _ in range(n_eff)]
    lpips_acc = [[] for _ in range(n_eff)]
    with torch.no_grad():
        for data_dict in loader:
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            idd = {k: v[:, :n_in] for k, v in data_dict.items()}
            tdd = {k: v[:, n_in:] for k, v in data_dict.items()}
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
                renders = model(idd, tdd, return_all_loops=True, n_loops_override=n_over)
            target = tdd["image"].float()
            for li, r in enumerate(renders):
                r = r.float().clamp(0, 1)
                mse = ((r - target) ** 2).flatten(1).mean(dim=1)
                psnr_acc[li].extend((-10.0 * torch.log10(mse)).cpu().tolist())
                lp = lpips_model(r.flatten(0, 1), target.flatten(0, 1),
                                 normalize=True).reshape(r.size(0), -1).mean(dim=1)
                lpips_acc[li].extend(lp.cpu().tolist())
    tag = path.split("/")[-2] if "/" in path else path
    print(f"\n=== {tag}  (n_eff={n_eff}, scenes={len(psnr_acc[0])}) ===")
    for li in range(n_eff):
        print(f"  loop {li+1}: PSNR {np.mean(psnr_acc[li]):.3f}  LPIPS {np.mean(lpips_acc[li]):.4f}")
