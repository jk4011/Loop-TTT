import argparse
import functools
import math
import os
import random

import lpips
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup

from data import NVSDataset
from model import LaCTLVSM

parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact")
parser.add_argument("--expname", type=str, default="default")
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--log_every", type=int, default=100)

# Data
parser.add_argument("--data_path", type=str, required=True,
                    help="NVSDataset json or Re10K *_index.json (see --dataset)")
parser.add_argument("--dataset", type=str, default="re10k", choices=["nvs", "re10k"])
parser.add_argument("--num_workers", type=int, default=16)

# Training
parser.add_argument("--compile", action="store_true")
parser.add_argument("--actckpt", action="store_true")
parser.add_argument("--bs_per_gpu", type=int, default=8)
parser.add_argument("--num_all_views", type=int, default=15)
parser.add_argument("--num_input_views", type=int, default=8)
parser.add_argument("--num_target_views", type=int, default=8)  
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size H, W")
parser.add_argument("--scene_pose_normalize", action="store_true")

# Optimizer
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup", type=int, default=4000)
parser.add_argument("--steps", type=int, default=80000)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--lpips_start", type=int, default=5000, help="Iteration to start LPIPS loss")
parser.add_argument("--loop_sup_weight", type=float, default=0.0,
                    help="I6 per-loop render supervision: weight of the discounted "
                         "MSE on intermediate loop renders (0 = off). LPIPS stays final-only.")
parser.add_argument("--loop_sup_gamma", type=float, default=0.5,
                    help="Discount per loop for intermediate render MSE (earlier loops weigh less)")
parser.add_argument("--distill_weight", type=float, default=0.0,
                    help="Deep-teacher self-distillation: a no_grad teacher runs "
                         "n_loops+distill_extra passes (SAME tied weights); the student's "
                         "final render is pulled toward the teacher's deeper render. Imports "
                         "the more-loops quality gain at iso-inference-compute. 0 = off.")
parser.add_argument("--distill_extra", type=int, default=4,
                    help="Extra loop passes the distillation teacher runs beyond n_loops.")
parser.add_argument("--stoch_depth", type=str, default="",
                    help="Stochastic loop-count training: comma list of loop counts to "
                         "sample per step (e.g. '2,3,4,5,6', mean ~= n_loops so train "
                         "compute stays ~baseline). Eval uses n_loops. Empty = off.")
parser.add_argument("--kd_teacher", type=str, default="",
                    help="Cross-model knowledge distillation: path to a frozen pretrained "
                         "DEEP teacher checkpoint (e.g. a dedicated L2x6+sup model). The "
                         "shallow student's render is pulled toward the teacher's render. "
                         "Inference stays iso (student only); teacher is train-only.")
parser.add_argument("--kd_config", type=str, default="",
                    help="Config yaml for the KD teacher model.")
parser.add_argument("--kd_traj", type=str, default="",
                    help="Trajectory-KD: comma list of 1-based teacher loop indices, one per "
                         "student loop (e.g. '2,3,5,6'); intermediate student renders are pulled "
                         "toward the mapped teacher loop renders. Requires --kd_teacher.")
parser.add_argument("--kd_traj_weight", type=float, default=0.3,
                    help="Weight of the trajectory-KD waypoint MSE terms")
parser.add_argument("--kd_lpips_weight", type=float, default=0.0,
                    help="Perceptual (LPIPS) KD: pull student render toward teacher render in "
                         "VGG feature space. The x6/x8 teacher edge is disproportionately "
                         "perceptual (probe: LPIPS 0.287->0.271->0.263), which MSE-KD misses. "
                         "Gated on lpips_start like the main LPIPS loss. Requires --kd_teacher.")
parser.add_argument("--init_from", type=str, default="",
                    help="Warm-start: load model WEIGHTS ONLY from this checkpoint (fresh "
                         "optimizer/schedule); ignored when auto-resuming from outputs/<exp>")
parser.add_argument("--ema_decay", type=float, default=0.0,
                    help="If >0 keep an EMA shadow of the weights (saved under 'ema' in ckpts)")
parser.add_argument("--loop_param_lr_mult", type=float, default=0.0,
                    help="If >0, put per-loop conditioning params (gates/film/rho) in a wd=0 "
                         "group at lr*this-mult (fixes their weight-decay-to-identity pull)")
parser.add_argument("--loop_anneal", type=str, default="",
                    help="Deterministic loop-count schedule 'n:until_iter,...' e.g. '6:6000,5:12000' "
                         "(after the last boundary the config n_loops is used)")
parser.add_argument("--kd_weight", type=float, default=1.0,
                    help="Weight of the KD render-matching loss.")
parser.add_argument("--seed", type=int, default=95)

args = parser.parse_args()
model_config = omegaconf.OmegaConf.load(args.config)
output_dir = f"outputs/{args.expname}"
os.makedirs(output_dir, exist_ok=True)

dist.init_process_group(backend="nccl")
ddp_local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % 8))
torch.cuda.set_device(ddp_local_rank)

# Seed everything
rank_specific_seed = args.seed + dist.get_rank()
torch.manual_seed(rank_specific_seed)
np.random.seed(rank_specific_seed)
random.seed(rank_specific_seed)
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(rank_specific_seed)

model = LaCTLVSM(**model_config).cuda()

# Optimizers
# Per-loop conditioning params (gates/film/rho) are zero-init and dim>=2, so they
# would land in the weight-decay group and be continuously pulled back toward their
# identity value -- fighting the conditioning they are meant to learn. With
# --loop_param_lr_mult>0 they get their own wd=0 group at a higher lr.
LOOP_PARAM_KEYS = ("loop_film", "branch_gate", "state_gate", "loop_rho",
                   "loop_gate_bias", "loop_rot", "loop_temp")
def _is_loop_param(name):
    return args.loop_param_lr_mult > 0 and any(k in name for k in LOOP_PARAM_KEYS)
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and not _is_loop_param(n)]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and not _is_loop_param(n)]
loop_params = [p for n, p in model.named_parameters() if _is_loop_param(n)]
optim_groups = [
    {"params": decay_params, "weight_decay": args.weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
]
if loop_params:
    optim_groups.append({"params": loop_params, "weight_decay": 0.0,
                         "lr": args.lr * args.loop_param_lr_mult})
    print(f"optzone: {len(loop_params)} per-loop params -> wd=0, lr x{args.loop_param_lr_mult}",
          flush=True)
optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup,
    num_training_steps=args.steps,
)

# Load checkpoint
now_iters = 0
for try_load_path in [output_dir, args.load]:
    # Always try to load from output_dir first to resume training
    if try_load_path is None: continue
    try:
        if os.path.isdir(try_load_path):
            checkpoints = [f for f in os.listdir(try_load_path) if f.startswith("model_") and f.endswith(".pth")]
            if not checkpoints: continue
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
            checkpoint_path = os.path.join(try_load_path, latest_checkpoint)
        else:
            checkpoint_path = try_load_path
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        now_iters = checkpoint["now_iters"]
        break
    except:
        continue

if now_iters == 0 and args.init_from:
    init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    init_sd = init_ckpt["model"] if "model" in init_ckpt else init_ckpt
    missing, unexpected = model.load_state_dict(init_sd, strict=False)
    print(f"Warm-started weights from {args.init_from} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

ema_state = None
if args.ema_decay > 0:
    ema_state = {n: p.detach().clone().float() for n, p in model.named_parameters()}
    if now_iters > 0 and "ema" in checkpoint:
        ema_state = {n: v.float() for n, v in checkpoint["ema"].items()}

model = DDP(model, device_ids=[ddp_local_rank])

# This activation checkpointing wrapper supports torch.compile
if args.actckpt:
    torch._dynamo.config.optimize_ddp = False
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
        apply_activation_checkpointing,
    )

    wrapper = functools.partial(ptd_checkpoint_wrapper, preserve_rng_state=False)

    def _check_fn(submodule) -> bool:
        from model import Block
        return isinstance(submodule, Block)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=_check_fn,
    )

if args.compile:
    model = torch.compile(model)  

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value
    return new_state_dict

# Data
if args.dataset == "re10k":
    from data_re10k import Re10KDataset
    dataset = Re10KDataset(args.data_path, args.num_all_views, tuple(args.image_size), scene_pose_normalize=args.scene_pose_normalize)
else:
    dataset = NVSDataset(args.data_path, args.num_all_views, tuple(args.image_size), scene_pose_normalize=args.scene_pose_normalize)
datasampler = DistributedSampler(dataset)

dataloader = DataLoader(
    dataset,
    batch_size=args.bs_per_gpu,
    shuffle=False,
    num_workers=args.num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=2,
    sampler=datasampler,
    generator=dataloader_seed_generator,    # This ensures deterministic dataloader
)

if dist.get_rank() == 0:
    print(model)
    print(optimizer)
    print(lr_scheduler)
    print(f"Start training from iter {now_iters}...")

remaining_steps = args.steps - now_iters
lpips_loss_module = lpips.LPIPS(net="vgg").cuda().eval()

# Cross-model KD teacher (frozen, train-only): a dedicated deep model whose render
# the shallow student learns to match. Inference uses only the student.
kd_teacher = None
if args.kd_teacher:
    kd_cfg = omegaconf.OmegaConf.load(args.kd_config)
    kd_teacher = LaCTLVSM(**kd_cfg).cuda().eval()
    kd_ckpt = torch.load(args.kd_teacher, map_location="cpu", weights_only=False)
    kd_teacher.load_state_dict(kd_ckpt["model"] if "model" in kd_ckpt else kd_ckpt)
    for p in kd_teacher.parameters():
        p.requires_grad_(False)
    if dist.get_rank() == 0:
        print(f"KD teacher loaded from {args.kd_teacher}", flush=True)
for epoch in range((remaining_steps - 1) // len(dataloader) + 1):
    for data_dict in dataloader:
        data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        input_data_dict = {key: value[:, :args.num_input_views] for key, value in data_dict.items()}
        target_data_dict = {key: value[:, -args.num_target_views:] for key, value in data_dict.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            target = target_data_dict["image"]
            use_loop_sup = args.loop_sup_weight > 0
            # Stochastic depth: sample this step's loop count (mean ~= n_loops, so
            # average train compute stays ~baseline); the tied weights become
            # depth-robust and the deeper unrolls get real GT gradient.
            n_over = None
            if args.stoch_depth:
                choices = [int(c) for c in args.stoch_depth.split(",")]
                n_over = choices[torch.randint(len(choices), (1,)).item()]
            if args.loop_anneal:
                # Deterministic pure function of now_iters -> auto-resume reproduces it.
                for part in args.loop_anneal.split(","):
                    n_str, until_str = part.split(":")
                    if now_iters < int(until_str):
                        n_over = int(n_str)
                        break
            use_all_loops = use_loop_sup or bool(args.kd_traj)
            rendering = model(input_data_dict, target_data_dict,
                              return_all_loops=use_all_loops, n_loops_override=n_over)
            aux_loss = 0.0
            if use_all_loops and not use_loop_sup:
                renders = rendering
                rendering = renders[-1]
            if use_loop_sup:
                renders = rendering
                rendering = renders[-1]
                # Discounted MSE on intermediate loop renders (final render excluded
                # here; it gets the standard full loss below).
                weights = [args.loop_sup_gamma ** (len(renders) - 1 - i) for i in range(len(renders) - 1)]
                if weights:
                    aux_loss = sum(wt * F.mse_loss(r, target) for wt, r in zip(weights, renders[:-1]))
                    aux_loss = args.loop_sup_weight * aux_loss / sum(weights)

            if kd_teacher is not None:
                with torch.no_grad():
                    t_out = kd_teacher(input_data_dict, target_data_dict,
                                       return_all_loops=bool(args.kd_traj))
                t_render = t_out[-1] if args.kd_traj else t_out
                aux_loss = aux_loss + args.kd_weight * F.mse_loss(rendering, t_render.detach())
                if args.kd_traj:
                    # Waypoint KD: intermediate student renders imitate the teacher's
                    # iteration path (final loop excluded; endpoint KD above covers it).
                    traj_map = [int(c) - 1 for c in args.kd_traj.split(",")]
                    traj_terms = [F.mse_loss(renders[i], t_out[traj_map[i]].detach())
                                  for i in range(len(renders) - 1) if i < len(traj_map)]
                    if traj_terms:
                        aux_loss = aux_loss + args.kd_traj_weight * sum(traj_terms) / len(traj_terms)
                if args.kd_lpips_weight > 0 and now_iters >= args.lpips_start:
                    kd_lp = lpips_loss_module(rendering.flatten(0, 1),
                                              t_render.detach().flatten(0, 1),
                                              normalize=True).mean()
                    aux_loss = aux_loss + args.kd_lpips_weight * kd_lp

            if args.distill_weight > 0:
                # Deep-teacher self-distillation: same tied weights, more loop passes,
                # no grad -> a better (deeper-unroll) target the shallow student learns
                # to match at its final render. Iso-inference (student stays n_loops).
                base_model = model.module if hasattr(model, "module") else model
                n_teacher = base_model.n_loops + args.distill_extra
                with torch.no_grad():
                    teacher = model(input_data_dict, target_data_dict,
                                    n_loops_override=n_teacher)
                aux_loss = aux_loss + args.distill_weight * F.mse_loss(rendering, teacher.detach())

            l2_loss = F.mse_loss(rendering, target)
            psnr = -10.0 * torch.log10(l2_loss).item()
            if now_iters >= args.lpips_start:
                lpips_loss = lpips_loss_module(rendering.flatten(0, 1), target.flatten(0, 1), normalize=True).mean()
            else:
                lpips_loss = 0.0
            loss = l2_loss + lpips_loss + aux_loss
        loss.backward()

        # Gradident safeguard
        skip_optimizer_step = False
        if now_iters > 1000:
            global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

            if not math.isfinite(global_grad_norm):
                skip_optimizer_step = True
            elif global_grad_norm > 4.0:
                skip_optimizer_step = True

        if not skip_optimizer_step:
            optimizer.step()
            if ema_state is not None:
                with torch.no_grad():
                    for n, p in model.module.named_parameters():
                        # actckpt inserts "_checkpoint_wrapped_module." into names
                        ema_state[n.replace("_checkpoint_wrapped_module.", "")].lerp_(
                            p.detach().float(), 1.0 - args.ema_decay)
        lr_scheduler.step()     # Always step the lr scheduler and iters
        now_iters += 1


        if dist.get_rank() == 0:
            if now_iters % args.log_every == 0 or now_iters <= 100:
                import time
                elapsed = time.time() - globals().get("_last_log_time", time.time())
                globals()["_last_log_time"] = time.time()
                ips = args.log_every / elapsed if elapsed > 0 and now_iters > 100 else 0.0
                lpips_val = lpips_loss.item() if isinstance(lpips_loss, torch.Tensor) else lpips_loss
                print(f"Iter {now_iters:07d}, PSNR: {psnr:.2f}, LPIPS: {lpips_val:.4f}, it/s: {ips:.2f}", flush=True)
            if now_iters % args.save_every == 0:
                save_dict = {
                    "model": remove_module_prefix(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "now_iters": now_iters,
                    "epoch": epoch,
                }
                if ema_state is not None:
                    save_dict["ema"] = {n: v.to(torch.bfloat16) for n, v in ema_state.items()}
                torch.save(save_dict, f"{output_dir}/model_{now_iters:07d}.pth")
            

        if now_iters == args.steps:
            break




        
        
    