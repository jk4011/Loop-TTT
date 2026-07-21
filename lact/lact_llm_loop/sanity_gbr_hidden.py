# -*- coding: utf-8 -*-
"""Q26 GbR + h-PRA (branch-split input rope + hidden rotary) sanity suite.

New combined kernel: prenorm_block_causal_lact_swiglu_branch_hidden_rope —
the branch-split input routing of the GbR kernel (rotated q/k feed one
SwiGLU branch, plain copies the other) plus the hidden rotary of the h-PRA
kernel (rotate h = silu(w0 x_g) * (w2 x_c) before w1 on update and apply,
inverse rotation in the manual inner-loop backward, delta_only supported).

Modes:
  a : hidden identity (hcos=1, hsin=0, i.e. gain 0) — combined kernel must
      reproduce the BRANCH kernel bit-exactly on the same (rotated, plain)
      tensor pairs, with and without muon+momentum.
  b : same tensors in both branch slots — combined kernel must reproduce
      the HIDDEN kernel (plain-ladder path) bit-exactly, for
      delta_only in {False, True}, with and without muon+momentum.
  c : model level — gate-GbR + hidden_rope with ttt_hrope_gain=0.0 must
      match a plain gate-GbR model's loss BIT-EXACTLY (state_dict copy);
      backward through the combined path (gain 0 AND gain 1) is finite;
      gain-1 loss is finite and distinct.
  d : 100-step training smoke of gate + hidden gain 1.0 (w128); loss must
      decrease and stay finite; smoke artifacts deleted afterwards.

GPU discipline: uses ONLY a GPU that is free (no lock file in
../lact_nvs/outputs/.gpu_locks/ AND <2 GiB used); creates its own lock and
removes it on exit. If no GPU is free, everything runs on CPU with tiny
dims (dynamo disabled; fla's triton-backed RMSNorm / RotaryEmbedding /
GatedMLP and flash-attn are monkeypatched with exact-semantics torch shims
— both sides of every equivalence use the SAME shims, so the comparisons
still isolate the kernel/dispatch difference; mode d then runs an
in-process tiny training loop instead of train_small.py, which hardcodes
device="cuda").
"""

import argparse
import atexit
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ.setdefault("HF_HOME", "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_inductor"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

LOCK_DIR = os.path.join(_REPO_ROOT, "lact_nvs", "outputs", ".gpu_locks")
_MY_LOCK = None


def _release_lock():
    global _MY_LOCK
    if _MY_LOCK and os.path.exists(_MY_LOCK):
        os.remove(_MY_LOCK)
        print(f"[gpu] released lock {_MY_LOCK}")
    _MY_LOCK = None


def pick_device():
    """Free GPU = no lock file AND <2 GiB used. Claim it with a lock file;
    otherwise fall back to CPU (tiny dims). Must run BEFORE importing torch
    so CUDA_VISIBLE_DEVICES / TORCHDYNAMO_DISABLE take effect."""
    global _MY_LOCK
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        rows = [tuple(int(x) for x in line.split(","))
                for line in out.strip().splitlines() if line.strip()]
    except Exception as e:
        print(f"[gpu] nvidia-smi failed ({e}) -> CPU")
        rows = []
    locked = set()
    for f in glob.glob(os.path.join(LOCK_DIR, "gpu*")):
        m = re.match(r"gpu(\d+)$", os.path.basename(f))
        if m:
            locked.add(int(m.group(1)))
    for idx, mem in rows:
        if idx in locked or mem >= 2048:
            continue
        lock = os.path.join(LOCK_DIR, f"gpu{idx}")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, b"q26_sanity_gbr_hidden\n")
            os.close(fd)
        except FileExistsError:
            continue
        _MY_LOCK = lock
        atexit.register(_release_lock)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        print(f"[gpu] using free GPU {idx} (lock {lock})")
        return "cuda"
    print("[gpu] no free GPU (locks/busy on all) -> CPU with tiny dims")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch busy GPUs
    os.environ["TORCHDYNAMO_DISABLE"] = "1"  # eager on CPU (both sides alike)
    return "cpu"


DEVICE = pick_device()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lact_model.layer_lact_swiglu as lls  # noqa: E402
import lact_model.modeling_lact as lml  # noqa: E402
from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig  # noqa: E402
from lact_model.ttt_operation import (  # noqa: E402
    l2_norm,
    prenorm_block_causal_lact_swiglu_branch_rope,
    prenorm_block_causal_lact_swiglu_hidden_rope,
    prenorm_block_causal_lact_swiglu_branch_hidden_rope,
)

BASE_JSON = os.path.join(SCRIPT_DIR, "configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json")
VAL_CACHE = os.path.join(SCRIPT_DIR, "val_cache_fla-hub_transformer-1.3B-100B_4096_ds42.pt")

CPU = DEVICE == "cpu"


# ------------------------------------------------------ CPU shims
# fla's RMSNorm / RotaryEmbedding / GatedMLP and flash-attn are triton/CUDA
# only. On CPU we monkeypatch them with exact-semantics torch versions.
# Every equivalence test compares two models/kernels built with the SAME
# shims, so the comparison still isolates exactly the code under test.


class ShimRMSNorm(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        d = x.float()
        y = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.type_as(x)


class ShimRotary(nn.Module):
    """NeoX-style rotate-half rotary over the full head dim."""

    def __init__(self, dim, base=10000.0, **kw):
        super().__init__()
        inv = 1.0 / (float(base) ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    @staticmethod
    def _rh(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seqlen_offset=0, max_seqlen=None, cu_seqlens=None):
        pos = torch.arange(q.shape[1], device=q.device, dtype=torch.float32)
        pos = pos + seqlen_offset
        ang = pos[:, None] * self.inv_freq[None].to(q.device)
        cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None, :]
        qo = (q.float() * cos + self._rh(q.float()) * sin).type_as(q)
        ko = (k.float() * cos + self._rh(k.float()) * sin).type_as(k)
        return qo, ko


class ShimGatedMLP(nn.Module):
    def __init__(self, hidden_size, hidden_ratio=4, intermediate_size=None,
                 hidden_act="swish", fuse_swiglu=True):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * (hidden_ratio or 4) * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x, **kw):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def shim_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
    """Sliding-window causal attention, [b, s, h, d] in/out like flash-attn."""
    b, s, hn, hd = q.shape
    q_ = q.permute(0, 2, 1, 3).float()
    k_ = k.permute(0, 2, 1, 3).float()
    v_ = v.permute(0, 2, 1, 3).float()
    i = torch.arange(s, device=q.device)[:, None]
    j = torch.arange(s, device=q.device)[None, :]
    mask = j <= i
    if window_size[0] >= 0:
        mask = mask & ((i - j) <= window_size[0])
    att = (q_ @ k_.transpose(-1, -2)) / math.sqrt(hd)
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    return (att @ v_).permute(0, 2, 1, 3).type_as(q)


if CPU:
    lls.RMSNorm = ShimRMSNorm
    lls.RotaryEmbedding = ShimRotary
    lls.flash_attn_func = shim_flash_attn_func
    lml.TransformerMLP = ShimGatedMLP


# ------------------------------------------------------ shared builders

# model dims: production 200M small arch on GPU, tiny on CPU
MODEL_DIMS = (
    dict(hidden_size=128, num_hidden_layers=2, num_attn_heads=4,
         num_lact_heads=2, lact_chunk_size=128, window_size=128,
         max_position_embeddings=4096) if CPU else
    dict(hidden_size=768, num_hidden_layers=12, num_attn_heads=12,
         num_lact_heads=4, lact_chunk_size=1024, window_size=128,
         max_position_embeddings=4096)
)
SEQ_LEN = 512 if CPU else 4096


def build_config(**extra):
    with open(BASE_JSON) as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)
    cfg.update(dict(vocab_size=32000, use_fused_kernel=False,
                    bos_token_id=1, eos_token_id=2, ttt_prenorm=True))
    cfg.update(MODEL_DIMS)
    if CPU:
        # triton-fused ops unavailable; the fla imports are shimmed above
        cfg.update(dict(fuse_norm=False, last_layer_fuse_norm=False,
                        fuse_swiglu=False, fuse_cross_entropy=False))
    cfg.update(extra)
    return LaCTSWIGLUConfig(**cfg)


def build_model(seed=42, **extra):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return LaCTForCausalLM(build_config(**extra)).to(DEVICE)


def get_batch(bs=2, seq=SEQ_LEN, row0=0):
    cache = torch.load(VAL_CACHE, map_location="cpu")
    return cache[row0:row0 + bs, :seq].to(DEVICE)


def model_loss(m, x, backward=False):
    if CPU:
        loss = m(input_ids=x, labels=x).loss
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(input_ids=x, labels=x).loss
    if backward:
        loss.backward()
    return loss


# ------------------------------------------------------ kernel test rig


def kernel_inputs():
    """Random fast-weight/QKV set + a rotated copy of q/k (input rope) and a
    hidden-ladder (hcos, hsin) at gain 1 plus the identity pair (gain 0)."""
    torch.manual_seed(0)
    if CPU:
        bh, d, d_h, seq, chunk = 4, 32, 32, 512, 128
    else:
        bh, d, d_h, seq, chunk = 8, 192, 192, 4096, 1024
    P = d // 2
    inv = (1.0 / (500000.0 ** (torch.arange(P, dtype=torch.float32) / P))).to(DEVICE)
    pos = torch.arange(seq, device=DEVICE, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]  # [s, P]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[None]  # [1, s, d]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[None]

    def rot(x):  # NeoX rotate_half, matching the layer's input-rope pairing
        x1, x2 = x.chunk(2, dim=-1)
        return (x * cos + torch.cat([-x2, x1], dim=-1) * sin).type_as(x)

    w0 = (torch.randn(bh, d_h, d) / d ** 0.5).to(DEVICE)
    w2 = (torch.randn(bh, d_h, d) / d ** 0.5).to(DEVICE)
    w1 = (torch.randn(bh, d, d_h) / d_h ** 0.5).to(DEVICE)
    q_plain = l2_norm(torch.randn(bh, seq, d, device=DEVICE))
    k_plain = l2_norm(torch.randn(bh, seq, d, device=DEVICE))
    v = torch.randn(bh, seq, d, device=DEVICE)
    q_rot, k_rot = rot(q_plain), rot(k_plain)
    lr0 = 1e-3 * torch.rand(bh, seq, 1, device=DEVICE)
    lr1 = 1e-3 * torch.rand(bh, seq, 1, device=DEVICE)
    lr2 = 1e-3 * torch.rand(bh, seq, 1, device=DEVICE)
    mom = torch.rand(bh, seq, 1, device=DEVICE)

    # hidden ladder (gain 1, frac 0.5 -> P_h = d_h // 4), theta 500k
    P_h = d_h // 4
    h_inv = (1.0 / (500000.0 ** (torch.arange(P_h, dtype=torch.float32) / P_h))).to(DEVICE)
    h_ang = h_inv[:, None] * pos[None, :]  # [P_h, s]
    hcos, hsin = h_ang.cos(), h_ang.sin()
    hcos_id = torch.ones_like(hcos)
    hsin_id = torch.zeros_like(hsin)

    return dict(w0=w0, w1=w1, w2=w2, q_plain=q_plain, k_plain=k_plain,
                q_rot=q_rot, k_rot=k_rot, v=v, lr0=lr0, lr1=lr1, lr2=lr2,
                mom=mom, chunk=chunk, hcos=hcos, hsin=hsin,
                hcos_id=hcos_id, hsin_id=hsin_id)


# ---------------------------------------------------------------- a


def a(args):
    """Identity hidden rotation (gain 0): combined == branch kernel,
    bit-exact, on genuinely split (rotated, plain) tensor pairs."""
    t = kernel_inputs()
    ok_all = True
    for setting, use_muon, momentum in [
        ("muon+mom", True, t["mom"]), ("plainsgd", False, None),
    ]:
        for name, (qg, kg, qc, kc) in [
            ("gate-split", (t["q_rot"], t["k_rot"], t["q_plain"], t["k_plain"])),
            ("content-split", (t["q_plain"], t["k_plain"], t["q_rot"], t["k_rot"])),
        ]:
            with torch.no_grad():
                o_br = prenorm_block_causal_lact_swiglu_branch_rope(
                    t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
                    qg, kg, qc, kc, t["v"],
                    t["lr0"], t["lr1"], t["lr2"], chunk_size=t["chunk"],
                    use_muon=use_muon, momentum=momentum)
                o_cmb = prenorm_block_causal_lact_swiglu_branch_hidden_rope(
                    t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
                    qg, kg, qc, kc, t["v"],
                    t["lr0"], t["lr1"], t["lr2"],
                    t["hcos_id"], t["hsin_id"], chunk_size=t["chunk"],
                    use_muon=use_muon, momentum=momentum, delta_only=False)
            diff = (o_br.float() - o_cmb.float()).abs().max().item()
            ok = diff == 0.0
            ok_all &= ok
            print(f"[a:{setting}:{name}] max|branch - combined(hcos=1,hsin=0)| "
                  f"= {diff:.3e} -> {'PASS (bit-exact)' if ok else 'FAIL'}")
    print(f"[a] {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------- b


def b(args):
    """Same tensors in both branch slots: combined == hidden kernel
    (plain-ladder path), bit-exact, for delta_only in {False, True}."""
    t = kernel_inputs()
    qq, kk = t["q_rot"], t["k_rot"]  # standard rope input, as in real hpra
    ok_all = True
    for setting, use_muon, momentum in [
        ("muon+mom", True, t["mom"]), ("plainsgd", False, None),
    ]:
        for d_only in (False, True):
            with torch.no_grad():
                o_h = prenorm_block_causal_lact_swiglu_hidden_rope(
                    t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
                    qq, kk, t["v"],
                    t["lr0"], t["lr1"], t["lr2"],
                    t["hcos"], t["hsin"], chunk_size=t["chunk"],
                    use_muon=use_muon, momentum=momentum, delta_only=d_only)
                o_cmb = prenorm_block_causal_lact_swiglu_branch_hidden_rope(
                    t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
                    qq, kk, qq, kk, t["v"],
                    t["lr0"], t["lr1"], t["lr2"],
                    t["hcos"], t["hsin"], chunk_size=t["chunk"],
                    use_muon=use_muon, momentum=momentum, delta_only=d_only)
            diff = (o_h.float() - o_cmb.float()).abs().max().item()
            ok = diff == 0.0
            ok_all &= ok
            print(f"[b:{setting}:delta_only={d_only}] "
                  f"max|hidden - combined(same,same)| = {diff:.3e} -> "
                  f"{'PASS (bit-exact)' if ok else 'FAIL'}")
    print(f"[b] {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------- c


def c(args):
    """Full model: gate-GbR + hidden gain 0.0 == plain gate-GbR loss,
    bit-exact (state_dict copy); backward finite for gain 0 and gain 1;
    gain-1 loss finite and distinct."""
    x = get_batch(args.bs)
    ref = build_model(seed=42, ttt_branch_rope="gate")
    ref.train()
    loss_gbr = model_loss(ref, x).float().item()
    sd = ref.state_dict()
    del ref
    torch.cuda.empty_cache()
    print(f"[c] plain GbR(gate) loss           = {loss_gbr:.10f}")

    ok_all = torch.isfinite(torch.tensor(loss_gbr)).item()
    losses = {}
    for name, gain in [("g0", 0.0), ("g1", 1.0)]:
        m = build_model(seed=42, ttt_branch_rope="gate",
                        ttt_hidden_rope=True, ttt_hrope_gain=gain)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (name, missing, unexpected)
        m.train()
        loss = model_loss(m, x, backward=True)
        losses[name] = loss.float().item()
        finite = torch.isfinite(loss).item()
        n_grads, n_bad = 0, 0
        for pname, p in m.named_parameters():
            if p.grad is None:
                continue
            n_grads += 1
            if not torch.isfinite(p.grad).all().item():
                n_bad += 1
                print(f"[c:{name}] NON-FINITE grad: {pname}")
        ok_all &= finite and n_bad == 0
        print(f"[c] GbR+hidden gain={gain} loss    = {losses[name]:.10f} "
              f"finite={finite}; backward: {n_grads} grads, {n_bad} non-finite")
        del m, loss
        torch.cuda.empty_cache()

    exact = losses["g0"] == loss_gbr
    distinct = losses["g1"] != loss_gbr
    ok_all &= exact and distinct
    print(f"[c] gain0 == plain GbR bit-exact: {exact} "
          f"(diff = {abs(losses['g0'] - loss_gbr):.3e})")
    print(f"[c] gain1 distinct from GbR: {distinct}")
    print(f"[c] {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------- d


def d(args):
    """100-step gate+hidden(g1.0) w128 training smoke. GPU: train_small.py
    subprocess (production dims, killed by PID on timeout, dir deleted).
    CPU: in-process tiny loop (train_small.py hardcodes device='cuda')."""
    if not CPU:
        return _d_gpu(args)
    return _d_cpu(args)


def _d_gpu(args):
    out_dir = os.path.join(SCRIPT_DIR, "outputs", "q26_smoke_gatehpra_w128")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "train_small.py"),
        "--out_dir", out_dir,
        "--window_size", "128",
        "--bs", "8",
        "--steps", "100",
        "--log_every", "10",
        "--val_every", "1000000",
        "--save_every", "1000000",
        "--extra_json", json.dumps(
            {"ttt_branch_rope": "gate", "ttt_hidden_rope": True,
             "ttt_hrope_gain": 1.0}),
    ]
    log_path = os.path.join(out_dir, "smoke.log")
    print(f"[d] launching: {' '.join(cmd)}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=SCRIPT_DIR)
        print(f"[d] PID {proc.pid}")
        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"[d] TIMEOUT: killing PID {proc.pid}")
            proc.kill()
            proc.wait()
    with open(log_path) as f:
        log = f.read()
    steps = re.findall(r"step=(\d+) loss=([\d.]+)", log)
    losses = [(int(s), float(v)) for s, v in steps]
    print("[d] logged losses: " + " ".join(f"{s}:{v:.4f}" for s, v in losses))
    finite = all(torch.isfinite(torch.tensor(v)).item() for _, v in losses)
    nan_warn = "non-finite loss" in log
    ok = (proc.returncode == 0 and len(losses) >= 2 and finite
          and not nan_warn and losses[-1][1] < losses[0][1])
    if not ok:
        print(f"[d] rc={proc.returncode} tail of log:\n" + log[-2000:])
    else:
        shutil.rmtree(out_dir)
        print(f"[d] loss {losses[0][1]:.4f} -> {losses[-1][1]:.4f}, "
              f"smoke dir deleted")
    print(f"[d] {'PASS' if ok else 'FAIL'}")
    return ok


def _d_cpu(args):
    torch.manual_seed(42)
    m = build_model(seed=42, ttt_branch_rope="gate",
                    ttt_hidden_rope=True, ttt_hrope_gain=1.0)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1)
    cache = torch.load(VAL_CACHE, map_location="cpu")
    bs, seq, steps = 2, SEQ_LEN, 100
    losses = []
    for step in range(steps):
        r0 = (step * bs) % (cache.shape[0] - bs)
        x = cache[r0:r0 + bs, :seq].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.float().item())
        if step % 10 == 0 or step == steps - 1:
            print(f"[d] step={step} loss={losses[-1]:.4f}", flush=True)
    finite = all(torch.isfinite(torch.tensor(v)).item() for v in losses)
    first10 = sum(losses[:10]) / 10
    last10 = sum(losses[-10:]) / 10
    ok = finite and last10 < first10 and losses[-1] < losses[0]
    print(f"[d] finite={finite}; mean(first10)={first10:.4f} -> "
          f"mean(last10)={last10:.4f}; loss[0]={losses[0]:.4f} -> "
          f"loss[-1]={losses[-1]:.4f}")
    print(f"[d] (CPU tiny-dims in-process loop: {MODEL_DIMS['hidden_size']}d/"
          f"{MODEL_DIMS['num_hidden_layers']}L, seq {seq}, bs {bs}, w128; "
          f"train_small.py is CUDA-only)")
    print(f"[d] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["a", "b", "c", "d", "all"])
    p.add_argument("--bs", type=int, default=2)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.mode == "all":
        results = {m: globals()[m](args) for m in ("a", "b", "c", "d")}
        print("[all] " + " ".join(f"{k}:{'PASS' if v else 'FAIL'}"
                                  for k, v in results.items()))
        sys.exit(0 if all(results.values()) else 1)
    else:
        sys.exit(0 if globals()[args.mode](args) else 1)
