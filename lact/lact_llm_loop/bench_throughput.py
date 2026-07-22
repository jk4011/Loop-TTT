# Absolute-throughput bench at the LARGE LM scale (d768, seq4096, bs8 — the real
# 3B-token run setup), synthetic tokens (no HF streaming). One GPU, sequential.
# Baselines included: 12L unique depth (iso-compute competitor), 3L (1/4 compute),
# naive 3Lx4 loop, +3 dials, +dials+inner. All fp32 params + bf16 autocast + AdamW,
# no gradient checkpointing (same as train_small).
# Usage: CUDA_VISIBLE_DEVICES=<g> python bench_throughput.py
# BENCH_COMPILE=1: torch.compile each LaCTBlock (per-block scope — robust with the
# HF cache/flash-attn outer loop; the dials all live inside blocks, which is the
# fusion surface under test). Longer warmup to absorb compilation.
import os, sys, time, torch
sys.path.insert(0, ".")
from lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from lact_model.modeling_lact import LaCTForCausalLM
import train_small as ts

COMPILE = os.environ.get("BENCH_COMPILE", "0") == "1"
BS, SEQ, WARM, MEAS = 8, 4096, (25 if COMPILE else 10), 30
BASE = dict(hidden_size=768, num_attn_heads=12, num_lact_heads=4,
            lact_chunk_size=1024, window_size=1024, vocab_size=32000,
            max_position_embeddings=4096, fuse_cross_entropy=True)
VARIANTS = [
    ("orig_12L (iso-compute unique)", dict(num_hidden_layers=12, n_loops=1)),
    ("l3 (1/4 compute)",              dict(num_hidden_layers=3, n_loops=1)),
    ("naive 3Lx4",                    dict(num_hidden_layers=3, n_loops=4)),
    ("3Lx4 + 3dials",                 dict(num_hidden_layers=3, n_loops=4, loop_dials=True)),
    ("3Lx4 + dials+inner",            dict(num_hidden_layers=3, n_loops=4, loop_dials=True, loop_inner="full")),
]

class A:
    loop_param_lr_mult = 16.0
    weight_decay = 0.1
    lr = 3e-4

def bench(name, kw):
    torch.manual_seed(0)
    m = LaCTForCausalLM(LaCTSWIGLUConfig(**BASE, **kw)).cuda()
    if COMPILE:
        for i in range(len(m.model.layers)):
            m.model.layers[i] = torch.compile(m.model.layers[i])
    n_par = sum(p.numel() for p in m.parameters())
    opt = ts.build_optimizer(m, A())
    x = torch.randint(0, 32000, (BS, SEQ), device="cuda")
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(input_ids=x, labels=x)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    for _ in range(WARM):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEAS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / MEAS
    print(f"{name}: {n_par/1e6:.1f}M params, {dt*1000:.1f} ms/step, "
          f"{BS*SEQ/dt:,.0f} tok/s", flush=True)
    del m, opt
    torch.cuda.empty_cache()

print(f"=== LM-LARGE THROUGHPUT BENCH (d768 seq{SEQ} bs{BS}, "
      f"{'per-block torch.compile' if COMPILE else 'eager'}, "
      f"{MEAS} timed steps) ===", flush=True)
for name, kw in VARIANTS:
    bench(name, kw)
print("=== LM BENCH DONE ===", flush=True)
