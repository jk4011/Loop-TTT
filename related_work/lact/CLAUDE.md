# LaCT — Test-Time Training Done Right (arXiv:2505.23884, ICLR 2026)

**Role for us: this is our substrate.** The Loop-TTT project loops the LaCT block; everything here is
the thing being looped. Official code is NOT cloned here — it already lives in this repo at
`../../lact/` (our working copy, incl. `lact_nvs/` for the NVS experiments). Upstream:
github.com/a1600012888/LaCT. This folder: paper PDF + LaTeX source (`latex/`).

## What the paper does
Fixes the hardware inefficiency of prior TTT layers (TTT-Linear/MLP update fast weights every
16–64 tokens -> <5% FLOPs utilization, custom kernels). LaCT uses **large-chunk TTT**: update the
fast weights once per 2K–1M-token chunk with plain PyTorch ops, reaching ~70% utilization and
enabling much larger nonlinear fast-weight networks (up to ~40% of model params).

## Key method (what our loop wraps)
- **Fast network**: SwiGLU-MLP `fW(x) = W2 (SiLU(W1 x) ⊙ W3 x)` — nonlinear, unlike the matrix
  state of DeltaNet/GDN-style linear mixers (the LT2 family).
- **Update**: self-supervised objective on a chunk (negative dot product `-fW(k)·v` with per-token
  LR weighting), one optimizer step per chunk; **Muon-style normalized update** (msign/Newton-Schulz
  orthogonalization of the gradient) + L2-normalized fast weights per update. This bounded-update
  property is exactly the stabilizer LT2 says looping needs (their "delta rule bounds updates" role).
- **Apply**: `o = fW(q)` for queries of the chunk; block = window attention (locality) + large-chunk
  TTT (long-range) + feed-forward. Order for NVS: apply-then-update per chunk (intra-chunk
  bidirectional; chunk = one view or group of views).
- Optional learnable per-channel forget gate (weight decay on fast weights) — the "gating"
  ingredient in LT2's stability recipe.

## Key numbers
- NVS (their Sec. 5.1, LVSM-style decoder-only backbone): matches full attention at 37.9 dB PSNR
  (48 input views, obj-level); prefill latency 16s -> 1.4s vs full attention at high view counts.
  Scales to 128K-token multi-view context.
- LM 760M/3B and 14B-token video DiT results: on par with or better than GLA/DeltaNet baselines
  with far larger state size.

## What we borrow / compare against
1. **The block we loop** is `../../lact/lact_nvs` block: window-attn + SwiGLU-MLP fast weights +
   Muon update. Loop-TTT question: what happens to fast-weight state W across loop iterations —
   reset / carry / gated blend (survey Sec. D1). Carrying W across loops = extra inner-loop
   optimization steps on the same chunk with re-mixed features.
2. **Stability prediction**: Muon normalization + fw-norm + optional forget gate give LaCT both
   LT2 stability ingredients (bounded updates + forgetting) -> we predict LaCT loops stably where
   naive TTT-SGD would not. Verify with grad-norm curves (LT2 Fig. 6 style).
3. **Baselines to beat**: non-looped LaCT at matched params (loop should win) and deeper non-looped
   LaCT at matched FLOPs (be honest about the 2607.10110 caveat: params + K-knob is our claim).
4. Muon msign iterations are themselves a fixed-point loop inside the update — do not confuse depth
   loops with these when instrumenting.
