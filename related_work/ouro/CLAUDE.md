# Ouro — Scaling Latent Reasoning via Looped Language Models (arXiv:2510.25741, ByteDance Seed et al.)

Folder: paper PDF + `hf_modeling_code/` (modeling_ouro.py / configuration_ouro.py / config.json from
huggingface.co/ByteDance/Ouro-1.4B). No standalone official GitHub training repo; the open-source
release is the HF checkpoints (Ouro-1.4B, 2.6B, -Thinking variants) with trust_remote_code modeling.

## What the paper does
The existence proof that looped LMs work at frontier data scale: **LoopLM pre-trained on 7.7T
tokens**, 1.4B/2.6B params, up to **T=4 recurrent steps** over the full layer stack
(`F^(t) = lmhead ∘ M_L^t ∘ emb`). Reasoning is built into pre-training via latent iterative
computation instead of longer CoT.

## Key method
- **Deep supervision at every loop step**: LM head decodes after each iteration; per-step CE L^(t);
  total adaptive loss averages over steps t=2..Tmax with exit-gate weighting.
- **Learned adaptive depth (early exit)**: an exit gate in parallel with the LM head outputs a
  per-step halt probability. Trained with **entropy regularization under a uniform prior over exit
  steps** (KL to uniform; geometric/Poisson priors bias shallow) so depth exploration stays
  unbiased; a later stage tunes the compute-performance trade-off by input difficulty.
- Architecture: standard decoder-only (MHA + RoPE, SwiGLU), **sandwich RMSNorm for loop stability**
  (explicitly citing Huginn), SmolLM2 49,152 vocab. Deliberately vanilla otherwise.

## Key numbers
- Ouro-1.4B ~= 4B standard transformers; Ouro-2.6B ~= 8B on most benchmarks: **2-3x parameter
  efficiency**, holding at multi-trillion-token scale.
- Gains are NOT from more raw capacity: controlled ablations attribute them to improved
  **knowledge manipulation** (extraction/composition), not more stored knowledge — converges with
  the Latent-Thoughts finding (loops buy reasoning, not memorization).
- Performance peaks at the trained depth (T=4, sometimes T=3/5 for Thinking variants) and degrades
  on extrapolated steps — fixed-T training does not extrapolate (contrast Huginn's sampled-r,
  which extrapolates to r>32; another datapoint for sampling loop counts if we want a K knob).

## What we borrow / compare against
1. **Scale legitimacy citation** + the sandwich-norm confirmation independent of Huginn.
2. **Design contrasts to test in NVS**: (a) deep supervision every loop (Ouro) vs final-step only
   (Deja View/LT2) — for us intermediate supervision means decoding target views at every loop:
   costly, Deja View argues against; try both at small scale. (b) fixed T + learned exit (Ouro) vs
   sampled K (Huginn/Deja View).
3. The exit gate is the future-work extension for us: per-scene or per-chunk learned loop counts
   (easy scenes exit early — an NVS latency story). Their uniform-prior entropy trick is the known
   fix for exit collapse; 2607.10110 already ported it to looped Mamba.
4. `hf_modeling_code/modeling_ouro.py` shows the loop + per-step head + gate wiring in ~30KB of
   readable HF-style code — the quickest reference for an exit-gate implementation.
