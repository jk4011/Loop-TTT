# LVSM — Large View Synthesis Model (arXiv:2410.17242, ICLR 2025 oral)

Folder: paper PDF + LaTeX (`latex/`) + official code `LVSM/` (github.com/haian-jin/LVSM).

## What the paper does
Transformer-only novel view synthesis from sparse posed inputs with **minimal 3D inductive bias**:
no volumes, no 3DGS, no epipolar structure. Two variants:
- **Encoder-decoder**: input tokens -> fixed set of 1D latent tokens -> decode target view.
- **Decoder-only** (the stronger one): concatenate input-view tokens + target Plücker-ray tokens
  into one sequence, full self-attention stack (24 blocks, d=768 at base), directly regress target
  patches. This is the architecture family LaCT-NVS (and hence our project) builds on.

## Key method
- Tokenization: images patchified (8x8) with per-pixel Plücker ray embeddings for pose; target view
  is queried purely by its ray tokens.
- Loss: photometric (MSE + perceptual) on rendered target views only; no depth/geometry supervision.
- Bidirectional attention among input tokens; QK-norm needed for stability at scale.

## Key numbers
- Object-level (GSO): ~+1.5–3.5 dB PSNR over LGM/GS-LRM-class baselines; scene-level (RealEstate10K):
  beats GPNR/pixelSplat/GS-LRM with the same data. Decoder-only > encoder-decoder in quality;
  scales cleanly with model size and #views; runs with as little as 1–2 input views.

## Why it is in the loop-TTT library
LVSM is the **task backbone and the depth-redundancy target**:
1. Our experiments inherit the LVSM decoder-only recipe via `../../lact/lact_nvs` (LaCT swaps full
   attention for window-attn + TTT). Looping a LaCT block = replacing the L-deep LVSM-style stack
   with K iterations of one block — the NVS twin of what Deja View did to VGGT-style models.
2. RAPTOR's Block-Recurrent Hypothesis says trained ViT stacks are ~k distinct blocks repeated;
   Deja View showed the reconstruction-transformer version. LVSM's plain 24-block stack is exactly
   the kind of depth we argue is "iteration paid for in unique parameters".
3. Comparisons to run: (a) LVSM/full-attention non-looped reference; (b) looped full-attention
   LVSM block (no TTT) — isolates what TTT state adds to looping; (c) looped LaCT block (ours).
4. Code notes: repo has both variants, Plücker-ray tokenizer, deferred patch loss; good source for
   the exact token layout and QK-norm placement if we retokenize.
