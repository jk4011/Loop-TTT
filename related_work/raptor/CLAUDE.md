# RAPTOR — Block-Recurrent Dynamics in Vision Transformers (arXiv:2512.19941, ICLR 2026, Kempner)

Folder: paper PDF + official code `raptor/` (github.com/KempnerInstitute/raptor).

## What the paper does
**Block-Recurrent Hypothesis (BRH)**: a trained L-layer ViT approximately equals unrolling k << L
weight-tied blocks. They (1) discover the recurrent phase structure in pretrained ViTs, and
(2) distill DINOv2 ViT-B into "Raptors" — k tied blocks with per-phase repetition counts — at a
small fraction of unique depth.

## Key method
- **Phase discovery**: build the layer-representational-similarity matrix, segment depth into
  contiguous phases via **weighted max-cut solved by dynamic programming**; repetition counts
  n_1..n_k sum to L. Layer-swap test validates phases: within-phase layers are interchangeable,
  cross-phase swaps collapse the model.
- **Distillation recipe** (post-hoc, teacher frozen):
  - Match ALL intermediate activations of the teacher, phase-aligned.
  - Hybrid loss `lambda * L_teacher-forced + (1-lambda) * L_autoregressive + Omega(theta)` with
    **lambda annealed 1 -> 0** (open-loop to closed-loop; classic recipe against compounding
    error in recurrent rollouts), then a stage-2 pure autoregressive finetune.
  - **Learnable layer-index embeddings** make the tied block non-autonomous (per-iteration
    conditioning — same trick class as Deja View's time gates, but discrete index).
  - Reuses the teacher's patch embed + final LN.

## Key numbers
- ImageNet-1k linear probe vs DINOv2 ViT-B teacher (84.5): k=2 -> 81.2±0.2 (96%), k=3 -> 83.0±0.1
  (98%), k=4 -> 83.2±0.1 (saturates). Dense tasks keep a gap: ADE20k 43.0 vs 47.5 mIoU,
  NYUv2 RMSE 0.618 vs 0.578.

## What we borrow / compare against
1. **Justification evidence**: BRH says LVSM/LaCT-style stacks are already ~few recurrent phases —
   cite for "depth is redundant, loop it". If time permits, run their max-cut phase analysis on a
   trained (non-looped) LaCT-NVS stack to *show* the phase structure we then make explicit; a
   killer analysis figure and the repo has the tooling (`raptor/` package: similarity matrices,
   DP segmentation, swap tests).
2. **Prelude/coda insight**: phases mean first/last layers differ from the loopable middle —
   supports a Huginn-style (prelude, looped core, coda) partition of the LaCT stack rather than
   looping everything.
3. **Distillation as a second path**: if from-scratch looped training underperforms, their
   anneal-open-to-closed-loop distillation from our own deep LaCT checkpoint is plan B (and a
   baseline reviewers may ask for).
4. Distance from us: post-hoc compression of quadratic-attention ViTs, no from-scratch training,
   no fast weights, no NVS. Deja View is the closer methodological baseline; RAPTOR is the
   analysis/evidence toolkit.
