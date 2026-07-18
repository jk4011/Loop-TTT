# Deja View — Looping Transformers for Multi-View 3D Reconstruction (arXiv:2605.30215, NVIDIA)

Folder: paper PDF + official code `dvlt/` (github.com/nv-tlabs/dvlt).

## What the paper does
**Our vision-side twin.** Replaces the deep decoder stack of feed-forward 3D reconstruction
transformers (VGGT/Pi3-style) with **one shared looped block** applied K times. Thesis: depth in
recon transformers is implicit iterative refinement paid for inefficiently in unique parameters —
make the iteration explicit. 117M params competitive with VGGT (1257M), Pi3 (959M), MASt3R-SfM
(690M) on DTU/ETH3D/7-Scenes/ScanNet++/nuScenes.

## Key method / recipe
- Backbone: DINOv2 ViT-B encoder (P=14, d=768) -> per-view tokens z0 (+ per-view register tokens
  and a camera token; 2D RoPE). Looped block = frame-attention (per-view) + global-attention
  (all views) sub-blocks, pre-norm + LayerScale. Shallow untied ray/depth decoder heads.
- **Time-conditioned recurrence** over a partition 0=t0<...<tK=1 of the unit interval:
  `z_{k+1} = f_theta(z_k, t_k, t_{k+1})`. Conditioning on the *continuous* interval (not a discrete
  step index) decouples the weights from any specific K -> one checkpoint serves a range of K.
- Cheap untied per-step params: an MLP on sinusoidal embeddings of (t_k, t_{k+1}) emits **three
  channel-wise scale vectors (s_attn, s_mlp, s_out)** gating the attention branch, MLP branch, and
  residual stream ("state gate"). Everything else fully tied.
- **K ~ Beta(2,1) scaled into [8,16] per training batch**; inference default K=16; K is a
  quality/compute knob within the trained range.
- **Supervision at final step z_K only** (no deep supervision; saves K-1 decoder passes).
  Two-stage: 200K iters end-to-end + 40K depth-head finetune; 29-dataset mix, p_i ∝ sqrt(N_i).

## Key numbers (the ablation that matters to us, their Tab. 4/5)
- Tied 16-step block **beats 16 fully-untied independent blocks**: inlier ratio 61.1 (untied) ->
  66.4 (shared) -> 69.2 (+time gates +state gate). Explicit iteration is a *stronger inductive
  bias*, not merely param saving.
- Mechanism observed: feature norm grows monotonically over iterations while direction
  cos(z_k, z_K) -> 1 — "directional refinement"; decoder LayerNorm absorbs the norm growth.
  Decoding intermediate z_k gives monotone metric improvement across k=1..16.

## What we borrow / compare against
1. **The recipe to copy first**: K ~ Beta(2,1) in [8,16], final-step-only supervision, and the
   3 channel-wise time-gates as our only untied per-loop params — directly portable to the LaCT
   block (gate window-attn branch, TTT branch, and residual separately; a per-loop gate on the
   TTT *update* magnitude is our natural 4th gate they don't have).
2. **Headline template**: reproduce their tied-beats-untied result with LaCT layers in NVS, and
   attribute the mechanism to fast-weight state carry (they have no state — their loop only
   refines features; ours also *re-optimizes memory*).
3. Note the conflict with Huginn: Deja View conditions on time and works in [8,16]; Huginn found
   step conditioning breaks path independence/extrapolation. If we want K extrapolation beyond the
   trained range, prefer Huginn-style unconditioned core; if we want a bounded K knob, Deja View's
   continuous-time gates are the proven vision recipe.
4. Code notes: `dvlt/src` has the looped block, time-gate MLP, and K-sampling dataloader logic —
   use as reference implementation for our loop wrapper.
