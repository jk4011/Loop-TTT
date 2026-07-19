# BRIEFING for idea-generation subagents (Loop-TTT project)

## The task
Weight-tied **looped TTT** (LaCT Large-Chunk Test-Time-Training layer) for **novel view
synthesis** (LVSM-style, RE10K, 256², 8 input + 8 target views, dim 256, patch 16).
The naive loop = 2 unique Blocks cycled **4×** (effective depth 8), fast weights RESET
to the learned init each loop pass.

**GOAL (hard):** beat the naive loop by **≥ +0.5 dB PSNR** while keeping
**architecture AND computation essentially identical** (~1.0× FLOPs, same params, same
wall-clock). Method must stay **minimal** (one-sentence explainable). Resource
reallocation (more loops = more compute; width-for-depth = architecture search) is
REJECTED by the PI — the mechanism must exploit the **unique physics of TTT × looping**.

## Architecture (the substrate you are improving)
`Block = [ per-view SelfAttention (length_dim "l") , cross-view TTT layer (length_dim
"vl") , MLP ]`. **ALL cross-view information flows through the TTT layer** (attention is
per-view only). Data flow: posed multi-view images → ray/Plücker + RGB tokens → patchify
→ Block stack looped n× → decode target tokens to RGB. Input-view tokens UPDATE the fast
weights; target-view tokens only APPLY (read) them.

### The TTT layer (`lact_ttt.py`, the fast-weight mechanism — read it)
- Fast weights `w0, w1, w2` form a per-head **SwiGLU MLP**: `f_W(x) = (silu(x@w0) * (x@w2)) @ w1`.
  These are **ACTIVATIONS generated at test time, not parameters** — this is the key
  TTT×loop resource: the loop can manufacture many memory *instances* through time for free.
- Update (one step per pass): from tokens (k,v), compute the manual SwiGLU backward of the
  inner loss (a dot-product / regression loss k→v), form gradients, **orthogonalize each
  gradient with Newton–Schulz / Muon** (`zeropower_via_newtonschulz5`, `muon_update_steps=5`),
  add to the weights, then **re-normalize each weight column to its original norm (weight-norm)**.
- Per-token learning rates come from `lr_fc` (softplus around base_lr=0.01).
- `ttt_op_order` = list of `TTTOperator(start, end, update, apply)` segments; the kernel just
  iterates them. This is the control primitive (train/reconstruct/render differ only here).
- Research code lives in `lact_ttt_loop.py` (`LoopFastWeightGluMLPMultihead`); `lact_ttt.py`
  is the untouched baseline. `model.py` `LaCTLVSM` holds the loop wrapper (`n_loops`,
  `ttt_state_mode` reset|carry, and many flags). Eval uses `forward()`.

## THE THREE CORE FACTS (all measured, do not contradict)
1. **The NS-orbit pathology (core challenge).** The update `W ← weight_norm(W + NS(grad))`
   has Newton–Schulz normalize the gradient to a *fixed-norm* rotation regardless of
   magnitude, and weight-norm pins W to a sphere. So every step is a ~constant-angle move.
   Fine for ONE step (LaCT's design), but under looping (multi-step) it **orbits the optimum
   instead of converging**. Measured: naive `carry` (multi-step, same tensor) is −0.45 dB
   vs reset. Probe on trained models: post-update fit cos(f_W(k),v) DEGRADES across carry
   loops (0.70→0.41) while step size never decays.
2. **Fast-weight capacity is BINDING but the memory UNDERFITS.** Halving the TTT hidden
   dim: −0.84 dB. Yet a single update reaches cos(f_W(k),v) only 0.3–0.5 (far from 1). The
   memory is the bottleneck resource AND it is under-optimized — a paradox to exploit.
3. **Loop gain = joint input+target refinement.** Cutting target read-depth: −1.66; cutting
   input write-depth: −0.92; deepening both (more loops): +0.44 (but that costs compute).
   Neither side can be frozen/cached. In reset mode the gains come from *feature refinement*
   across passes (fast weights re-fit from scratch each pass — and that is FINE).

## THE STACKING LAW (your main lever)
Mechanisms on **DISTINCT axes ADD** (nearly linearly); mechanisms on the **same axis do
not**. Confirmed:
- ep2 (fit axis) +0.077, sup (loss axis) +0.073 → ep2+sup **+0.154** (additive).
- + gates (conditioning axis) → ep2+sup+gates **+0.187** (adds +0.033, additive).
- boost + sup → +0.117 (SUB-additive: boost overlaps the "make each loop contribute" family).
So the path to +0.5 is **finding many orthogonal positive axes**, each minimal.

## WHAT WORKS (positive, iso-compute, s95 paired Δ vs naive 22.204)
- **boost +0.099** (0% slower): each loop's memory inits FRESH but regresses onto the
  RESIDUAL the previous loop's memory left, `v − f_{W_{ℓ-1}}(k)` → n memory instances
  specialize on disjoint residual layers, effective capacity ×n. Revives the dead carry family.
- **ep2 +0.077** (−6% speed): 2 inner update steps per pass on the SAME (k,v) chunk
  (drift-free multi-step; attacks underfit). ep3 saturates (worse) — fit-axis peak is 2.
- **sup +0.073** (3-seed confirmed): decode target render after each loop, discounted MSE loss.
- **gates +0.033–0.04**: Déjà-View per-loop branch+state channel gates + LT2 rho gate (0 FLOPs).
- (loop count itself: +0.44–0.58 but costs 1.25–1.5× compute — the rejected lever.)

## WHAT FAILS (do NOT re-propose these)
- **carry** (naive multi-step, same tensor) −0.45. **delta-write alone** +0.03 (helps carry
  +0.23 but weak alone). **rho / rho2 / lrs** (step-size or lr modulation) ≈ 0 — Muon erases
  magnitude, so any lr/step-size knob is a no-op. **per-loop learned init (pli)** −0.083
  (shared init is BETTER — meta-learning/regularization; "shared init is a compromise" is FALSE).
- **read_refine** (re-query memory at apply) ≈ 0 (apply axis dead at fixed strength 0.15).
- **input injection** (Huginn add) slightly negative. **render feedback** (RAFT-style) −0.20.
- **Schedule surgery all fails**: late-join targets −1.66, read-heavy −0.92, sequential
  chunked writes −1.27, SfM/incremental view registration −1.97, muon-steps cut −0.34,
  TTT-capacity cut −0.84.
- **Resource reallocation** (width↔depth frontier) peaks at only +0.13 (inverted-U at d240×5).
- **momentum** (cross-loop heavy-ball, reset W + carry pre-NS gradient momentum M) — just had
  a kernel bug fixed, re-running; result unknown as of this briefing.

## Constraints for any proposed mechanism
- ~1.0× FLOPs & wall-clock (≤ ~1.1× tolerated if strongly justified); minimal/one-sentence;
  0 or tiny params preferred; must degrade to baseline via zero-init so it never hurts.
- Must be implementable as a small diff in `lact_ttt_loop.py` / `model.py` (see kernel above).
- Judge every result 3-seed paired (single-seed <0.15 dB is noise).

---

# ADDENDUM (2026-07-20): Wave-10 results + HARD task-agnostic constraint

## NEW HARD CONSTRAINT: the method must be TASK-AGNOSTIC.
The mechanism must extend beyond NVS to LLM / video / any sequence task. **Do NOT
propose anything that uses camera pose, Plücker/epipolar geometry, multi-view 3D
structure, rays, depth, or image-specific priors.** (A camera-geometry addressing
idea worked in a sibling project but is REJECTED here for this reason.) The fast
weights, the loop, the update/apply, per-token features — those are the substrate;
your mechanism must only touch those, not the task semantics.

## Wave-10 (100-idea round 1) RESULTS — the optimization family is DEAD
All measured s95 paired vs naive loop 22.204; most also confirmed neutral by design.
- **Acceleration/extrapolation FAILS hard.** cross-loop weight momentum (heavy-ball) −0.18;
  feature-space Anderson/Nesterov extrapolation (x += β(x−x_prev)) −2.49 (collapsed to a
  shallow L2-equivalent). LESSON: **the loop needs GENUINE iterations, not extrapolation to
  a fixed point.** The refinement IS the computation.
- **Exact/preconditioned inner optimization is NEUTRAL.** Gauss-Newton/RLS readout on w1
  via Richardson (−0.03) AND a definitely-non-identity diagonal preconditioner (+0.01).
  DECISIVE: conditioning the fast-weight update DIRECTION does not improve PSNR →
  **fit-quality of the memory is NOT the bottleneck** (contradicts the "underfit" intuition;
  the memory underfits but that underfit is not what caps quality).
- **Cumulative/true-gradient boosting FAILS (−1.52).** Carrying the residual vector in token
  space is stale because the value target is recomputed each loop (moving target). boost
  (+0.099) works precisely because it re-evaluates the previous memory on the CURRENT keys,
  staying in the current representation. LESSON: cross-loop state must be re-expressed in the
  current features, not carried as a frozen vector.
- **Neutral/dead:** Polyak iterate averaging (−0.01), spectral coarse-to-fine NS-step schedule
  (−0.15), key DC-decorrelation (+0.017), per-loop q/k temperature (−0.01), per-loop learned
  init "pli" (−0.083), read-side refinement (0), input injection (slightly −), render feedback
  (−0.20), all schedule surgery (late-join −1.66, chunked −1.27, sfm −1.97, read-heavy −0.92).

## THE META-LESSON (build on this)
Two clean categories emerged:
- **WORKS** = cheap structure that makes each loop pass do DIFFERENT / genuinely-contributing
  work, keeping real iterations: ep2 (2 real inner update steps, fit +0.077), sup (per-loop
  deep supervision, loss +0.073), gates (per-loop channel conditioning, +0.033), boost
  (each loop's fresh memory targets the previous memory's residual re-evaluated on current
  keys, capacity +0.099). Orthogonal-axis STACK ep2+sup+gates = **+0.283 dB (3-seed paired)**,
  same params, +6% wall-clock. This is the confirmed best.
- **FAILS** = anything that tries to make the loop's OPTIMIZATION better (accelerate it,
  precondition it, converge its orbit, average its iterates) or that freezes cross-loop state.

So: the loop's value is **iterative feature refinement through genuine repeated computation**,
NOT optimization quality. New ideas should add task-agnostic structure that makes passes
contribute complementary work, or find genuinely new orthogonal axes — and must survive the
"is this just an optimization tweak?" test (those are dead) and the "does this need task
semantics?" test (those are rejected).

## Confirmed task-agnostic winners to STACK onto (orthogonal axes add):
fit (ep2) · loss (sup, = per-loop deep supervision) · conditioning (gates) · capacity (boost).
Target: find 2-3 more orthogonal task-agnostic axes, each ~+0.05–0.1, to reach +0.5.
