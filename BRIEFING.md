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

## ADDENDUM 3 (2026-07-20) — after 100-idea round 2 (waves 11–15). GOAL RAISED TO +1.0 dB.

New target: **+1.0 dB over naive loop (22.204 s95 / 22.117 3-seed) at ~iso compute for BOTH
training AND inference.** Confirmed stack so far: ep2+sup+gates = +0.283 (3-seed).

Round-2 results (all s95 paired vs naive 22.204 unless noted):
- **KD BREAKTHROUGH (the one new positive):** train-time cross-model knowledge distillation.
  A frozen, separately-trained DEEPER looped teacher (L2×6+sup, 22.781) supervises the L2×4
  student's rendering via MSE (weight 1.0). KD alone +0.143; stacked on ep2+sup+gates
  **+0.297 (s95)**, i.e. KD adds ~+0.11 on top of the stack, nearly orthogonally. Inference
  strictly iso (student unchanged); costs a one-time teacher train + ~frozen fwd per step.
  This is currently the ONLY mechanism that imports any of the "more-loops" gain.
- **More-loops gain exists but is compute-bound:** L2×6+sup = +0.578 at 1.5× inference.
  Self-distill (student's own deeper unroll as teacher) COLLAPSES (extrapolation off the
  training loop-count fails). Stochastic loop-count training: fails (−0.1 to −0.3, hurts the
  4-loop operating point). Only cross-model KD transfers any of it.
- **Last orthogonal-axis candidates all NULL:** per-loop stream RMS-renorm (−0.02), fused
  softmax readout over per-loop target feats (+0.035), train-time loop dropout (−0.09).
  LT2 SDPA output gate: ~0. QKV LoRA per-loop routing: ~0. Per-loop random orthogonal
  rotation of k/q (rot_bag): −0.15. NL-Cond gate bias: ~0.
- Axis census after ~200 ideas: fit(ep2)/loss(sup)/cond(gates)/capacity(boost)/KD(train-loss)
  are the only positive axes found. Everything on optimization-, schedule-, readout-,
  addressing-, extrapolation-axes is dead or null at this scale.

## WHAT ROUND 3 MUST DO DIFFERENTLY
+0.283 (stack) + ~+0.11 (KD) ≈ +0.39 confirmed; we need **≥ +0.6 more**. Incremental
single-axis tweaks measured so far give ≤+0.1 each and the known-axis pool is exhausted.
So round 3 asks for **fundamentally different mechanisms**, e.g. (non-exhaustive, think freely):
- Ways to make a WEIGHT-TIED loop behave like a much DEEPER net (the L2×6/L2×8 gain, +0.6~,
  is the prize — how can iso-inference computation emulate it? KD is one; find others).
- Mechanisms exploiting the TTT fast-weight STATE across loops in ways not yet tried
  (we tried carry/momentum/boost/cumboost/read-refine; boost alone was positive).
- Train-time-only structure (loss shaping, curricula, auxiliary tasks, distillation variants
  — inference stays identical; this family contains the only recent winner).
- Changing WHAT each pass computes (input/target routing, chunking across loops, token
  subset scheduling) — but note all naive schedule surgery failed; needs a new principle.
- Anything task-agnostic that makes pass ℓ's computation CONDITIONAL on pass ℓ−1's OUTPUT
  in a nonlinear, learnable way (the loop's value = genuine iterative refinement).
Hard filters unchanged: task-agnostic (no camera/geometry), ~iso train & inference FLOPs
(≤~1.1×), minimal (one-sentence explainable), prefer TTT-state × loop synergy.

## ADDENDUM 4 (2026-07-20) — ROUND 4: TTT×LOOP NOVELTY ONLY. This is the real target.

PI가 방향을 날카롭게 재설정했다. 지금까지의 "확정 스택" 성분 대부분은 PI가 원하는 것이 아니다:
- **ep2 제외**: TTT gradient step을 그냥 2번 = 계산량 증가일 뿐, naive, 차별성 없음. 신규 금지.
- **KD 제외**: 별도 teacher 훈련하는 cross-model distillation = 계산량, naive. 신규 금지.
- **sup 시큰둥**: DVLT의 가변 loop-count 감독과 유사, 특별하지 않음.
- **gates OK지만 아님**: vanilla transformer에도 됨 → TTT 특화 아님. generic add-on으로만 허용.
- **boost = 가장 novel, 정확히 추구 방향**: fast weight가 활성값(TTT) × loop가 메모리 인스턴스를
  여러 개 만듦(looping) 둘 다 활용. 이런 것을 더 만들어야 한다.

### 목표(불변): naive loop(22.204) 대비 +1.0dB @ iso train&inference, task-agnostic, minimal.
### 새 필수 제약: 메커니즘은 **vanilla transformer로 포팅 불가능한 TTT×loop 고유 물리**여야 한다.
fast weight(=활성 메모리), update(write)/apply(read) 비대칭, Muon/Newton-Schulz update, loop마다
메모리를 재생성/재-update — 이 substrate를 건드려야 한다. 단순 feature/loss/조건화(generic)는 아니다.

### 실패한 TTT×loop 실험 전체 부검 (반드시 이걸 넘어서라 — 같은 실수 반복 금지):
- 상태 축적/carry 전멸: carry −0.45, carry+lrs −0.445, carry+rho −0.477, momentum −0.18,
  cumboost(잔차 token-space carry) −1.52. → **Muon이 크기를 지워 궤도순환(NS-orbit 병리).**
- inner-update 최적화 개선 전부 null: precond(Gauss-Newton) −0.03, 대각 precond +0.01,
  epavg(Polyak) −0.01, c2f_muon −0.15, muon2(NS 5→2) −0.34. → **fit 품질은 병목이 아니다.**
- 기타 null/음성: pli(per-loop init) −0.083, read_refine 0, key_center +0.017, loop_temp −0.01,
  rot_bag(랜덤 직교회전) −0.15, nl_cond −0.059, qkv_route ~0.
- schedule surgery(문맥 분할) 전멸: late-join −1.66, read-heavy −0.92, chunk −1.27, sfm −1.97.
- 유일 성공: **boost +0.099** (reset + 이전 메모리를 **현재 키로 재평가**한 잔차만 새 메모리가 fit).
  delta +0.056(약함).

### 부검이 남긴 3 DESIGN RULE (새 아이디어는 이 규칙을 지켜야 살아남는다):
1. **NS-orbit 병리**: Muon/NS가 gradient를 고정-norm 회전으로 정규화 → 크기 정보 소실. 그래서
   (a) 가중치를 loop 간 축적하면 수렴 대신 순환 → **매 loop fresh 메모리가 사실상 필수**,
   (b) lr/크기 조절 knob은 전부 no-op. **오직 방향(direction)만 살아남는다 — 이걸 역이용하라.**
2. **cross-loop 신호는 반드시 "현재 특징"으로 재-표현하라**. boost(현재 키로 재평가)=+0.099 vs
   cumboost(얼린 잔차 벡터 carry)=−1.52. 상태를 얼려서 들고 오면 특징 drift로 stale해진다.
3. **fit 품질(메모리를 더 잘 채우기)은 병목이 아니다**. 값은 **각 pass가 보완적(서로 다른) 일**을
   하게 만드는 데서 나온다. boost가 정확히 그것(각 메모리가 잔차의 다른 층 담당).

### 그러니 ROUND 4가 탐색할 곳 (자유롭게, 단 위 3규칙 준수 + TTT×loop 고유):
- **Boosting 확장**: multi-memory boosting(모든 이전 메모리 합의 잔차), subspace/head-partitioned
  writes(loop마다 fast-weight의 다른 부분공간/헤드만 갱신 → 간섭 없이 용량 ×n), 직교-잔차 targeting
  (이전 예측 방향 성분 제거 — "방향만 산다" 규칙 역이용), block-partitioned boosting.
- **메모리-앙상블 readout**: 최종 apply가 모든 per-loop 메모리 출력 f_{W_ℓ}(q)을 결합(읽기측 boost).
- **fast weight가 활성값이라는 점 극한 활용**: loop 사이로 미분되는 메모리, 메모리 출력의
  cross-loop 직교/비상관 aux loss(train-only, 추론 iso), 메모리를 미분가능 scratchpad로.
- **update/apply 비대칭**: transductive write(읽을 query 분포에 write), loop마다 무엇을 write vs
  read할지, write-token subset vs apply-all.
- **state→loop 피드백**: 메모리 상태 통계(예: 방향 drift)로 다음 pass를 조건화(단 generic gate와
  달리 TTT 상태에서 나온 신호여야).
- **multi-timescale 메모리**: 느린/빠른 메모리, loop마다 다른 rate/scale로 write.
- **key/query 기하 across loops**: 메모리가 보완적이 되도록 loop마다 update-key/apply-query를 다르게.
Hard filters: task-agnostic(카메라/geometry 금지), iso train&inference(≤~1.1×), minimal(한 문장),
그리고 "vanilla transformer로 포팅 가능한가?" 테스트를 통과하면 그건 우리 것이 아니다(gates류).
