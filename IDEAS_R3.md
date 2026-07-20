# IDEAS_R3 — 라운드 3 (10 에이전트 × 10 아이디어, 2026-07-20)

목표: naive loop +1.0dB @ iso train&inference, task-agnostic. 확정 자산: ep2+sup+gates(+0.283 3-seed)
+ cross-model KD(+0.11 s95) ≈ +0.39. 필요: +0.6 추가.

## 에이전트 간 수렴 테마 (여러 에이전트가 독립 도출 = 강한 신호)

- **T1. Trajectory/waypoint KD** (a1#2, a6#1, a10#1): student loop ℓ 렌더 ← teacher loop t(ℓ)={2,3,5,6}
  렌더. Endpoint KD가 놓치는 "어떻게 refine하는가"를 전수. Teacher return_all_loops로 ~무료.
- **T2. Teacher-init (warm start)** (a1#1, a2#1/#3): L2×6 teacher state dict가 student와 **shape 동일
  검증됨** → weight 이식 + KD로 fine-tune. Basin 상속; floor = 순수 KD.
- **T3. 더 강한/앙상블 teacher** (a1#3/#6, a6#2, a10#2/#7): L2×8+sup teacher(지금 완료 직전),
  스택된 teacher(L2×6+ep2sg), teacher pool 샘플링, progressive 8→6→4 체인.
- **T4. Feature-space KD** (a1#4/#9, a6#5, a10#3): 최종 feature 또는 TTT readout 출력 f_W(q)를
  train-only projector로 매칭. RGB 3채널 병목 우회, "state의 행동" 증류(미시도 공간).
- **T5. Loop-count 커리큘럼** (a2#2/#6/#7, a7#7): 6→5→4 계단/연속 anneal, 마지막 ~60%는 순수 4.
  실패한 uniform stochastic(정상 혼합)과 다름 — 종점이 배포 objective 순수 최적화.
- **T6. 토큰 축 iso-FLOP 재배분** (a3 전체, a9#3/#4/#5): pass의 80.7%가 token-linear (FLOP 감사),
  NS 절감은 사망 확인. 싼 pass(토큰 병합/부분집합)로 5-6 pass를 1.0×에. 문맥 기아 회피 원칙:
  K/V·write 커버리지·read는 항상 전 토큰 (죽은 schedule 계열과의 차이).
- **T7. 훈련역학 수리** (a7): **발견 — per-loop gate/FiLM 파라미터가 wd=0.05 그룹에 들어가 identity로
  계속 끌려감** (gates는 억압된 채 +0.033). wd=0 + lr↑ 존 분리. + EMA eval(무료), clip-skip 진단.
- **T8. Noise-annealed 반복 (디퓨전 원리)** (a8#1/#2, a5#2): pass별 σ 감쇠 feature noise(train만),
  같은-forward 다음-pass 자기증류(consistency). 새 축(수축성), ~0 비용.
- **T9. TTT-state 신물리** (a4#1/#2, a5#4, a9#2): block-partitioned boosting(좌표당 1회 write →
  orbit 구조적 불가), state 통계→다음 pass 조건화(state2gate), fit-residual로 write budget 재배분.
- **T10. 프로브** (a10#4/#9/#10): ×6 vs ×4 per-loop 궤적 / v-target drift / 주파수 대역 오차 —
  +0.578의 정체 규명 (H1 궤적 재배속 / H2 더 나은 write target / H3 고주파 용량).

## 실행 선정 (우선순위순, 근거)

1. **kd_traj** (T1): 현 최강 스택(ep2gs+KD)에 waypoint 항 추가. 코드 몇 줄, 비용 ~0. 즉시.
2. **kd8** (T3): teacher만 L2×8+sup로 교체 — teacher 품질 스케일링 시험. 코드 0줄. g1 비는 즉시.
3. **warm6_kd** (T2): --init_from 플래그 신설, teacher-init + KD. Basin 상속의 결정적 시험.
4. **anneal654** (T5): 6(0-6k)→5(-12k)→4(-30k) 결정적 스케줄. 3줄. 1.15× train, 1.0× inference.
5. **optzone** (T7): per-loop 파라미터 wd=0/lr×8 존. ep2gs 스택에서 단독 측정 (gates 축 증폭 기대).
6. **EMA** (T7): 모든 신규 런에 shadow 추가, eval 이중 측정 (훈련 불변 → attribution 무비용).
7. **loopnoise** (T8): σ=[0.12,0.06,0.03,0] pass별 노이즈 (train만). 새 축, ep2gs와 직교 예상.
8. **micropass** (T6): [B1B2,B1,B1B2,B1,B1B2] 정확히 1.00×에 5 pass. 코드 소, 신규 파라미터 0.
9. **probe_traj** (T10): ×6/×4 per-loop PSNR/LPIPS 곡선 — KD 계열 라우팅 (30분, GPU 틈새).
10. 2차 대기: feat-KD(T4), bpb(T9), state2gate(T9), TokenPyramid(T6 full), ensemble-teacher(T3),
    shortcut-consistency(a10#6), sup-γ curriculum(a7#8), c2f-sup(a6#3).

판정 규칙 불변: s95 단독 → +0.15↑면 s96/97, 스택 후보는 ep2gs(+KD) 위에서 paired.
## 부록: 에이전트별 원본 (compact)

# Agent 1: Distillation science
1. teacher-init-student — init student blocks FROM L2×6+sup teacher ckpt (identical shapes, per-loop arrays sliced to 4) + standard KD. Progressive-distillation's key trick. 1.0×/1.0×. --init_from flag, load_state_dict strict=False before DDP. Risk: off-operating-point at 4 loops (cf stoch-depth fail) → LR warmup.
2. trajectory-kd — match student loop ℓ to teacher loop g(ℓ) ({1→2,2→3,3→5,4→6}), discounted MSE (soft-target sup). Teacher return_all_loops, <1% cost. Risk: over-constrains early loops; weight ≤0.5.
3. best-stack-teacher-kd — teacher = L2×6/L2×8 trained with ep2+sup+gates stack (~22.9-23.1); absorption ~25% → +0.18-0.23. Existing --kd_teacher. Risk: too-large gap transfers worse.
4. feature-kd-final — match final pre-decoder target-token features via train-only linear P + normalized cosine/MSE, w~0.5. FitNets logic; render-MSE compresses through tiny decoder. Risk: feature-basis misalignment → P + normalized loss.
5. progressive-chain-kd — 8→6→4 staged: L2×8+sup → distill into L2×6 (teacher-init+KD) → that teaches L2×4. Small gaps transfer larger fraction. Reuses idea-1 code. Risk: gains may not compound; run direct 8→4 control.
6. cached-ensemble-kd — cache K teachers' renders (s95/96/97 L2×6+sup) to tmpfs, KD target = ensemble mean (or 1-of-K noisy teacher). Ensemble +0.2-0.3 better; online cost ~0. Risk: dataloader determinism for keying; fallback live 1-of-K.
7. consistency-kd — per-loop student renders pulled toward TEACHER FINAL render (target = α·t_final+(1−α)·GT) — consistency-model logic on loop count. ~0 cost. Flag --kd_consist_alpha. Risk: same loss-axis as sup → sub-additive.
8. advantage-gated-kd — weight KD per-patch by teacher-beats-student margin vs GT: σ((e_s−e_t)/τ) patch-pooled detached. Distill advantage, not mistakes. 5 lines. Risk: gate ~0.5 everywhere if errors correlate (safe no-op).
9. state-kd — distill teacher's TTT-layer OUTPUT stream at mapped loops via per-block linear heads + cosine, w~0.1. Only cross-view channel = invariant across models. Risk: internal-rep mismatch; small weight.
10. kd-absorption-probe — diagnostics: (a) log KD-loss floor (optimization vs mismatch), (b) kd_weight sweep 0.5/2/4, (c) oracle 2×-dim student, (d) teacher-init basin probe. Decides where to spend slots.

# Agent 2: Weight transfer & loop-count curriculum
KEY FACT VERIFIED: L2×6+sup teacher ckpt has NO per-loop-shaped params — state dict shape-identical to L2×4 student. Weight transfer = drop-in. train.py --load restores optimizer+iters so need tiny --init_from (weights only) flag.
1. warm6-kd4 — init student from L2×6+sup teacher weights + fine-tune 30k at 4 loops WITH KD from same teacher. Floor=KD(+0.11), upside=basin retention. lr 5e-5. Risk: washout→collapses to plain KD (no regression).
2. anneal-step-654 — loop count 6(0-6k)→5(6-12k)→4(12-30k), deploy at 4. Uniform stochastic failed b/c terminal mixture; here final 18k pure deployment objective. 1.15× train. 3 lines. Risk: switch shocks (switch while lr high).
3. warm6-ft4 — teacher-init, NO KD, lr 5e-5; variant freeze blocks 2k iters (interface recalibration). Cleanest basin test + ablation for #1. Risk: washout→+0.
4. exit4-anytime-teacher — fine-tune EXISTING L2×6 teacher 5k iters with exit-4 loss at parity, eval at n_loops_override=4. Cheapest probe: does deep basin have good 4-loop slice? 0.25× one-time. Risk: collapse is representational → decisive info anyway.
5. l2sp-anchor-ft — teacher-init FT + L2-SP penalty λ‖θ−θ_T‖² on block params (anchor weight space, KD anchors function space; stacks with #1). λ≈1e-4. Risk: λ too strong pins at collapsed teacher@4.
6. homotopy-blend-54 — train with render = α·r5+(1−α)·r4, α annealed 1→0 over 18k, then pure 4. Continuous deformation, no off-distribution jump. 1.15× train. Risk: 5-loop-dominated early objective.
7. decay-mixture-anneal — per-step count ~ {4,5,6} probs annealed (0,0,1)→(1,0,0) over 18k + 12k pure 4. Differs from failed uniform stochastic (stationary mixture). 1.15×. Risk: may inherit stochastic failure.
8. wise-soup-alpha — post-hoc lerp θ_α = α·θ_teacher+(1−α)·θ_student (student fine-tuned FROM teacher), eval sweep α∈{0.1..0.5} at 4 loops. 0× train. Can never lose (α=0=student).
9. film-loopcount-then-anneal — zero-init FiLM on total loop count, mixed-count train 18k → pure 4. Removes weight-sharing conflict that killed stochastic. Risk: sub-additive with gates.
10. short-lowlr-adapt — teacher-init + 6k iters lr 2e-5 const at 4 loops. 0.2× probe: is deep basin retainable at all? Pairs with #4 to interpret.

# Agent 3: Iso-FLOP reallocation
AUDIT (measured): pass(2 Blocks)≈37.3 GFLOPs. Per-Block: Attn 17.3% (qkv 8.7, SDPA 5.8, proj 2.9) · TTT 59.6% (qkv 8.7, update 17.3, NS-5 13.5, apply 17.3, proj 2.9) · MLP 23.1%. 80.7% token-linear. Update grads = chunk-SUMS over 2048 tokens; state fixed-size regardless of token count. NS cut EXCLUDED (muon2 −0.34 measured dead). Baseline already 1 chunk (fewer chunks no-op). Prior fails cut MODEL-side capacity; TOKEN axis untried & holds 80% FLOPs.
1. TokenPyramid — early passes on pooled tokens (¼,½,½,½,1,1 → 6 passes = 1.00×; safe 5-pass ½,½,1,1,1 = 1.03×). Delta broadcast unmerge. Marginal pass ≈ +0.2-0.3. Risk: coarse-write artifacts (reset mitigates).
2. ActiveSet — middle passes process top-50% changed tokens (cached k,v still feed attn/update) → 5-pass 0.99×, no resolution loss. Risk: gather/scatter engineering.
3. write-read-interleave — odd full, even = half-token apply-only re-read of prev weights (0.302/pass) → 6-pass 0.98×. ≠read-heavy (inputs never leave, writes interleaved). Risk: mild carry flavor; read_refine≈0 warning.
4. ttt-only-half-passes — alternate full and TTT-only(½) passes (0.366) → 6-pass 1.02× = 6 cross-view mixings. Risk: skipping MLP under-conditions features.
5. matryoshka-passes — cheap passes: half tokens + half MLP/TTT hidden (nested slicing) 0.355 → 6-pass 1.02×, 3 full passes keep full memory (≠ti1 −0.84 which cut ALL passes). Risk: rides capacity sensitivity.
6. micro-pass-insertion — [B1B2,B1,B1B2,B1,B1B2] = 5 passes at EXACT 1.00×, zero new params, pure schedule. Risk: B1/B2 imbalance; half-unit pass < half gain.
7. input-only-merging — merge only INPUT tokens 2:1 except last pass; targets always full (protects −1.66 sensitivity) → 5-pass 0.98×. Risk: leans on write-depth (−0.92) sensitivity.
8. multigrid-boost — TokenPyramid × boost chained across scales (coarse memory stores structure, fine stores detail; re-eval on current keys makes cross-scale legal) = 1.08×. Risk: residual dominated by resolution mismatch.
9. cached-attn-cheap-passes — cheap passes reuse pooled prev attn output via zero-init gate (0.481) → 5-pass 0.99×. Risk: staleness (extrapolation ghost).
10. merged-update-diagnostic — update from 4:1-pooled (k,v,lr), apply full: go/no-go gate for the merge family (chunk-sum should survive pooling; Jensen gap risk). 4-pass 0.87× ablation first.

# Agent 4: TTT-state physics
1. bpb — Block-Partitioned Boosting: carry, but loop ℓ writes only disjoint 1/n block of hidden units (cols w0/w2, rows w1); apply reads full; +delta. Each coord written ONCE → orbit impossible; block ℓ fits residual of all prev blocks (multi-memory boosting in one forward, capacity ×n). 1.0×. Risk: loop-0 block frozen crude; rank d_h/n per loop.
2. state2gate — per-unit cos(w_col, w_init_col) of w0/w2 (write-load; only informative stat under weight-norm) → zero-init linear → next pass's SwiGLU gate_bias. Data-dependent state→computation feedback (untried axis). ~1.0×. Risk: stat near-constant → clean null.
3. div-loss — train-only aux λΣcos²(o_ℓ,o_m) on per-loop TTT apply outputs (target tokens; already computed). Directly optimizes boost's specialization property. Strictly iso inference. Risk: sub-additive w/ sup.
4. ens-read — final loop's apply adds Σγ_ℓ·f_{W_ℓ}(q_final) (all saved memories re-queried with final queries), γ zero-init. Read-side boost lesson. +2-3%. Risk: representation drift of early memories.
5. boostΣ — boost target subtracts SUM of ALL prev memories' preds on current keys (proper stagewise boosting; fixes last-only double-count; ≠cumboost b/c re-evaluated). +5-8%. Risk: replaces boost, same axis.
6. γ-carry — W_ℓ = W_init + γ⊙(W_{ℓ−1}−W_init), per-unit learnable γ zero-init (=reset). Persistence spectrum at unit resolution in one run; +delta for fixed point. 1.0×. Risk: learns γ≈0 (informative null).
7. proj-boost — boost target removes component ALONG prev prediction direction (v − (v·p̂)p̂). Direction-orthogonality = native currency under NS/weight-norm. 1.0×. Risk: replaces boost; discards magnitude part of +0.099.
8. alt-update — carry+delta, even loops write (w0,w2), odd loops w1 (block-coordinate). Breaks orbit resonance, passes do different WORK. 0.95×. Risk: smells optimization-tweak (dead family).
9. ep-delta — ep2's 2nd epoch regresses innovation v−f_W(k) (within-pass boost); may unlock ep3. ~1.01×. Risk: fit axis sub-additive.
10. detach-boost — detach (wp0,wp1,wp2) in boost's pred_prev (proper stagewise; removes circular objective). 1 line. Risk: small either way.

# Agent 5: Adaptive conditioning (on pass ℓ−1 OUTPUT, not index)
1. AdaGate — gate = static_gate[ℓ] + V·tanh(U·pool(x)), pool=per-ch mean+std, V zero-init. Extends gates(+0.033) index→content. ~1.0×. Risk: sub-additive w/ gates.
2. ΔFiLM — FiLM delta from per-ch stats of Δ=x_ℓ−x_{ℓ−1} (change-conditioned; progress descriptor). MLP r16 zero-init. Risk: Δ ≈ depth-drift not scene progress.
3. AgreeGate — per-token cos(y_ℓ, y_{ℓ−1}) of TTT readouts gates the TTT branch (converged tokens damped/amplified, learned sign). Pure TTT×loop physics. Risk: could re-enter dead averaging family.
4. StateProbe — per-head fast-weight drift cos(vec(W_end),vec(W_init)) → MLP → zero-init bias on x at next pass start. State→loop feedback untried. Risk: drift near-constant (Muon fixed step) → dead signal.
5. TokGate — per-token scalar branch gate 1+g[ℓ]·tanh(w·x_t). Token-local adaptive mixing, cheapest token-granular. Risk: rank-1 too little info.
6. SumTok — per-view pooled summary tokens appended to next pass's attention (gated zero-init). Explicit cross-pass memory channel outside TTT state. ~1.005×. Risk: frozen-summary violates re-expression lesson.
7. NL-Adapter — x ← x + g_ℓ⊙tanh(a_ℓ⊙x+b_ℓ) between passes. Minimal nonlinear content transition; probe of whole family. ~0 cost.
8. ProgBias — per-view convergence scalar r_ℓ=‖Δx‖/‖x‖ → embed → bias. Coarser ΔFiLM ablation.
9. HyperGLU — pooled-x generates rank-4 zero-init delta on Block-MLP SwiGLU gate proj. Content-conditioned operating point. Risk: nlcond −0.059 precedent.
10. ErrWrite — per-token write lr × (1+g·e_t), e_t=1−cos(f_{W_{ℓ−1}}(k_t),v_t) INSIDE NS arg (survives Muon). AdaBoost example-weighting; free w/ boost. Risk: rho≈0 precedent.
NOTE: 1/2/5/7/8 same axis as gates — test as REPLACEMENT too. 3/4/10 = TTT-state-coupled (most novel). 6 = new channel.

# Agent 6: Loss & supervision engineering
1. traj-KD — teacher per-loop renders supervise student per-loop renders (map 1→2,2→3,3→5,4→6), weight 0.3. Teacher return_all_loops (~free). Risk: path misalignment; ablate map.
2. teacher-max — retrain teacher as L2×6+ep2+sup+gates (~23.0) + SWA last-k ckpts; identical KD pipeline. Pure supervision-source engineering. Risk: KD gain may saturate in teacher quality.
3. c2f-sup — per-loop GT sharpness pyramid: loops 1-2 blurred/downsampled GT, final full GT+LPIPS; anneal σ→0 after 20k. Spectral role decoupling (passes do different work). Risk: early blur commitment breaks residual chain.
4. delta-sup — direction-only cos loss: increment (r_ℓ−r_{ℓ−1}) toward current residual (target−r_{ℓ−1}).detach(). Explicit descent-step constraint, residual re-expressed live (obeys Wave-10 law). Risk: same family as sup.
5. feat-KD — FitNets per-loop feature hints via zero-init linear head (dropped at inference). Constrains intermediate computation of tied blocks. Risk: feature-basis misalignment → cosine/CKA fallback.
6. trust-KD — per-pixel KD weight exp(−|t_render−target|/τ): imitate teacher only where it's good; enables kd_weight>1. Risk: gate correlates with easy pixels; sweep τ.
7. loop-LPIPS-lite — LPIPS on ONE random intermediate loop per step (~1.05×), or free Laplacian-edge MSE variant. LPIPS = known naive-loop weakness. Risk: unstable on poor early renders.
8. decorrel-sup — penalize cos²(d_ℓ, d_{ℓ+1}) of successive render deltas → complementary passes enforced in loss (boost's mechanism, loss-side). w 0.05. Risk: satisfiable by noise.
9. hinge-mono — Σ max(0, mse_{ℓ+1}−mse_ℓ+m) per sample: strict monotone improvement. NOTE self_distill in queue unmeasured — run that first. Risk: sandbagging early loops.
10. grad-balance — adaptive per-loop loss weights ∝1/EMA(L_ℓ) (GradNorm-lite) or Kendall; + Charbonnier variant. Reweights same info — weakest family, run last.

# Agent 7: Training dynamics (outer loop)
CODE FINDING: loop_film/branch_gate/state_gate/loop_rho are ≥2-dim → in wd=0.05 AdamW group → optimizer actively decays per-loop conditioning toward zero-init identity. Gates got +0.033 WHILE suppressed.
1. ema-eval — EMA shadow (β≈0.9995) of outer weights, eval EMA copy. Distinct from failed Polyak (that was fast-weight space). FREE PILOT TODAY: average last ~5 ckpts of existing run, re-eval (tail-SWA). Risk: cosine-to-zero already quasi-averages.
2. optimizer-zones — move per-loop params (loop_film, branch_gate, state_gate, loop_rho) to wd=0 group with 5-10× lr. Could double/triple conditioning axis, may revive loop_film. 1-line grouping change. Risk: decay was accidental regularization.
3. per-pass-weight-noise — pass ℓ runs with W+ε_ℓ (fresh per pass, σ∝RMS, annealed→0 by 20k); decorrelates 4 positions' gradients + flatness. Depth stays 4 (≠ failed stochastic count). Risk: σ tuning, bf16 swamping.
4. recipe-retune — bs8×60k iters, lr 7e-5, warmup 8k at iso-FLOPs (recipe was tuned for non-looped L8). 2× optimizer steps. Risk: MUST also retune naive baseline for comparability (extra run).
5. grad-scale-rebalance — GradScale identity-fwd op per pass, equalize per-position grad contributions (last pass dominates 8:1 w/ γ=0.5). First run free diagnostic: per-position grad norms via hooks. Risk: compounding to earlier passes.
6. stoch-grad-truncation — detach stream at ONE random boundary/step (p≈0.5), sup keeps signal; forward bit-identical. TBPTT cure for interference. 0.9× train. Risk: gradient starvation of write path.
7. progressive-loop-growth — n_loops 1(0-2k)→2→3→4(6k-30k); last 80% at eval depth; ≠ stochastic (brief monotone ramp, fixed thereafter). Saved FLOPs → extra final iters. Risk: transition spikes.
8. sup-gamma-curriculum — γ 1.0→0.3 anneal by 20k (uniform early = balanced position gradients when interference worst). 2 lines. Risk: sup near-saturated.
9. free-sam — perturb by ρ·g_prev/‖g_prev‖ (recycled grad, no extra backward), anneal ρ→0. Sharp landscape target. Risk: stale direction; sweep ρ.
10. fix-clip-skip — currently pre-clip norm>4 → WHOLE STEP DISCARDED; log skip rate, if >1-2% take clipped step instead (skip only non-finite; circuit breaker >20). Free diagnostic for interference hypothesis either way.
NOTE: {3,9} flatness pair, {5,6,8} interference pair — measure within family first.

# Agent 8: Noise/diffusion-style iteration
1. anneal-noise (loop-as-denoiser) — train-only per-pass feature noise x += σ_ℓ·rms(x)·ε, σ geometric decreasing (0.12,0.06,0.03,0), inference clean. New axis: contraction/robustness. ~0 cost. Differs from drop_loop: noise ADDS work, dropout deletes iteration. Risk: train/test mismatch → σ_last=0 + sup mitigate.
2. traj-consistency-selfdistill — Σ‖render_ℓ − detach(render_{ℓ+1})‖² within same forward (loop index = consistency time). NOT failed self-distill (that extrapolated beyond trained depth; this interpolates). ~0 cost. Synergizes w/ #1. Risk: sub-additive w/ sup.
3. fw-init-jitter — train-time reset init w := colnorm(w_shared + σε) per pass (stays on weight-norm sphere); inference clean. Implicit memory ensemble, decorrelates n memories. pli failed b/c deterministic; jitter keeps meta-learned mean. Risk: NS washes perturbation → null.
4. Δ-FiLM (noise-level conditioning) — FiLM conditioned on rms(x_ℓ−x_{ℓ−1}) via zero-init linear (data-dependent σ-conditioning, diffusion's essential knob). Works at inference too. Risk: sub-additive w/ gates.
5. kv-bagging — train-only jitter on UPDATE-segment k,v (bootstrapped memory fits); read path clean. Decorrelates memories via data (orthogonal to boost's targets). Risk: NS normalizes → null.
6. increment-jitter (ShakeDrop) — α_ℓ~U(1−δ,1+δ) mean-1 scale on loop residual, δ annealed [0.3,0.2,0.1,0]. Keeps every pass contributing (vs drop_loop removal). Risk: same family as drop_loop.
7. learned-heun — pass pairs as Heun steps: blend ½(Δ1+Δ2) with zero-init learned α. TRAINED-THROUGH recombination of computed increments (vs dead post-hoc extrapolation). Risk: highest — extrapolation family dead; α→0 safe null.
8. snr-sup-weighting — sup per-loop weights = 1/(1+σ_ℓ²) matching #1's noise schedule (min-SNR). Bundle-only.
9. noise-curriculum — input noise annealed over TRAINING iters (0.2→0 by 10k), pass 0 only. Zero final mismatch. Piggyback.
10. twin-traj-distill — 10% of steps: 2nd noisy forward, distill toward mean render. Ensemble→KD channel. 1.1× train. Risk: weak 2-sample teacher.

# Agent 9: Token routing per pass
FLOP AUDIT: per Block ≈9.3 GMAC = attn 1.61 + TTT-proj 1.07 + TTT-apply 1.61 + MLP 2.15 + TTT-update 1.61 + NS 1.26. Write side 31%, target tokens 34%. Baseline 4×2 blocks ≈74.5 GMAC.
Anti-starvation principle: route which tokens get RECOMPUTED / how much WRITE budget — never which tokens exist in context (K/V, write coverage, reads always span all tokens).
1. transductive-write — last pass adds TTT update segment over TARGET tokens' own (k,v) (write on the query distribution you serve). +7.7% (or ~1.0× w/ #6). Zero-init per-loop lr gate. Risk: self-referential error reinforcement.
2. residual-weighted-write — per-token write lr × (1+γ_ℓ(r_i/r̄−1)), r_i = prev-pass memory fit residual. Enters BEFORE NS → changes gradient direction (survives Muon; why scalar rho was no-op). Free w/ boost. Risk: sub-additive w/ boost.
3. freeze-and-fund — passes 3-4 recompute only top-50% tokens by Δ (frozen keep cached feats, STILL serve as K/V + write sources); saved 40%×2 passes funds full 5th loop (~1.08×). Extra genuine loop = biggest known gain. Risk: high engineering (gather/scatter+compile).
4. c2f-target-ladder — passes 1-2 target tokens 2×2-merged (shifted grids), 3-5 full; sup at matching scale; funds 5th loop (1.12×) or strict 0.87×. Fixes late-join failure (targets present, just cheap). Risk: seam artifacts.
5. hard-token-micropass — after pass 4, extra micro-pass on worst 25% tokens ranked by TTT fit residual (free difficulty oracle from state!) w/ cached full K/V. +7-8.7%. Risk: sub-additive w/ #3.
6. coarse-write-ladder — passes 1-2 update from 2×2-merged (k,v) super-tokens (re-normalized), 3-4 full. Multiresolution memory; coarse fittable in one NS step (anti-orbit). ≤1.0×. ≠chunked (compressed, not partitioned). Risk: pooled kv off key manifold.
7. Δ-gated-token-intensity — per-token residual gate g_i=σ(a_ℓ·Δ̂_i+b_ℓ) on pass increment (soft freezing). 0 FLOPs, 2 scalars/loop. Conditioning axis, content-conditional. Probe for #3/#5.
8. residual-routed-twin-adapters — rank-8 adapter pair mixed per token by TTT residual rank (function routing). +0.3%. Risk: too small capacity at dim 256.
9. complement-write-bagging — pass ℓ updates from rotated 75% overlapping subset (union 2 passes=100%); apply/attn see all. Bagged boosting w/ reset+boost. −4%. Risk: schedule-family ghost; test 87.5%.
10. cross-pass-kv-mixing — half heads use cached prev-pass K/V (learned λ zero-init). Trajectory attention. ≤1.0×. Risk: frozen-state lesson (ranked last).

# Agent 10: Theory + literature
Hypotheses for +0.578: H1 re-paced trajectory (gain in late passes, perceptual). H2 better WRITE TARGETS (deeper features → better (k,v)); fit quality flat. H3 high-freq compositional capacity (LPIPS 0.2877→0.2718 beats even L8).
1. LoopTraj-KD — waypoint distillation: student loop ℓ ← teacher loop t(ℓ)={2,3,5,6} render, w≈0.3 + keep endpoint KD. RELAY/progressive-distill import. Risk: fights student's own pacing.
2. stoch-ensemble-teacher — per-step sample 1 teacher from pool {L2×6+sup, L2×8+sup, best gen-1 student 22.501 (born-again)}. Raises effective teacher excess to ~+0.8-0.9 at zero marginal cost. TAKD mitigation built in.
3. ttt-bottleneck-feature-KD — match final-loop TTT-layer outputs f_W(q) on target tokens (post c_proj) via train-only projector. Distills the STATE'S BEHAVIOR (all cross-view info flows through TTT read). Untried KD space.
4. PROBE per-loop trajectory ×6 vs ×4 — return_all_loops eval both ckpts; if ×6 loop-4 < ×4 final → H1 (trajectory KD); if ≥ → function KD. 30 min.
5. LPIPS-KD — KD in VGG/LPIPS space between student and teacher renders (w≈0.5). ×6 gain is disproportionately perceptual; MSE-KD transfers only L2 component. ~1.3× train. Risk: overlaps GT-LPIPS axis but teacher target achievable.
6. shortcut-consistency dual-budget — train at B∈{4,6} p(6)=0.25, condition gates on (loop,B), align B4 final to B6 route via stop-grad; deploy B4. LoopFormer(2602.11451)/ELT(2604.09168) recipe = the FIX for stochdepth failure (conditioning + alignment loss). 1.12× train. Highest variance.
7. progressive-halving-KD — ×8→×6′(KD)→×4 chain, existing infra, zero new code. TAKD small-hops. Risk: absorption saturation.
8. final-features-memory-rewrite — ONE extra TTT-only micro-pass after loop 4: update from FINAL (deepest) input features, apply to final targets, zero-init gate. H2-derived: reset-mode memories always fit on shallower features than the read consuming them. ~1.07× inference (borderline). ep2 refits same features; this refits BETTER features.
9. PROBE fit-vs-target on ×6 — probe_loop_fit.py + v-target drift per loop; discriminates H2. Minutes.
10. PROBE frequency-band error per loop — FFT band-split error per loop both ckpts; routes among KD variants. <1h.
