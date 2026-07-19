# 100-idea synthesis (10 subagents) — selection into experiment_queue

목표: naive loop +0.5dB @ iso-compute. 선택 기준: (a) 확인된 winner(boost/ep2/sup/gates)와
**직교축**, (b) 미니멀·0~tiny param·zero-init degrade, (c) ≤1.1× FLOPs, (d) 실증 실패군과 무관.

## Agent 3 — Multi-view geometry (완료)
새 직교축 2개 발굴: **메모리 ADDRESSING**, **target-conditioned write(task-structure)**.
- **[강력] G1. Plücker reciprocal-product addressing**: 광선 교차 bilinear form
  `d_t·m_i + m_t·d_i`(epipolar 일치=0)를 TTT key/query에 주입(6채널, γ zero-init). target 쿼리가
  입력 픽셀의 epipolar line에 집중. 이전 프로젝트 attention-RoPE +1.7dB의 TTT판. 축=ADDRESSING(신규).
- **[강력] G6. Loop-refined disparity latent**: target 토큰당 depth latent z를 loop마다 정제,
  쿼리 sample point `p=o+z·d`로 epipolar line 위의 점을 찾음. G1(line)과 직교(point). loop-specific.
- **[중] G3. Target-relative write priming**: 입력 토큰을 target centroid 상대 pose로 key에 조건화
  (update segment only, zero-init). 유한 메모리를 target이 쿼리할 픽셀에 배분. 축=task-structure(신규).
- **[중] G7. Relative-pose value steering**: view-dependent radiance를 ray_d로 v에 FiLM/회전(+역변환 on read).
  Muon이 크기만 지우므로 방향 steering은 생존. 축=content/value(신규).
- **[보조] G4. Undiluted Plücker re-injection**: pose를 TTT layer에서 매 loop 재주입(zero-init linear).
  G1/G3/G6의 pedestal. 축=input-feature.
- 나머지 G2(Plücker-RoPE, G1과 같은 addressing 축 경쟁), G5(per-loop coarse-to-fine addressing,
  G2 확장), G8(input-side consistency feedback), G9(geometric read nudge), G10(recentering, 이미 됨).

## Agent 6 — Information flow (완료)
- **[강력] I2. Anderson/Richardson feature extrapolation**: `x += β[loop_idx]·(x − x_prev)`,
  β zero-init. loop 자체 수렴방향 외삽 → 공짜 유효깊이. <1.01×. 축=feature-acceleration(신규, 모두와 stack).
- **[강력] I1. Residual write-back**: 메모리 miss `r=v−f_W(k)`를 zero-init proj+tanh gate로 input
  토큰 output에 더함(디코드 RGB 아님 → render_feedback −0.20과 구분). ~1.05×. 축=feature-write(신규).
- **[강력/train-only] I3. Supervise-the-fit**: `L_fit=mean(1−cos(f_W(k),v))`를 aux loss로.
  eval FLOPs 0. 축=loss/memory-space(sup=output-space와 직교).
- **[중] I4. Δx-gate (adaptive depth)**: per-token `‖x_ℓ−x_{ℓ-1}‖`로 TTT output gate(zero-init).
  수렴 토큰 down-weight = 가변깊이 @ iso-compute. 축=data-dependent conditioning.
- **[중/train-only] I7. Input-token self-recon loss**: 입력 토큰을 decoder로 RGB 복원 aux loss.
  축=loss/input-render. **[upgrade] I6. Cumulative-residual boost**: boost 타겟을 Σ_{j<ℓ}f_{W_j}(k)로
  (모든 이전 스테이지). boost 축 upgrade(stack 아님), ~1.1× 경계.
- 나머지 I5(misfit feature gate, rho를 lr 아닌 gate로), I8(residual-refocused ep2, ep2 축 경쟁),
  I9(render self-distillation, sup 축), I10(key-space decorrelation, 투기적).

## Agent 2 — Fast weight as operator (완료)
- **[강력/신규축 DEPTH] O3. OpCompose**: loop가 연산자를 합성 — loop ℓ가 `k̃=norm(f_{W_{ℓ-1}}(k))`로
  update, 최종 apply는 `f_{W_n}∘…∘f_{W_1}(q)`. γ zero-init blend. 4 loop에서 depth-8 fast-net.
  ~1.06×. 축=depth(신규, 단독 +0.5 가능성). 高보상/高분산.
- **[강력] O4. OrthoWrite**: carry update grad를 현재 weight column space에 직교 투영
  `g⊥=g−W(Wᵀg)/‖W‖²`, γ zero-init. 궤도 병리(0.70→0.41) 직격 → carry 부활. ≤1.01×. 축=carry-stability.
- **[중] O2. OpBank-Read**: apply에서 연산자 bank 앙상블 `o=Σβ_ℓ f_{W_ℓ}(q)`, β zero-init(β_last=1).
  가장 민감한 target-read(−1.66) 무료 심화. ≤1.02×. 축=read/ensemble.
- **[upgrade] O1. FullBoost**: boost 타겟 `v−Σ_{j<ℓ}f_{W_j}(k)`(모든 이전). boost 축 upgrade. ~1.05×.
- 나머지 O5(OpAvg Polyak, carry 궤도 중심 평균, α=0이 carry라 위험), O6(DecorrKeys 고정 직교회전으로
  bank 패턴 직교화, boost와 조합), O7(EpAvg, ep2 축), O8(OpAnderson 연산자 외삽), O9(OpMoE query-routing),
  O10(LowRankCarry).

## Agent 8 — Multi-scale / frequency (완료)
- **[강력/신규축 spectrum] M1. Spectral coarse-to-fine Muon**: NS 스텝을 loop마다 스케줄
  {2,4,6,8}(총합 20=baseline 5×4) → 초기 low-freq, 후기 high-freq update. **NS는 update의 모양만
  바꿈(크기 아님) → 죽은 lr-knob 함정 회피.** ~1.00×, ~10줄. 축=update-spectrum(신규). LPIPS 겨냥.
- **[중] M3. Gate temperature scale-space**: `silu(τ_ℓ·qi@w0)` per-loop τ(zero-init log). update측은
  NS가 지우지만 read 비선형 모양은 실제 변함. ~1.00×. 축=read-nonlinearity.
- **[중] M2. Detail-emphasis target**: `vi += λ_ℓ(vi−blur(vi))` per-loop(zero-init), 후기 loop 강화.
  ~1.00×. 축=target-detail. **[중] M4. Head-split wavelet**: coarse heads→blur(v), fine heads→highpass(v).
- 나머지 M5(band-boost, boost축), M6(ep 재배분, ep2축), M7(view-consensus split), M8(freq sup, sup축),
  M9(spectral init tilt, pli 인접 위험), M10(Laplacian residual, gates 인접).

# ★★ 수렴 신호 (독립 3+ 에이전트가 동일 flagship에 도달) ★★
**Preconditioned / exact-LS inner update** = Agent1#1(Gauss-Newton whitening) = Agent10#1(RLS-Readout/Kalman)
= Agent4의 feature-solver와 결합. `w1`은 SwiGLU feature H의 선형 readout → 최적 update는
`(HᵀH+λI)⁻¹HᵀV` (닫힌형). NS는 scale-invariant라 preconditioner 크기는 지워지고 **방향만 살아남음**
→ 죽은 lr-knob 함정 회피. binding underfit(−0.84 자원) 직격 + carry에 고정점 부여(궤도 죽임).
Neumann series 2~3항으로 solve 없이 compile-safe 구현 가능. **1순위 구현.**

## Agent 1 — Inner-optimizer 이론 (완료)
- **[★flagship] A1. Gauss-Newton feature whitening**: NS 전에 raw grad를 inverse feature-Gram으로
  좌곱 `NS((Ph+λI)⁻¹·hiddenᵀvt)` 등. delta on이면 정확한 GN step. 축=conditioning(신규). ~1.05×.
- **[강력/신규축] A2. Mahalanobis inner loss (right factor)**: 타겟을 value-Gram으로 우곱
  `vt@(Pv+λI)⁻¹`. inner-loss metric 변경 = sup(outer)와 직교, A1(left)과 직교(Shampoo L⊗R). ~1.03×.
- **[강력/신규축] A3. Polyak iterate averaging**: ep 반복의 (w1) iterate 평균을 apply에 사용.
  ep3 포화(궤도 초과)를 fit으로 전환 → E=3,4 단조화. 축=apply/trajectory(신규, read_refine과 다름).
- 나머지 A4(diagonal Fisher, A1 저비용판), A5(Gauss-Seidel sweep), A6(conjugate multi-step, ep2축),
  A7(Anderson carry), A8(two-sided Shampoo=A1∘A2), A9(best-iterate selection), A10(loop-progressive Gram).

## Agent 4 — Dynamical systems / DEQ (완료)
- **[강력/신규축] D1. Anderson acceleration on FEATURE state**: `x_{ℓ+1}=Σβ_j x_{ℓ-j}+ηΣβ_j g_{ℓ-j}`
  (feature 고정점, magnitude-preserving, Muon 무관). 4→8 loop 갭 절반이면 ~+0.2. ~1.00×, ~35줄.
  fallback=Nesterov `x̃=x+μ(x−x_prev)`. 축=feature-solver(신규).
- **[중] D3. Feature over-relaxation**: `x_{ℓ+1}=x_in+diag(α_ℓ)Δ_ℓ`, α zero=1. lrs를 feature축으로
  이동(거긴 magnitude 살아있음). ~1.00×. 축=feature step-size(신규).
- **[중/신규축] D4. Learned readout aggregation**: 마지막만 decode 대신 `x̄=Σw_ℓ x_ℓ`(softmax, one-hot init).
  궤도 대신 안정 평균 readout. ~1.00×. **[train-only] D5. DEQ self-distillation**: no_grad 8~12 loop
  x*_deep를 4-loop 타겟으로. inference iso. D6(fixed-point consistency aux loss, train-only).
- 나머지 D2(Mann weight damping=O4와 유사), D7(implicit DEQ), D8(two-timescale), D9(Aitken readout), D10(halting).

## Agent 10 — Cross-domain wildcards (완료)
- **[★flagship] W1. RLS-Readout**: w1을 정확 LS `(HᵀH+λI)⁻¹HᵀV`로(=A1의 readout판, MesaNet).
  carry형=정보필터 `A_ℓ=γA+HᵀH, b_ℓ=γb+HᵀV`. binding underfit 직격. ≤1.05×. gate zero→baseline.
- **[강력] W2. GN whitened features**(=A1). **[중] W4. Heun predictor-corrector**: raw grad 2개 평균 후
  1 NS = 정확한 단일스텝(ep3 overshoot보다 나음). **[중/신규] W6. Coarse-to-fine memory (multigrid)**:
  update용 k,v를 loop마다 pool factor 2^(L-1-ℓ) → 초기 저주파, 후기 고주파. 초기 pass 더 쌈(~0.98×). LPIPS.
- **[중] W7. Snapshot-ensemble readout**(=D4), W5. Anderson carry(=A7), W8. CLS fast+slow w1, W9. Nesterov momentum, W10. Reservoir+exact readout(진단용).

## Agent 5 — Memory & capacity (대기)
## Agent 2 — Fast weight as operator (대기)
## Agent 4 — Dynamical systems / DEQ (대기)
## Agent 5 — Memory & capacity (대기)
## Agent 6 — Information flow (대기)
## Agent 7 — Latent-reasoning 이식 (대기)
## Agent 8 — Multi-scale / frequency (대기)
## Agent 9 — Ensemble / boosting (대기)
## Agent 10 — Cross-domain wildcards (대기)
