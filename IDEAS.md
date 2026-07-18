# IDEAS.md — Looped-TTT 가설 목록

> 3개 브레인스토밍 subagent(optimization / architecture / multi-view task 관점) 결과를 종합·중복제거한 목록.
> 랭킹 = 기대 임팩트 × 단순성. 상태는 experiment_queue.md와 동기화.

## 중심 이론 (세 에이전트가 독립적으로 수렴한 관찰)

**TTT layer는 loop에 대해 stateful하다.** Attention은 loop를 돌아도 같은 연산의 반복(stateless)이지만,
TTT의 fast weight는 loop pass마다 다시 update될 수 있다 → **looping = 같은 scene regression에 대한
inner-loop epoch 추가**. 이것이 attention looping에는 없는 고유한 설계 공간이다.

**핵심 병리 (The Core Challenge):** LaCT의 update는
`w ← weight_norm(w + NS(lr·g))` 인데, Newton–Schulz(Muon) orthogonalization이 gradient의
**크기 정보를 지운다** (`zeropower_via_newtonschulz5`는 steps=0에서도 Frobenius normalize).
따라서 residual(fit 정도)이 작아져도 update 스텝 크기가 줄지 않는다 → 단일 스텝 체제(LaCT 원본)에선
문제없지만, **loop로 여러 스텝을 돌리면 최적점 주위를 일정 각도로 궤도 순환(orbit)할 뿐 수렴하지 않는다.**
또한 carry loop에서 k,v가 loop마다 재인코딩되므로 inner 목표 자체가 drift하는 feedback(game) 구조.

→ **방법론 방향: "iterate해도 되는 inner optimizer로 고치기"** — 아래 I2~I6이 모두 이 병리의 fix 계열.

---

## Tier A — 라운드 1-3 (기반 + 핵심 fix)

### I1. Carry vs Reset (기반 2×2 ablation) — `실행 중 (라운드 1)`
- **가설:** fast-weight state를 loop 간 carry(= inner epoch 추가)하면 reset(naive, attention식 loop)보다 낫다.
- **메커니즘:** loop i의 `{w0,w1,w2}`를 loop i+1의 init으로 전달. `ttt_state_mode: reset|carry` 구현 완료.
- **비고:** carry가 지면 그것이 곧 핵심 병리의 증거 → I2~I6이 fix. 라운드 1 = {L8, L2, L2×4 reset, L2×4 carry}.

### I2. Residual-Gated LR (self-annealing) — 유력 flagship
- **가설:** 토큰별 lr에 현재 misfit ρ = 1 − cos(f_w(k), v)를 곱하면 update 크기 ∝ residual이 되어
  carry loop가 수렴형이 되고, naive loop와 고정 스케줄을 모두 이긴다.
- **메커니즘:** update branch에서 `f_k = hidden @ w1` (bmm 1회 추가), `lr ← lr·ρ` (detach).
  잘 fit된 토큰은 이후 loop에서 자동으로 write 중단 (per-token early stopping).
- **수학:** gradient flow의 본질 (‖Δw‖→0 as fit→perfect) 복원. 궤도 순환 → 수축(contraction).
  부산물: loop별 mean ρ가 무료 수렴 신호 → I9 adaptive halting과 직결. ~8줄, +15% TTT update FLOPs.

### I3. Delta Writes (innovation만 기록)
- **가설:** value 타겟을 `v − f_w(k)` (residual)로 바꾸면 loop가 fixed point를 갖고 L 증가에 단조 개선.
- **메커니즘:** update 시 `v_eff = v − pred` 사용. NVS 해석: 8개 view의 중복 정보 중 **novelty만**
  메모리에 기록 (redundancy-aware view integration; Déjà View의 "loop가 중복 계산을 학습" 스토리의 TTT판).
- **비고:** I2와 형제 관계 (I2 = lr 스케일링으로 소프트하게, I3 = 타겟 자체를 바꿈). ~10줄.
  NS의 크기 정규화를 우회하려면 `‖v_eff‖/‖v‖` rescale 필요 → base_lr 재튜닝 리스크.

### I4. Per-Loop LR Schedule (loop-index conditioning의 TTT 원형)
- **가설:** loop별 학습되는 lr bias (예: `softplus(lr_fc(x) + loop_lr_bias[ℓ])`)만으로
  loop 대칭성이 깨져 naive loop 대비 개선. 학습된 스케줄이 감쇠형이면 "loop=epoch, 스케줄 필요" 증거.
- **메커니즘:** `loop_lr_bias: nn.Parameter(zeros(L_max))` + `info["loop_idx"]`. ~6줄, 파라미터 ~수십 개.
- **비고:** 최소 변경 대비 정보량 최대. 모든 다른 아이디어의 base로 stack 가능.

### I5. Cross-Loop Muon Momentum
- **가설:** loop 간 gradient momentum buffer를 carry하고 `NS(μm + g)`로 update하면 naive carry보다 낫다.
- **수학/서사:** Muon은 원래 momentum의 orthogonalization으로 설계됨 — LaCT는 단일 스텝이라 buffer를
  버렸다. **looping이 multi-step 체제를 복원하므로 잃어버린 momentum을 되찾아준다.**
  连속 loop의 gradient는 양의 상관 → momentum이 drift로 인한 회전 성분(oscillation) 감쇠. ~15줄.

### I6. Per-Loop Render Supervision + Test-Time Loop Scaling
- **가설:** 매 loop 후 target 토큰을 decode해 discounted loss(γ^(L−ℓ))를 걸면 각 loop가
  "개선 연산자"가 되어 최종 PSNR 상승 + L_test > L_train 외삽(anytime inference) 가능.
- **메커니즘:** decoder는 LN+Linear라 loop별 decode 비용 무시 가능. ~20줄.
  중간 render 시각화 = coarse-to-fine 여부 무료 해석 figure. stochastic L 샘플링과 조합.

## Tier B — 라운드 4+ (메커니즘 분해 / 조합)

### I7. Two-Timescale: Inner Epochs vs Outer Loops
- **가설:** 총 update 횟수를 맞추면 {outer 6×inner 1} vs {outer 2×inner 3} vs {outer 1×inner 6}으로
  looping 이득을 "epoch 효과 vs feature 재인코딩 효과"로 분해 가능. inner epoch은 attention/MLP를
  건너뛰므로 훨씬 저렴 — 저비용 스케줄 발견 가능성.
- **메커니즘:** `ttt_op_order`에 update 세그먼트 반복 추가. 파라미터 0, ~6줄.

### I8. Write→Read Phase Split
- **가설:** 앞 K loop만 update(write)하고 뒤는 apply-only(read)로 얼리면 uniform보다 낫고 FLOPs도 절감.
- **메커니즘:** loop index에 따라 두 가지 `ttt_op_order` 선택. 자기참조 drift(자기 출력에 overfit) 차단.
  "전반은 scene을 메모리에 쓰고 후반은 읽는다" — 한 문장 설명 가능. ~8줄.

### I9. Self-Halting (TTT fit loss 기반 adaptive depth)
- **가설:** fast weight의 (k,v) fit quality는 scene 난이도 신호 → per-scene adaptive loop count로
  PSNR 손실 없이 평균 연산량 감소. attention loop에는 없는 내재적(intrinsic) halting 신호.
- **메커니즘:** I2의 ρ 로깅 + stochastic L 훈련 + 추론 시 Δρ < τ에서 중단. Huginn류 학습된 halting과 차별.

### I10. Anchored Carry Gate (α-blend)
- **가설:** `w = σ(α)·w_carried + (1−σ(α))·w_init` 학습 게이트가 hard carry/reset 둘 다 이긴다.
- **메커니즘:** TTT layer당 3 스칼라. weight-norm이 두 끝점을 같은 norm shell에 두므로 blend가 자동 스케일.
  학습된 α 값 자체가 "carry가 얼마나 유용한가"의 해석 지표. ~8줄.

### I11. Incremental View Registration (per-loop view scheduling)
- **가설:** carry 상태에서 loop마다 인접 view 서브셋만 update(마지막 loop에 전체)하면
  incremental SfM처럼 각 view가 기존 consensus에 대해 정합되어 전체-chunk 반복보다 낫다.
- **메커니즘:** loop별 `ttt_op_order` view-span 구성. NVS 고유 스토리. I1+I3와 조합 시 시너지. ~25줄.

### I12. KM Damping / Polyak Averaging (안정화 보조)
- **가설:** 활성값 recursion 감쇠 `x ← x + α(F(x)−x)` 또는 apply를 fast-weight 이동평균으로 하면
  naive loop의 불안정(있다면)이 해소.
- **비고:** 라운드 1 로그에서 oscillation 관찰될 때만 투입 (진단 조건부).

---

## 실행 원칙
- 모든 방법은 **미니멀** 유지 (사용자 요구): 한 문장으로 설명 가능해야 하고, 기본 diff ≤ ~30줄.
- 라운드마다 4 GPU × ~2.2h (30k iters + eval). 유망하면 seed 3개 (95/96/97) 검증.
- 유전알고리즘식: 이긴 variant에 다음 아이디어를 stack, 진 아이디어는 폐기하되 기록.
- 진단 로깅(loop별 inner fit ρ)은 가능한 모든 라운드-2+ 런에 포함 (비용 무시 가능, 해석 자료).
