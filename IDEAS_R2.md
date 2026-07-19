# Round-2 100-idea synthesis (task-agnostic, post-wave-10) — 10 subagents

제약: task-agnostic(카메라/geometry 금지), 비-최적화(가속/precond/frozen-carry 전멸), 직교축 stacking.
확정 baseline = ep2+sup+gates = +0.283 (3-seed). 통합 이론(agent 9): loop는 inner opt를 푸는 게
아니라 각 pass가 새 비선형 feature-interaction을 주입하는 프로그램. 가치=계산 novelty. → (C1)입력측
선형/스케일 죽음, (C2)이득은 비선형/출력측/cross-pass 다양성·표현력·비선형깊이에서만.

## ★ 다수 에이전트 수렴 (최우선)
- **StochDepth (loop 수 샘플링 훈련)** [5-6 agents: 2,3,4,5,10,6]: {2..6} 샘플, eval 고정 4. depth-robust,
  0-param. 우리가 fixed-L 훈련해서 test-time 외삽이 실패한 원인 직격. 축=regularization(신규).
- **DropLoop (stochastic depth on stream/branch)** [agents 4,5,10]: 훈련 중 loop 확률적 skip. 축=regularization.

## Agent 6 — looped-LLM transfer (완료, 최강 외부증거)
- **[★] Attn-OutGate (LT2 SDPA gate)**: per-head sigmoid gate(zero-init) — 우리 attention은 gate 없이 4×
  반복돼 RMS 누적. LT2 +1.43 정량. 축=attn-branch(신규). **구현·실행중(g0)**.
- **ConcatInject**(Huginn concat adapter, ADD변형만 실패했음), **CoreNorm/Sandwich**(RMSNorm@depth),
  **ScratchTokens**(persistent latent workspace), **PathIndep**(init noise, train-only).

## Agent 9 — theory-first (완료)
- **[★] NL-Cond**: SwiGLU preact에 per-loop 가산 bias(gate_before_act += b0[loop]). 재흡수 불가(가산),
  gates(선형)와 직교 stack. 0 FLOPs. 축=nonlinear-conditioning(신규).
- **Fused-Readout**: per-loop target feature의 학습 결합으로 decode. 축=output-composition.
- **Novelty-Write**: TTT 출력을 residual에 직교 투영(complementary). 고-천장 다양성 test.
- NL-Out, Deep-Read(f∘f), Div-EP(=hep).

## Agent 5 — winner siblings (완료)
- **lob (Loop-Output Boosting)**: per-loop 예측을 잔차 head로 합산. 축=output-composition(=Fused-Readout).
- **hep (Heterogeneous Epochs)**: ep2 2번째 스텝을 innovation(v−f_k)로. ep2 축 upgrade, ~0 cost. 유력.
- **bag (write bagging)**: per-loop token dropout=변량 앙상블. twb(target self-write, 고-천장 but +compute).

## Agent 8 — update/apply schedule (완료)
- **SCW (Self-Consistent Write)**: target 토큰도 메모리에 write(현재 feature 재계산, gate zero-init).
  target은 지금껏 read만 함 — 최대 미탐 구조 변화. ~+15% compute. 축=write-coverage(신규).
- **TR/MSR (read-maturity)**: ep2의 W1,W2 두 성숙도에서 read 블렌드. **IO2**(=hep, ~free ep2 upgrade),
  **UES**(per-loop epoch 배분, iso). DPR(dual-projection read).

## Agent 1 — tie/untie (완료)
- **QKV-Route**: to_qkv per-loop LoRA(r=8). 축=transform(신규). **구현·실행중(g2)**. (이론상 입력측 선형=중립 위험)
- **WN-Radius**(post-NS 반지름), **LoRA-Init**(공유 init+저랭크 delta, boost겹침), BranchLoRA(gates 일반화).
  경고: lr_fc untie 금지(pre-NS 죽음).

## Agent 3 — diversity (완료)
- **RotBag**: per-loop 고정 랜덤 직교회전 q/k, 0-param. **구현·실행중(g3)**. (이론상 입력측 회전=중립 위험, 결정 test)
- **DecorPen**(증분 decorrelation loss), SnapEns(snapshot 앙상블), DiagFlip(±1 sign, 저렴 probe).

## Agent 2 — loss structure (완료)
- **distill (deep-teacher)**: no_grad teacher n_loops+K 반복 → per-loop distill. **+0.44를 iso-inference로**.
  train-only. 축=target-source(신규, sup과 직교). 고-천장.
- stoch-depth(=StochDepth), c2f-target, consist/mono, ema-teacher.

## Agent 4 — feature-stream (완료)
- **StreamNorm**: per-loop RMS 재정규화(residual saturation 공략, Huginn 선례). 축=normalization(신규).
  TwoRate(slow/fast dual stream), U-Loop(비인접 skip), LoopBasis(per-loop re-basing).

## Agent 7 — adaptive compute (완료)
- **phalt (per-token soft halting + ponder)**: 수렴 토큰 over-processing 보호. 축=adaptive-depth(신규).
  router(MoR, 고-천장/고위험), cgate(data-dependent gate). loopdrop(=StochDepth).

## === 선별 실행 순서 (직교축 우선, 이론 정합) ===
WAVE-A (실행중): Attn-OutGate(g0), QKV-Route(g2), RotBag(g3).
WAVE-B: NL-Cond, StochDepth, distill, hep.
WAVE-C: StreamNorm/CoreNorm, Fused-Readout/lob, DropLoop, ConcatInject, phalt, SCW.
각 단독 s95(vs naive 22.204) → 양성이면 ep2+sup+gates(+0.283)에 스택 → 3-seed.
