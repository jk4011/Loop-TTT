# Experiment Queue

> GPU가 비는 대로 위에서부터 실행. 결과는 RESULTS.md에 기록.
> 표준 프로토콜: `lact/lact_nvs/chain_run.sh <gpu> <exp> <config> [seed]`
> = 30k iters, bs16, lr1e-4, warmup1.5k, LPIPS from 5k, RE10K 256², 8in+8tgt(15 views), seed 95
> → eval: 256 held-out scenes, 8in/4tgt, PSNR/LPIPS (paired per-scene) → outputs/<exp>/eval.json
> 명명: r<round>_<config요약>_s<seed>

## RUNNING (2026-07-18 ~17:40 시작)

| GPU | exp | config | 예상 종료 |
|---|---|---|---|
| 0 | r1_lact_l8_s95 | lact_l8_d256_p16 (baseline 상한) | ~2.2h |
| 1 | r1_lact_l2_s95 | lact_l2_d256_p16 (param-matched 하한) | ~20min (L2라 빠름) |
| 2 | r1_loop_l2x4_s95 | loop_l2x4_d256_p16 (naive loop, reset) | ~2.2h |
| 3 | r1_loop_l2x4_carry_s95 | loop_l2x4_carry_d256_p16 (I1 carry) | ~2.2h |

## QUEUE (우선순위순 — GPU 빌 때마다 pop)

1. **r2_loop_l2x4_carry_lrs** — I4 per-loop LR schedule (carry 위에). 코드: loop_lr_bias 구현 필요.
2. **r2_loop_l2x4_carry_rho** — I2 residual-gated LR (carry 위에). 코드: ρ-gating 구현 필요.
3. **r2_loop_l2x4_delta** — I3 delta writes (carry 위에). 코드: v_eff 구현 필요.
4. **r2_loop_l2x4_inject** — input injection (Huginn식) 단독 효과 확인 (reset 위에; naive baseline 강화용).
5. r3: 라운드 1/2 승자 조합 + I5 momentum / I6 per-loop supervision.
6. r3+: L4×2, L1×8 loop-shape sweep (승자 세팅에서).
7. 유망 variant → seed 96, 97 재현 검증.

## 완료 후 규칙
- eval.json 나오면 RESULTS.md에 PSNR/LPIPS + baseline 대비 paired Δ 기록.
- 유망(ΔPSNR > +0.15dB) → seed 3개 검증 큐에 추가.
- checkpoint는 최종(model_0030000.pth)만 유지, 중간 것 삭제 (lustre 용량 관리).
- 노드 리셋 대비: /tmp/re10k 사라지면 `data_preprocess/reshard_re10k.py` 재실행 (~3분).

## DONE

(없음 — 라운드 1 진행 중)
