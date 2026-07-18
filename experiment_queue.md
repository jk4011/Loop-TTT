# Experiment Queue

> GPU가 비는 대로 위에서부터 실행. 결과는 RESULTS.md에 기록.
> 표준 프로토콜: `lact/lact_nvs/chain_run.sh <gpu> <exp> <config> [seed]`
> = 30k iters, bs16, lr1e-4, warmup1.5k, LPIPS from 5k, RE10K 256², 8in+8tgt(15 views), seed 95
> → eval: 256 held-out scenes, 8in/4tgt, PSNR/LPIPS (paired per-scene) → outputs/<exp>/eval.json
> 명명: r<round>_<config요약>_s<seed>

## RUNNING (seed 검증 wave, 2026-07-18 밤 — 전부 ~22:40 종료 예상)

| GPU | exp | config |
|---|---|---|
| 0 | r1_lact_l8_s96 | L8 seed 96 |
| 1 | r1_loop_l2x4_s96 | reset loop seed 96 |
| 2 | r1_lact_l8_s97 | L8 seed 97 |
| 3 | r1_loop_l2x4_s97 | reset loop seed 97 |

## QUEUE (우선순위순 — GPU 빌 때마다 pop; 라운드 3 config 준비·검증 완료)

1. **r3_loop_l2x4_carry_rho2_s95** — I2' 수정판: post-NS residual scaling (프로브가 지목한
   병리를 정확히 겨냥; carry 구제의 결정적 시험) (`loop_l2x4_carry_rho2_d256_p16.yaml`).
2. **r3_loop_l2x4_delta_s95** — 챔피언(reset)에 I3 delta writes.
3. **r3_loop_l1x8_s95** — 극한 tying: 1블록×8loop.
4. **r3_loop_l4x2_s95** — 완화 tying: 4블록×2loop.
5. **r4_loop_l2x4_sup_s95** — 챔피언 + I6 per-loop supervision (구현·스모크 완료):
   `chain_run.sh <gpu> r4_loop_l2x4_sup_s95 config/loop_l2x4_d256_p16.yaml 95 --loop_sup_weight 0.5`
6. 보류: reset+lrs. r4 후보: r3 결과 따라 delta/rho2 조합, I8 write→read split,
   test-time loop-count sweep (n_loops을 eval에서 바꿔 외삽 확인), 승자 seed 검증.

## 3-seed 판정 기준 (교훈, 2026-07-18)
- L8 앵커 seed 분산 0.22dB — **단일 seed Δ<0.2dB는 절대 결론 금지.**
- 앞으로 승자 후보는 s95에서 이기면 s96/s97 즉시 큐잉, 3-seed paired 평균으로만 헤드라인.

## 완료 후 규칙
- eval.json 나오면 RESULTS.md에 PSNR/LPIPS + baseline 대비 paired Δ 기록.
- 유망(ΔPSNR > +0.15dB) → seed 3개 검증 큐에 추가.
- checkpoint는 최종(model_0030000.pth)만 유지, 중간 것 삭제 (lustre 용량 관리).
- 노드 리셋 대비: /tmp/re10k 사라지면 `data_preprocess/reshard_re10k.py` 재실행 (~3분).

## DONE (상세는 RESULTS.md)

- r1_lact_l2_s95: 19.719 / 0.3860 (하한 앵커)
- r1_lact_l8_s95: 21.955 / 0.2839 (상한 앵커)
- r1_loop_l2x4_s95: **22.204 / 0.2877 — L8 역전 +0.249dB (t=14.2)**
- r1_loop_l2x4_carry_s95: 21.751 / 0.3007 (reset 대비 −0.45dB — core challenge 실증)
