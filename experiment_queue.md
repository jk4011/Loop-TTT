# Experiment Queue

> GPU가 비는 대로 위에서부터 실행. 결과는 RESULTS.md에 기록.
> 표준 프로토콜: `lact/lact_nvs/chain_run.sh <gpu> <exp> <config> [seed]`
> = 30k iters, bs16, lr1e-4, warmup1.5k, LPIPS from 5k, RE10K 256², 8in+8tgt(15 views), seed 95
> → eval: 256 held-out scenes, 8in/4tgt, PSNR/LPIPS (paired per-scene) → outputs/<exp>/eval.json
> 명명: r<round>_<config요약>_s<seed>

## RUNNING (라운드 2, 2026-07-18 저녁)

| GPU | exp | config | 예상 종료 |
|---|---|---|---|
| 0 | r2_loop_l2x4_carry_delta_s95 | I3 delta writes (carry) | ~20:55 |
| 1 | r2_loop_l2x4_carry_lrs_s95 | I4 per-loop LR bias (carry) | ~20:20 |
| 2 | r2_loop_l2x4_inject_s95 | input injection (reset 챔피언 강화) | ~21:05 |
| 3 | r2_loop_l2x4_carry_rho_s95 | I2 residual-gated LR (carry) | ~20:45 |

## QUEUE (우선순위순 — GPU 빌 때마다 pop)

1. **r1 seed 검증 (헤드라인 보호)**: loop_l2x4(reset) s96, s97 + lact_l8 s96, s97 — 4런.
   "naive loop가 L8 역전 +0.25dB"는 단일 seed라 반드시 3-seed 확인.
2. **r3_loop_l2x4_lrs_s95** — champion(reset)에 I4 per-loop LR bias (lrs는 carry 불필요; config 추가 필요).
3. r3: loop shape sweep (reset): loop_l1x8, loop_l4x2 — 재사용 극한/완화 지점 탐색.
4. r3: r2 결과에 따라 — carry가 구제되면 구제법 조합/seed 검증; 안되면 reset 계열 심화
   (I6 per-loop supervision, I8 write→read split, I7 inner epochs).

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
