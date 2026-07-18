# RESULTS.md — 모든 평가 결과 (진실의 원천)

> 프로토콜: 30k iters, bs16, seed 95(기본), RE10K 256², 8in+8tgt → eval 256 scenes, 8in/4tgt.
> Δ는 같은 seed의 비교 대상과 per-scene paired 차이. 단일 seed ~0.1dB는 노이즈 (이전 프로젝트 F18).
> 참고 앵커(이전 프로젝트, L6/d256/p16 동일 프로토콜): PSNR 21.970 / LPIPS 0.2883.

## 라운드 1 — looping 기반 검증 (2026-07-18 시작)

| exp | config | params | PSNR | LPIPS | vs L8 | vs L2 | 비고 |
|---|---|---|---|---|---|---|---|
| r1_lact_l8_s95 | L8 비루프 | ~13M | (진행중) | | — | | 상한 앵커 |
| r1_lact_l2_s95 | L2 비루프 | ~3.9M | (진행중) | | | — | 하한 앵커(param-matched) |
| r1_loop_l2x4_s95 | L2×4 reset | ~3.9M | (진행중) | | | | naive loop |
| r1_loop_l2x4_carry_s95 | L2×4 carry | ~3.9M | (진행중) | | | | I1 |

### 판정 기준
- loop(reset) > L2: looping이 TTT에서 기본 동작함 (RQ1 yes)
- carry vs reset: TTT 고유 상태 축의 효과 (I1)
- loop vs L8: 갭 크기 = 이후 방법론이 메꿔야 할 목표 (Déjà View처럼 역전이 최종 목표)
