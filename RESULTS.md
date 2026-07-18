# RESULTS.md — 모든 평가 결과 (진실의 원천)

> 프로토콜: 30k iters, bs16, seed 95(기본), RE10K 256², 8in+8tgt → eval 256 scenes, 8in/4tgt.
> Δ는 같은 seed의 비교 대상과 per-scene paired 차이. 단일 seed ~0.1dB는 노이즈 (이전 프로젝트 F18).
> 참고 앵커(이전 프로젝트, L6/d256/p16 동일 프로토콜): PSNR 21.970 / LPIPS 0.2883.

## 라운드 1 — looping 기반 검증 (2026-07-18 시작)

| exp | config | params | PSNR | LPIPS | vs L8 | vs L2 | 비고 |
|---|---|---|---|---|---|---|---|
| r1_lact_l8_s95 | L8 비루프 | ~13M | 21.955±0.141 | 0.2839 | — | +2.24 | 상한 앵커 (이전 L6 앵커 21.970과 일치 → L6→L8 깊이 포화) |
| r1_lact_l2_s95 | L2 비루프 | ~3.9M | 19.719±0.133 | 0.3860 | | — | 하한 앵커(param-matched) |
| r1_loop_l2x4_s95 | L2×4 reset | ~3.9M | **22.204±0.147** | **0.2877** | **+0.249 (t=14.2)** | +2.49 | naive loop — **L8 역전!** |
| r1_loop_l2x4_carry_s95 | L2×4 carry | ~3.9M | 21.751±0.145 | 0.3007 | | +2.03 | I1 |

### 라운드 1 결론 (2026-07-18, seed 95 단일 — seed 검증 필요)
- **RQ1 YES+**: naive loop가 L2 대비 +2.49 dB일 뿐 아니라 **비루프 L8을 +0.249 dB(t=14.2) 역전**
  (Déjà View의 attention 결과가 TTT에서도 성립; 파라미터 1/3.5).
- **Carry는 reset 대비 −0.45 dB.** "carry=inner epoch 추가"의 naive 직관 실패 = IDEAS.md의
  core challenge(NS-normalized update의 비수렴/궤도 병리 + self-referential drift) 실증.
  → r2의 rho/delta/lrs가 carry를 구제하는지가 방법론 스토리의 핵심 분기.
- L6→L8 비루프는 포화(21.97→21.955)인데 loop는 그 벽을 넘음 → "같은 파라미터를 더 깊게 재사용"이
  깊이 추가보다 나은 체제 존재.

## 라운드 2 — core-challenge fix 계열 (실행 중)

| exp | config | PSNR | LPIPS | vs reset-loop | 비고 |
|---|---|---|---|---|---|
| r2_loop_l2x4_carry_lrs_s95 | carry + I4 per-loop LR bias | (진행중) | | | carry 구제 시도 1 |
| r2_loop_l2x4_carry_rho_s95 | carry + I2 residual-gated LR | (진행중) | | | carry 구제 시도 2 |
| r2_loop_l2x4_carry_delta_s95 | carry + I3 delta writes | (진행중) | | | carry 구제 시도 3 |
| r2_loop_l2x4_inject_s95 | reset + input injection | (진행중) | | | champion(reset) 강화 시도 |
