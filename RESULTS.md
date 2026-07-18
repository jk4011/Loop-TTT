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

## Seed 검증 (헤드라인 3-seed)

| config | s95 | s96 | s97 | 평균 |
|---|---|---|---|---|
| loop L2×4 reset | 22.204 / 0.2877 | 22.088 / 0.2893 | 22.060 / 0.2915 | **22.117 / 0.2895** |
| L8 비루프 | 21.955 / 0.2839 | 22.165 / 0.2752 | 22.174 / 0.2770 | **22.098 / 0.2787** |

- seed-matched paired ΔPSNR(loop − L8): +0.249 / −0.078 / −0.113 → **3-seed 평균 +0.019 ≈ 0.**
- **최종 판정: naive loop는 PSNR에서 L8과 동급, LPIPS는 +0.011 약간 열세, 파라미터 1/3.5.**
  s95의 "+0.25dB 역전"은 seed 운 (L8 seed 분산 0.22dB — 단일 seed 판정 금지 재확인).
- 프레이밍: naive loop = parity. **novel method의 목표 = 동급을 확실한 우위로** (그리고 LPIPS 갭 해소).
  Δ 판정은 항상 3-seed 평균 + seed-matched paired로.

## 라운드 2 — core-challenge fix 계열 (실행 중)

| exp | config | PSNR | LPIPS | vs reset-loop | 비고 |
|---|---|---|---|---|---|
| r2_loop_l2x4_carry_lrs_s95 | carry + I4 per-loop LR bias | 21.759±0.146 | 0.2995 | −0.445 | **구제 실패** — plain carry와 동일 |
| r2_loop_l2x4_carry_rho_s95 | carry + I2 residual-gated LR | 21.727±0.146 | 0.3009 | −0.477 | **구제 실패** — plain carry와 동일 |
| r2_loop_l2x4_carry_delta_s95 | carry + I3 delta writes | 21.984±0.148 | 0.2955 | −0.220 | **부분 구제** (+0.23 vs plain carry; L8은 초과, reset엔 미달) |
| r2_loop_l2x4_inject_s95 | reset + input injection | 22.149±0.148 | 0.2958 | −0.054 (t=−3.6) | 도움 안 됨 — attention-loop 안정화 기법이 TTT에 전이 안 됨 |

### 메커니즘 프로브 (probe_loop_fit.py, 16 scenes, 학습된 s95 체크포인트)

loop별 TTT layer의 (cos_pre = update 전 fast weight가 타겟을 설명하는 정도,
cos_post = update 후, dw1_rel = 상대 스텝 크기):

- **carry**: cos_pre는 loop마다 상승(0.14→0.33)하지만 **cos_post는 하락**(0.70→0.62→0.53→0.41),
  **dw1_rel은 감소하지 않음**(0.16→0.30→0.44→0.44) → **constant-angle orbit 병리가 그대로 측정됨.**
  후기 loop의 update는 사실상 역효과.
- **reset**(챔피언): 매 loop cos_pre ≈ −0.2~0 (백지에서 시작), cos_post 0.3~0.56. loop 이득은
  메모리 누적이 아니라 **feature 정제**에서 옴. x_rms 안정(발산 없음).
- **carry+delta**: carry 대비 cos_pre 최고(0.29~0.38), cos_post 하락 완화 → 부분 구제와 일치.
- **구현 통찰 (rho 실패 원인)**: per-token lr은 NS orthogonalization의 **안쪽**에 곱해지고 NS가
  결과를 재정규화하므로 lr의 크기 정보는 지워짐 (방향 가중만 남음). → 스텝 크기를 misfit에
  반응시키려면 **NS 이후에** 스케일해야 함 = `rho2` (post-NS chunk-level scaling, r3에서 검증).

## 라운드 3 (진행 중)

| exp | config | PSNR | LPIPS | 비고 |
|---|---|---|---|---|
| r3_loop_l2x4_carry_rho2_s95 | carry + I2' post-NS scaling | 21.789±0.146 | 0.3007 | **구제 실패** (+0.04 vs carry, 노이즈) — 스텝-크기 가설 최종 기각 |
| r3_loop_l2x4_delta_s95 | reset + delta | **22.260±0.148** | **0.2876** | **신기록 — reset 대비 paired +0.056, t=6.98** (s96/s97 검증 중) |
| r3_loop_l1x8_s95 | 1블록×8loop (~2M) | 22.007±0.147 | 0.3016 | reset L2×4 대비 −0.20 — 극한 tying도 L8 s95급 |
| r3_loop_l4x2_s95 | 4블록×2loop (~7M) | 22.077±0.147 | 0.2835 | PSNR은 L2×4 열세, LPIPS는 우세 |

- loop 형태 (s95): L1×8 22.007 < L4×2 22.077 < **L2×4 22.204** — PSNR 스윗스팟 = L2×4.
- **LPIPS는 고유 층수를 따라감**: L1×8 0.3016 > L2×4 0.2877 > L4×2 0.2835 ≈ L8 0.2839.
  → 지각 품질 갭은 고유 파라미터 부족 문제일 가능성; I6 supervision이 loop로 이를 메꿀 수 있는지 r4에서 확인.

- rho/rho2/lrs 모두 무효, delta만 유효 → **carry 병리 = 중복 콘텐츠 재기록 (WHAT), 스텝 크기(HOW BIG) 아님.**

### 라운드 2 결론
- **reset loop(22.204)가 챔피언 유지.** 시도한 4개 변형 모두 하회.
- carry 병리의 원인 규명: 스텝 크기(rho/lrs 무효)가 아니라 **중복 콘텐츠 재기록** — 타겟을
  innovation(v − f_w(k))으로 바꾼 delta만 +0.23dB 부분 구제. → "TTT loop에서 상태를 살리려면
  메모리가 이미 아는 것을 다시 쓰면 안 된다"는 mechanism-level 발견.
- Huginn식 input injection 무효 → TTT loop의 안정화 요구는 attention loop와 다름 (차별점 근거).
