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
| r3_loop_l2x4_delta_s95 | reset + delta | 22.260±0.148 | 0.2876 | s95 paired +0.056 (t=7.0) |
| r3_loop_l2x4_delta_s96 | reset + delta (s96) | 22.069±0.146 | 0.2888 | s96 paired −0.019 (t=−2.4). 교훈: within-seed paired t는 훈련-seed 분산을 못 잡음 |
| r3_loop_l2x4_delta_s97 | reset + delta (s97) | 22.111±0.147 | 0.2904 | s97 paired +0.050 (t=7.4) |

- **delta 3-seed 확정: paired Δ = +0.056/−0.019/+0.050 → 평균 +0.029dB** (3-seed 평균 22.147 vs reset 22.117).
  방향은 양성(2/3 seed 유의)이나 단독 효과는 소폭. carry에서는 +0.23으로 더 컸음(병리 fix로서의 역할).
| r3_loop_l1x8_s95 | 1블록×8loop (~2M) | 22.007±0.147 | 0.3016 | reset L2×4 대비 −0.20 — 극한 tying도 L8 s95급 |
| r3_loop_l4x2_s95 | 4블록×2loop (~7M) | 22.077±0.147 | 0.2835 | PSNR은 L2×4 열세, LPIPS는 우세 |

- loop 형태 (s95): L1×8 22.007 < L4×2 22.077 < **L2×4 22.204** — PSNR 스윗스팟 = L2×4.
- **LPIPS는 고유 층수를 따라감**: L1×8 0.3016 > L2×4 0.2877 > L4×2 0.2835 ≈ L8 0.2839.
  → 지각 품질 갭은 고유 파라미터 부족 문제일 가능성; I6 supervision이 loop로 이를 메꿀 수 있는지 r4에서 확인.

- rho/rho2/lrs 모두 무효, delta만 유효 → **carry 병리 = 중복 콘텐츠 재기록 (WHAT), 스텝 크기(HOW BIG) 아님.**

## Loop-count 스케일링 발견 (2026-07-19, big-swing phase의 근거)

| exp | compute | PSNR | LPIPS | 비고 |
|---|---|---|---|---|
| r3_loop_l2x6_delta_s95 | **1.5×** | **22.648±0.149** | **0.2725** | vs L2×4+delta **+0.388 (t=33.8)**; LPIPS가 L8(0.2839)을 대폭 역전 |

- **Loop 수가 유일한 dB급 레버임이 확정.** knob들(+0.03~0.08)과 대비됨.
- iso-compute 전략: target 토큰은 memory에 무영향(F1, 코드로 검증됨) → 초반 loop에서 target 제외
  (input-only pass = 0.66×)하면 같은 비용으로 loop를 더 살 수 있음 = **late-join**.
- L2×5-lj(target은 마지막 2 loop만) = input 깊이 10, 0.99× compute — wave 5 1순위.

## Wave 5 (big-swing, 진행 중)

| exp | 메커니즘 | compute | PSNR | LPIPS | vs naive(s95) | 판정 |
|---|---|---|---|---|---|---|
| r5_loop_l2x5_lj_delta_s95 | late-join (tgt 마지막 2 loop) | 0.99× | 20.540 | 0.3571 | **−1.664** | **대실패 — target 읽기 깊이가 핵심 변수** |
| r5_loop_l2x6_rh_delta_s95 | read-heavy (input 2, tgt 6 loop) | ~1.00× | 21.283 | 0.3219 | **−0.921** | **실패** — input 동결도 불가 |
| r6_loop_l2x6_delta_d224_s95 | width↔depth: d224로 6 loop | ~1.00× | 22.335 | 0.2879 | +0.132 (t=13.4) | 양성이나 부족 — 균일 width 절감이 비쌈 (d256×6 대비 −0.34) |
| r6_loop_l2x5_ti1_s95 | d256 유지, TTT inter 2→1로 5 loop | ~1.10× | 21.362 | 0.3207 | **−0.842** | **대실패 — fast-weight 용량은 binding** (state-size 중요성 재확인) |

### Wall-clock 실측 (훈련 효율의 진짜 비용)
- naive ×4: 4.55 it/s / **×5: 4.14 it/s (+10%)** / **×6: 3.91 it/s (+16%)** — LPIPS·고정비가 커서
  loop 추가의 wall-clock 비용은 FLOPs 비율보다 훨씬 작음.
- 실용 프레임: ×5+d+s = +0.39dB @ +10% 훈련시간; ×6 = +0.47dB @ +16%. **×6+sup(미실행) 예상 +0.53.**

### Wave 5 삼각측량 결론 (2026-07-19)
- target 깊이 축소 −1.66 / input 깊이 축소 −0.92 / **둘 다 깊게 +0.44 (1.5×)**
  → **loop 이득 = input·target joint co-refinement. 어느 쪽도 동결/캐시 불가.**
- 토큰 스케줄링·write 구조 개입 전멸 (chunk −1.27, sfm 대기). per-loop 조건화는 knob급(gates +0.04).
- iso-compute 유일 경로: **pass를 균일하게 싸게** = width↔depth 교환. L2×6@d224 ≈ 1.00× (d² linear + d³ NS).
| r5_loop_l2x4_gates_s95 | Déjà View gates | 1.00× | (진행중) | | | |
| r5_loop_l2x4_rfb_s95 | render feedback (+delta+sup) | 1.03× | 22.002 | 0.2938 | −0.202 | **실패** — 자기 base(delta+sup 22.307)보다 −0.30 |
| r5_loop_l2x4_gates_s95 | Déjà View gates | 1.00× | 22.244 | 0.2866 | +0.040 (t=7.6) | knob급 소폭 — Déjà View 스케일 전이 안 됨 |
| r5_loop_l2x4_chunk_s95 | 4-chunk 순차 delta write (muon→1) | ~1.00× | 20.939 | 0.3257 | **−1.265** | **대실패** — NVS엔 대청크가 옳음 (LaCT 선택 재확인) |
| r5_loop_l2x4_sfm_s95 | SfM (carry+delta+incremental) | 0.90× | 20.239 | 0.3520 | **−1.965** | **최악 실패 — carry 계열 0/6 사망** |
| r6_loop_l2x7_delta_d208_s95 | width↔depth: d208로 7 loop | ~1.00× | 22.269 | 0.2903 | +0.065 (t=6.8) | 균일 절감 프런티어는 좁을수록 하락 (d224×6보다 나쁨) |
| r5_loop_l2x6_s95 | L2×6 plain (delta 기여 분리 control) | 1.5× | **22.670** | **0.2711** | +0.466 | **delta는 6 loop에서 무기여** (delta판과 동일) — 레시피 미니멀화 가능 |
| r6_loop_l2x8_d192_s95 | width↔depth: d192로 8 loop | ~0.98× | 22.164 | 0.2954 | −0.040 | 깊은-좁은 끝은 음수 — **프런티어 = 역U자, 정점 d240×5/d224×6 (+0.13)** |
| r7_loop_l2x5_muon2_s95 | d256 유지, muon 5→2로 5 loop | ~1.15× | 21.865 | 0.2968 | **−0.339** | **NS 반복 절감은 해로움** — Muon orthogonalization 품질이 loop당 중요 |

### 희생-대상 패널 결론: loop를 사기 위해 무엇을 깎아도 손해
width(정점 +0.13) > NS steps(−0.34) > TTT용량(−0.84). **어떤 자원도 loop와 iso-compute 교환 시
정점이 +0.13.** → 자원 재배분으로는 +0.5 불가. TTT×loop 물리(wave 7)가 유일한 길.
| r6_loop_l2x5_delta_sup_s95 | 5 full loops + delta + sup | 1.25× | **22.596** | **0.2771** | **+0.392 (t=29.7)** | ×6(1.5×, 22.648)를 거의 따라잡음 — 곡선 5에서 대부분 포화 |
| r6_loop_l2x5_delta_sup_s96 | 〃 (s96) | 1.25× | 22.331 | 0.2807 | +0.243 (t=22.9) | |
| r6_loop_l2x5_delta_sup_s97 | 〃 (s97) | 1.25× | 22.537 | 0.2800 | +0.477 (t=34.9) | |

- **×5+delta+sup 3-seed 확정: paired ΔPSNR +0.392/+0.243/+0.477 → 평균 +0.371, LPIPS −0.010.**
  이게 현재 최고 확정 결과 (1.25× FLOPs / wall-clock +10%). "+0.5 @ iso-compute"엔 아직 미달.
| r6_loop_l2x5_delta_d240_s95 | width↔depth: d240으로 5 loop | ~0.97× | 22.338 | 0.2863 | +0.134 (t=12.6) | 균일-절감 프런티어 정점 (~+0.13에서 평탄) |
| r7_loop_l2x5_d240_gates_s95 | d240×5 + gates 스택 | ~0.97× | 22.366 | 0.2835 | +0.16 | gates가 d240×5(+0.13)에 미세 추가 |

- **Test-time loop 외삽 실패** (기존 ckpt eval): L2×6 모델→4/8 loop = 20.4/18.3; L2×4→5/6 = 20.7/19.1.
  고정-L 훈련은 다른 L로 추론 불가 (Ouro와 일치). test-time scaling엔 stochastic-L 훈련 필요 — 보류.

- late-join 실패의 함의: target 깊이 8→4에서 −1.66dB. **loop 이득의 상당 부분 = target-side 반복
  read-out.** input-side 절감(F1)은 유효하되 target쪽을 깎으면 안 됨 → read-heavy가 정확한 역실험.

## 라운드 4 (진행 중)

| exp | config | PSNR | LPIPS | paired Δ vs reset(s95) | 비고 |
|---|---|---|---|---|---|
| r4_loop_l2x4_sup_s95 | reset + I6 per-loop supervision | 22.287±0.149 | 0.2885 | **+0.083 (t=7.9)** | 두 번째 유효 개선 (loss-side) |
| r4_loop_l2x4_sup_s96 | 〃 (s96) | 22.130±0.147 | 0.2893 | +0.043 (t=5.0) | |
| r4_loop_l2x4_sup_s97 | 〃 (s97) | 22.154±0.148 | 0.2927 | +0.094 (t=11.8) | |

- **sup 3-seed 확정: 평균 +0.073, 3/3 seed 양성 — 첫 3-seed 통과 메커니즘.** (지금까지 확정 효과:
  sup +0.073 ✓, delta +0.029 (2/3), delta+sup +0.103 (s95 단일). 모두 소폭 → wave 5가 승부.)
| r4_loop_l2x4_delta_sup_s95 | delta + sup 결합 | 22.307±0.148 | 0.2879 | +0.103 | sub-additive (0.056+0.083→0.103); L2×4 iso-compute 최고 기록 |

### 라운드 2 결론
- **reset loop(22.204)가 챔피언 유지.** 시도한 4개 변형 모두 하회.
- carry 병리의 원인 규명: 스텝 크기(rho/lrs 무효)가 아니라 **중복 콘텐츠 재기록** — 타겟을
  innovation(v − f_w(k))으로 바꾼 delta만 +0.23dB 부분 구제. → "TTT loop에서 상태를 살리려면
  메모리가 이미 아는 것을 다시 쓰면 안 된다"는 mechanism-level 발견.
- Huginn식 input injection 무효 → TTT loop의 안정화 요구는 attention loop와 다름 (차별점 근거).
