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
| r5_loop_l2x6_s95 | L2×6 plain (delta 기여 분리 control) | 1.5× | 22.670 | 0.2711 | +0.466 | delta는 6 loop에서 무기여 (delta판과 동일) — 레시피 미니멀화 가능 |
| r7_loop_l2x6_sup_s95 | **L2×6 + sup** | 1.5× | **22.781** | **0.2718** | **+0.578 (t=46)** | 절대 최고치 (sup이 ×6에 +0.11 추가). 단 1.5× compute |
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

## Wave 7 — TTT×loop 고유 물리 (PI: 자원 재배분 금지, iso-compute)

| exp | 메커니즘 | it/s (vs naive 4.55) | PSNR | LPIPS | vs naive(s95) | 판정 |
|---|---|---|---|---|---|---|
| r7_loop_l2x4_boost_s95 | **boosting (잔차 특화, 유효용량 ×n)** | **4.60 (동일!)** | **22.303** | 0.2869 | **+0.099** | **진짜 iso-compute 최고 메커니즘** — 죽었던 carry 계열 부활 |
| r7_loop_l2x4_ep2_s95 | inner epochs=2 (underfit 공격) | 4.26 (−6%) | 22.281 | 0.2830 | +0.077 (t=6.3) | LPIPS −0.005 개선 (boost보다 큼) — underfit 실재 |
| r8_loop_l2x4_ep3_s95 | inner epochs=3 | 3.96 | 22.197 | 0.2834 | −0.007 | ep2보다 나쁨 — **fit축 sweet spot=ep2** (3스텝은 포화/overfit) |
| r7_loop_l2x4_boost_sup_s95 | boost + sup | | (진행중) | | | |
| r7_loop_l2x4_ep2_sup_s95 | **ep2 + sup (직교축)** | ~4.2 | **22.358** | **0.2829** | **+0.154** | **거의 완전 가산!** (0.077+0.083=0.160) — orthogonal 스택이 핵심 |
| r8_loop_l2x4_boost_ep2_sup_s95 | boost+ep2+sup (총동원) | ~4.2 | 22.349 | 0.2845 | +0.145 | ep2+sup(+0.154)보다 나쁨 — **boost는 ep2/sup과 같은 계열, 직교 아님** |
| r9_loop_l2x4_rr_s95 | read-side refinement (apply축) | ~4.2 | (진행중) | | | 미탐색 축 — 직교면 스택 가치 |
| r7_loop_l2x4_boost_sup_s95 | boost + sup | ~4.4 | 22.321 | 0.2880 | +0.117 | sup(+0.083) 위 boost +0.034 — 강한 sub-additive (같은 축) |
| r8_loop_l2x4_boost_ep2_s95 | boost + ep2 (용량×fit) | ~4.3 | 22.333 | 0.2840 | +0.130 | 부분 겹침 (둘 다 메모리-fit) |
| r9_loop_l2x4_ep2_pli_sup_s95 | ep2+pli+sup (3직교축) | | (진행중) | | | fit×메모리파라미터×loss |
| r8_loop_l2x4_pli_s95 | per-loop 학습 init (7.0M) | 4.56 (동일) | 22.121 | 0.2889 | **−0.083** | **해로움 — 공유 init가 더 낫다** (meta-learning/정규화 이득). "타협" 가설 기각 |

- **boost가 핵심 발견**: fast weight는 활성값이므로 loop가 시간축으로 메모리 인스턴스를 여러 개
  만들고, 각자 이전 loop의 잔차만 담당 → 동일 arch·동일 속도로 +0.10. delta/gates(+0.03~0.04)의
  2~3배이며 **속도 페널티 0**. attention loop엔 없는 성질을 직접 활용.

## ★ 확정 최고 iso-compute 결과 (2026-07-20)

**ep2+sup+gates 3-seed (seed-matched paired vs naive loop):**
| seed | PSNR | LPIPS | paired ΔPSNR | t |
|---|---|---|---|---|
| 95 | 22.390 | 0.2809 | +0.187 | 14.7 |
| 96 | 22.360 | 0.2840 | +0.272 | 26.3 |
| 97 | 22.450 | 0.2825 | +0.390 | 36.0 |
| **평균** | **22.400** | **0.2825** | **+0.283** | — |

→ **naive loop 대비 +0.283 dB (3-seed 확정), LPIPS −0.007, 파라미터 동일, wall-clock +6%.**
목표 +0.5의 57%. geometry가 직교 positive면 +0.5 도달 가능 → 전체 스택(geo+ep2+gates+sup) 실행 중.

| r12_loop_l2x4_boost_gates_sup_s95 | boost+gates+sup (대안 스택) | 22.364 | 0.2871 | +0.160 | ep2+sup+gates(+0.187) 하회 — ep2 스택이 최고 |

## Wave R2 — 라운드-2 task-agnostic 아이디어 (IDEAS_R2.md)

| exp | 메커니즘 | 근거 | PSNR | LPIPS | vs naive(s95) | 판정 |
|---|---|---|---|---|---|---|
| r13_loop_l2x4_attngate_s95 | Attn-OutGate (LT2 SDPA) | agent6, 최강증거 | (진행중) | | | attn-branch 축 |
| r13_loop_l2x4_qkvroute_s95 | QKV-Route (per-loop LoRA) | agent1 | (진행중) | | | transform 축 (이론상 중립위험) |
| r13_loop_l2x4_rotbag_s95 | RotBag (per-loop 회전) | agent3 | (진행중) | | | diversity 축 (이론 test) |
| r13_loop_l2x4_nlcond_s95 | NL-Cond (SwiGLU preact bias) | agent9 이론#1 | (진행중) | | | nonlinear-cond 축 |

## Wave 10+ — 100-아이디어 flagship (iso-compute, IDEAS_100.md / experiment_queue TIER1-3)

| exp | 메커니즘 | 근거 | PSNR | LPIPS | vs naive(s95) | 판정 |
|---|---|---|---|---|---|---|
| r10_loop_l2x4_mom_s95 | cross-loop momentum (heavy-ball) | 창의1 | 22.021 | 0.2929 | **−0.183** | 실패 — momentum 축 사망 |
| r11_loop_l2x4_pw1_s95 | precond_w1 (Gauss-Newton/RLS) | agents 1,10,4 | 22.179 | 0.2894 | **−0.025** | 중립 — c=‖H‖²가 느슨해 preconditioner≈항등이었을 가능성; 또는 fit이 병목 아님 |
| r11_loop_l2x4_cumboost_s95 | cumulative-residual boost | agents 9,5,2,6 | 20.686 | 0.3675 | **−1.518** | **참사** — 잔차 r이 stale(v-space가 loop마다 drift). boost가 현재 key로 재평가하는 게 우월 |
| r11_loop_l2x4_kcenter_s95 | key DC-decorrelation | agent 5 | 22.221 | 0.2874 | +0.017 | 사실상 중립 (noise) |
| r11_loop_l2x4_ltemp_s95 | per-loop q/k temperature | agents 5,8 | 22.193 | 0.2875 | −0.011 | 중립 (8번째) |
| r12_loop_l2x4_geo_s95 | **Plücker/epipolar addressing** | agent 3 | (진행중) | | | **geometry 축** — 유일한 미시도 고-천장 (이전 프로젝트 camera +1.7dB) |
| r11_loop_l2x4_fmom_s95 | feature Anderson/Nesterov | agents 4,6,7 | 19.719 | 0.3860 | **−2.485** | **붕괴(L2와 동일해로 수렴)** — 외삽이 loop 기여 무효화 |
| r11_loop_l2x4_c2fmuon_s95 | spectral coarse-to-fine NS steps | agents 8,10 | 22.059 | 0.2928 | **−0.145** | 실패 — 초기 NS 스텝 축소 해로움 (muon2와 일치) |

**Wave 10 중간 판정: 정교한 flagship 6연속 실패/중립** (momentum −0.18, feat_mom −2.49, pw1 −0.03,
cumboost −1.52, epavg −0.01, c2f_muon −0.15). → **확정 단순 축(ep2/sup/gates/boost)이 매우 강건.**
확정 최고 iso 스택 = **ep2+sup+gates: s95 +0.187, s96 +0.156** (3-seed 진행). 남은: pw1diag, key_center, ltemp.

**교훈: momentum·feat_mom 둘 다 실패 → loop는 가속/외삽이 아니라 실제 반복을 원한다.**
| r11_loop_l2x4_epavg_s95 | Polyak iterate averaging | agents 1,2,7 | 22.192 | 0.2871 | **−0.012** | 중립 — ep3 궤도 구제 실패 |
| r11_loop_l2x4_pw1diag_s95 | precond_w1 diagonal (결정 재시험) | agents 1,4 | 22.214 | 0.2868 | +0.010 | **중립 확정 → w1 방향 conditioning은 PSNR 무관. "exact inner optimization" 계열 전멸** |

## Wave 9+ — 직교축 가산 스택 & 창의 메커니즘 (iso-compute)

| exp | 스택/메커니즘 | PSNR | LPIPS | vs naive(s95) | 판정 |
|---|---|---|---|---|---|
| r7_loop_l2x4_ep2_sup_s95 | ep2+sup (fit×loss) | 22.358 | 0.2829 | +0.154 | 직교 가산 |
| r9_loop_l2x4_ep2_gates_sup_s95 | **ep2+sup+gates (fit×loss×조건화)** | **22.390** | **0.2809** | **+0.187 (t=14.7)** | **현재 iso 최고** — gates +0.033 추가(직교) |
| r10_loop_l2x4_mom_s95 | cross-loop momentum (창의 flagship) | (진행중) | | | inner-optimizer 축 — 궤도→수렴 |
| r9_loop_l2x4_rr_s95 | read-side refinement (apply축) | 22.197 | 0.2871 | −0.006 | **중립 — apply 축 무효** (고정 0.15) |

- **직교축 가산 법칙 재확인**: ep2(fit)+sup(loss)+gates(조건화) 각 축이 거의 완전 가산 → +0.187.
  남은 미탐색 직교축: inner-optimizer(momentum), apply(read_refine). 각각 positive면 계속 누적.

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

## Round-2 결과 (task-agnostic, 대부분 실패)
| exp | 메커니즘 | vs naive(s95) | 판정 |
|---|---|---|---|
| attngate | LT2 SDPA gate | -0.025 | 중립 (최강증거인데도) |
| qkvroute | per-loop LoRA (transform) | -0.097 | 실패 (이론: 입력측 선형=중립, 확인) |
| rotbag | per-loop 회전 (diversity) | -0.223 | 실패 (이론: 입력측 회전 재흡수, 확인) |
| nlcond | SwiGLU preact bias (이론#1) | -0.059 | 실패 (이론-정합도 안 됨) |
| distill | deep-teacher (teacher 8loop) | **-0.719** | 실패 - teacher가 깊은 unroll에서 붕괴(bad target) |
| stochdepth | loop수 {2..6} 샘플 훈련 | **-0.376** | 실패 - 다깊이 공유훈련이 각 깊이 희석; 깊은 unroll도 안 좋아짐 |

### ★ 중대 부정 발견: "more-loops +0.44~0.58"은 추론비용에 묶여 있어 iso-inference로 못 가져온다.
distill(bad teacher)/stochdepth(희석) 둘 다 사망. 깊이별 dedicated 모델에서만 이득. 확정 최고 iso = ep2+sup+gates +0.283.
실행중(마지막 미시도 직교축): streamnorm, fusedreadout, droploop.

## ★★ Cross-model KD 돌파구 (2026-07-20) — more-loops 이득을 iso-inference로 가져온 첫 성공
| exp | teacher | PSNR | LPIPS | vs naive(s95) | 판정 |
|---|---|---|---|---|---|
| r15_loop_l2x4_kd6sup_s95 | L2×6+sup (22.781) | 22.347 | 0.2858 | **+0.143** | KD 단독 성공 (self-distill 실패와 대조) |

- self-distillation/stoch-depth는 teacher가 student 자기 weight를 깊게 돌려 붕괴 → 실패.
  **cross-model KD**는 별도 잘 훈련된 deep teacher라 성공. teacher 22.781의 +0.58 중 student가 +0.143 흡수(25%).
- 추론 엄격 iso(student 4loop), teacher는 train-only frozen. **새 직교축(teacher-matching).**
- 강화 경로: (1) ep2+sup+gates+KD 스택(실행중), (2) 더 깊은/강한 teacher(L2×10+sup), (3) feature/LPIPS-space KD.

## KD 스택 & 강화 (2026-07-20)
| exp | 스택 | PSNR | vs naive(s95) | 비고 |
|---|---|---|---|---|
| r16_loop_ep2gs_kd6_s95 | ep2+sup+gates+KD | 22.501 | **+0.297** | KD가 확정스택에 +0.11 추가 (거의 직교) |

- KD 단독 +0.143, ep2+sup+gates 단독 +0.187(s95)/+0.283(3-seed) → 스택 +0.297(s95). **~직교 가산.**
- 3-seed 예상 ep2+sup+gates+KD ≈ +0.39. **현재 iso-inference 최고 경로.**
- 강화: 더 깊은 teacher(L2×8+sup 훈련중), feature/LPIPS-space KD, teacher 앙상블.
- 참고: KD는 2단계 파이프라인(deep teacher 훈련 → shallow distill). 배포 모델은 엄격 iso-inference.
  student 훈련은 teacher forward(no_grad)로 ~1.25× train. teacher 훈련은 일회성 asset.

### Wave 15 마무리 — 마지막 직교축 후보 3종 (2026-07-20, s95 paired vs naive 22.204)

| exp | PSNR | dPSNR | t | 판정 |
|---|---|---|---|---|
| r15_streamnorm (per-loop RMS renorm) | 22.183 | −0.021 | −4.4 | null |
| r15_fusedreadout (per-loop feat softmax 융합) | 22.239 | +0.035 | +4.5 | null(<0.15) |
| r15_droploop (train-time loop dropout p=0.1) | 22.115 | −0.089 | −9.8 | 음성 |

축 조사 종결: readout/norm/regularization 축 모두 null. 확정 스택은 ep2+sup+gates(+0.283)
+ KD(+0.11, s95)가 전부. → 라운드3 브레인스토밍(근본 메커니즘)으로 전환.

### L2×8+sup teacher (2026-07-20, s95 paired vs naive 22.204)

| exp | PSNR | LPIPS | dPSNR | t | 비고 |
|---|---|---|---|---|---|
| r16_loop_l2x8_sup (2× inference) | 23.000 | 0.2630 | **+0.797** | 51.0 | KD teacher 자산 |

Loop 스케일링 법칙 확장: ×4 22.204 → ×5 +0.39 → ×6 +0.578 → ×8 +0.797 (∝ ~log n_loops).
teacher가 강해질수록 KD로 가져올 원천 gain 증가. r17에서 이 teacher로 KD(품질 스케일링) 시험.

### KD 스택 3-seed 확정 (ep2+sup+gates + KD from L2×6+sup, 2026-07-20)

| seed | PSNR | dPSNR (paired vs naive 동seed) | t |
|---|---|---|---|
| s95 | 22.501 | +0.297 | 23.8 |
| s96 | 22.456 | +0.369 | 37.9 |
| s97 | 22.504 | +0.444 | 42.4 |
| **평균** | **22.487** | **+0.370** | — |

ep2+sup+gates 단독(+0.283 3-seed)에서 **KD가 +0.087 추가** (거의 직교, 3-seed 확정).
현재 최고 확정 스택 = **+0.370dB @ iso-inference**. 목표 +1.0까지 +0.63 남음.
다음: 더 강한 teacher(L2×8, +0.797), trajectory KD, warm-start, loop-anneal로 KD 흡수량 증대.

### Wave 17 초기 결과 (2026-07-20, s95)

| exp | PSNR | vs naive 22.204 | vs kd6-stack 22.501 | 판정 |
|---|---|---|---|---|
| r17_ep2gs_kdtraj (waypoint KD w=0.3) | 22.486 | +0.283 | −0.015 | **무효** (endpoint KD보다 개선 없음) |

Trajectory/waypoint KD 사망: student 4-loop 경로 ≠ teacher 6-loop 경로, 과잉 제약.
교훈: KD는 endpoint만으로 충분; 중간 waypoint 매칭은 도움 안 됨.
남은 KD 레버는 **teacher 품질**(kd8 진행 중)과 **weight-space warm-start**(warm6_kd 진행 중).

### EMA (outer-weight, decay 0.9995) — kdtraj run, s95
non-EMA 22.486 → EMA 22.483 (−0.004) = **무효**. Cosine-to-zero lr 꼬리가 이미 iterate 평균화.
EMA 축 접음 (agent-7이 예측한 null).

### 궤적 프로브 — +0.578의 정체 규명 (2026-07-20, 128 scenes, per-loop render)

| model | loop별 PSNR | 최종 LPIPS |
|---|---|---|
| ×4 sup | 18.44 / 21.15 / 22.08 / **22.13** | 0.287 |
| ×6 sup | 16.84 / 19.81 / 21.49 / 22.23 / 22.59 / **22.62** | 0.271 |
| ×8 sup | 15.23 / 18.12 / 20.14 / 21.41 / 22.07 / 22.50 / 22.85 / **22.85** | 0.263 |

**결론 (H1 궤적 재배속 확정):**
- ×6의 loop-4는 PSNR 22.23(≈×4 최종 22.13)이지만 LPIPS 0.420 vs 0.287 — **훨씬 덜 수렴**.
  ×6은 지각 정제를 loop 5-6로 미뤄 궤적을 늘림 → "더 나은 loop-4 함수"가 아니라 **늘어난 궤적**.
- 모든 모델 매 loop 단조 개선(orbit 없음), 진짜 반복. depth gain이 **불균형하게 지각적(LPIPS)**:
  0.287→0.271→0.263. PSNR 포화 후에도 LPIPS 계속 하락 = 후반 loop = 지각 디테일 합성기.

**라우팅 함의:**
1. kdtraj 무효 설명됨: student는 자기 4단계에 배속해야 함 (teacher 6단계 waypoint 강요 X).
2. **LPIPS-space KD 우선** (진행: r17_ep2gs_kd6_lpips): MSE-KD는 PSNR만, 지각 격차 놓침.
3. iso-FLOP로 loop 수 늘리기(micropass 등)가 원리적 레버 — 한계 loop이 진짜 가치 추가 확인.

### kd8 (L2×8 teacher KD) — teacher 품질 스케일링 실패 (2026-07-20, s95)

| exp | PSNR | LPIPS | vs naive | vs kd6-stack 22.501 | 판정 |
|---|---|---|---|---|---|
| r17_ep2gs_kd8 (L2×8 teacher) | 22.212 | 0.2995 | +0.008 | −0.289 | **급락/무효** |

**중대 발견**: 더 강한 teacher(L2×8 +0.797)가 KD를 오히려 **망침** (LPIPS도 악화 +0.012).
TAKD "teacher-student 격차 과대 → 전이 악화" 확증. 프로브가 보인 대로 L2×8의 22.85 endpoint는
8단계 궤적 산물 → 4단계 student에 MSE 강요 = 불가능한 압축 = 손상.
**KD sweet spot = 적당히 깊은 teacher (L2×6, gap 2). teacher 품질은 단조 증가 안 함.**
남은 KD 레버: LPIPS-KD(진행), warm-start(진행), progressive 8→6→4 (중간 단계가 gap 2라 유효 가능).

### Wave 17 마무리 + Wave 18 시작 (2026-07-20)

| exp | PSNR | dPSNR vs naive 22.204 | 판정 |
|---|---|---|---|
| r17_anneal654 (loop 6→5→4 커리큘럼) | 22.140 | −0.064 | 실패 (loop-count 스케줄 계열 재확인) |
| r17_ep2gs_kd8 (L2×8 teacher KD) | 22.212 | +0.008 | 실패 (teacher-too-strong/TAKD) |
| r17_ep2gs_kdtraj (waypoint KD) | 22.486 | +0.283 | 무효 (endpoint KD 대비 −0.015) |

KD 방향 상한 확인(+0.11), loop-count 커리큘럼 실패. → **Wave 18: TTT×loop 고유(boost 계열)로 전환.**
slice-boost it/s 9.3 (naive 4.55보다 빠름 — NS가 d_h/4 slice에 → update 비용↓, iso-compute 충족).

### Wave 18 1차 (boost 변형 4종, 2026-07-20, s95 paired vs naive 22.204 / boost 22.303)

| exp | PSNR | vs naive | vs boost | 판정 |
|---|---|---|---|---|
| r18_slice_boost (partitioned single-step writes) | 20.930 | **−1.273** | −1.37 | **참사** |
| r18_proj_boost (방향 deflation) | 22.061 | −0.143 | −0.24 | 실패 |
| r18_detach_boost (stagewise detach) | 22.236 | +0.032 | −0.067 | boost 하회 |
| r18_ensread (read측 앙상블) | 22.287 | +0.083 | −0.016 | boost 동급(무효) |

**교훈 (design rule 업데이트):**
1. **파티션은 자원 기아**: slice-boost의 pass당 write 폭 d_h/4=128이 치명적 (용량 halve −0.84,
   quarter는 −1.27과 유사 규모). chunk(토큰 분할 −1.27)와 동형 실패 — **binding 자원(용량)을
   pass 간 분할하면 안 됨**. orbit 회피보다 full-width write가 우선.
2. **boost의 magnitude는 유효 정보**: 방향만 빼는 proj-boost가 full 빼기보다 나쁨(−0.24).
   "방향만 살아남는다"는 write-gradient 얘기; target-space 잔차는 magnitude도 유의미.
3. **boost chain의 gradient는 필수**: detach하면 +0.099→+0.032. 이전 메모리가 "다음 잔차를
   쉽게 만들도록" 공동 훈련되는 것이 boost 이득의 상당 부분 (joint refinement, pli 교훈 재확인).
4. **read측 앙상블은 잉여**: ensread ≈ boost. residual stream이 이미 이전 메모리 기여를 운반.
→ plain boost(fresh full-width + 잔차 target + live gradient)가 국소 최적점으로 강건.
남은 미시도: multiboost(모든 이전 합의 잔차), boost-gain(state feedback), decorloss(train-only).

### Wave 18 2차 + boost seed 검증 (2026-07-20)

| exp | PSNR | vs naive(동seed) | 판정 |
|---|---|---|---|
| r18_multiboost (모든 이전 합 빼기) | 22.269 | +0.065 | boost 하회 |
| r18_boostgain (misfit-gated 빼기) | 22.211 | +0.007 | 무효 |
| r18_boost_s96 | 22.112 | +0.024 (t=2.8) | ⚠️ |
| r18_boost_s97 | 22.088 | +0.028 (t=4.6) | ⚠️ |

**중대: boost 3-seed 확정 = +0.050** (s95 +0.099 / s96 +0.024 / s97 +0.028).
s95는 seed 운. 방향은 3-seed 일관 양수(실재)나 크기는 절반. boost 계열 6개 변형 전부 boost 하회
→ plain boost가 국소 최적이며 그 최적값이 +0.05.
TTT×loop write/read/partition/feedback 축 사실상 소진. 남은 미시도: transductive-write(읽을 분포에
쓰기), misfit-echo(잔차→stream 주입), decorloss(train-only). + probe-memory-overlap(다양성 headroom 판정).

### Wave 18 3차 (2026-07-21)

| exp | PSNR | vs naive(동seed) | 판정 |
|---|---|---|---|
| r18_transwrite (target 토큰에도 write) | 22.005 | −0.199 | 실패 (self-echo) |
| r18_echo (잔차 stream 주입) | 22.089 | −0.115 | 실패 |
| r18_bgs_s96 (boost+gates+sup) | 22.187 | +0.099 | ✓ |
| r18_bgs_s97 | 22.249 | +0.189 | ✓ |

**boost+gates+sup 3-seed 확정 = +0.149** (s95 +0.160 / s96 +0.099 / s97 +0.189, t≥11).
ep2/KD 제외 제약 하의 확정 최고 스택. TTT×loop 고유 성분(boost)의 축 기여 +0.05,
generic 성분(gates/sup) +0.10.

**TTT×loop 고유 축 총결산 (라운드 2+4, ~300 아이디어, 30+ 런):**
생존 = boost(+0.050 3-seed) 단 하나. 변형 6종(slice/proj/detach/ensread/multi/gain) 전멸,
신물리 2종(transwrite/echo) 전멸, carry/momentum/partition/addressing/schedule 전멸.
boost의 필수 성분 완전 규명: fresh full-width + full 잔차 + live gradient (셋 다 건드리면 악화).

### 프로브: memory-overlap (boost ckpt, 16 scenes, 2026-07-21) — wave-18 평탄함의 정량 규명

- **pairwise cos(f_{W_i}(k), f_{W_j}(k)) = 0.52~0.71**: boost의 4개 메모리는 function-space에서
  **높게 중복** (상보적이지 않음). "이미 직교라 headroom 없음" 가설 기각.
- **잔차 사슬 ‖v−Σf_{W_ℓ}(k)‖/‖v‖ = 1.000 → 0.977**: 메모리 앙상블이 v의 **~2.3%만** 설명.
  fast-weight readout은 v의 재구성이 아니라 **작은 방향 신호**(o_norm으로 재스케일되는 message).

**핵심 통찰**: TTT 메모리는 이 스케일(d256, 1 NS step, weight-norm)에서 연상기억이 아니라
소규모 cross-view 라우팅 신호로 동작. write-target 공학(boost 계열 전부)은 **~2% 크기의 레버**를
조작한 것 → wave-18 전 변형이 ±0.05 안에 몰린 이유의 정량적 설명.
→ TTT×loop write측 개선의 구조적 상한 ≈ ±0.1dB. (논문 발견: "왜 looped-TTT 이득이 캡되는가")

### Wave 19 — optzone이 film을 부활시킴: 허용 스택 돌파 (2026-07-21, s95)

| exp | PSNR | LPIPS | vs naive | 한계 기여 |
|---|---|---|---|---|
| r19_gates_sup (attribution) | 22.310 | 0.2873 | +0.106 | — |
| (r12) bgs = boost+gates+sup | 22.364 | 0.2871 | +0.160 | boost +0.054 |
| r19_bgs_optzone | 22.412 | 0.2834 | +0.208 | optzone +0.048 |
| **r19_bgs_film_optzone** | **22.602** | **0.2770** | **+0.399** | **film(+optzone) +0.191** |

**발견**: loop_film은 원래 무효로 측정됐으나 그 원인이 **weight-decay가 per-loop 파라미터를
identity로 끌어내리는 결함**이었음(agent-7 발견). optzone(wd=0, lr×8)으로 수정하자 film이
+0.19를 냄. LPIPS도 0.2770 (naive 0.2877 대비 −0.0107, 역대 최고).
ep2/KD 없이 허용 성분만으로 +0.399 (s95) — 이전 허용 최고(+0.160)의 2.5배.
검증 중: bgsfo 3-seed(s96/s97), gfso(boost 제외 attribution), +nl_cond(같은 억압 수정 후보).

### Wave 19 검증 완료 (2026-07-21)

| exp | PSNR | dPSNR (동seed paired) | 판정 |
|---|---|---|---|
| r19_bgsfo_s96 | 22.391 | +0.303 | ✓ |
| r19_bgsfo_s97 | 22.480 | +0.420 | ✓ |
| **bgsfo 3-seed** | — | **+0.374** (0.399/0.303/0.420, t≈28-30) | **확정** |
| r19_gfso (boost 제외) | 22.539 | +0.335 | boost 기여 +0.064 유지 |
| r19_bgsfon_nlcond | 22.632 | **+0.428** (s95 최고) | nl_cond 부활 +0.029 |

**허용 스택(ep2/KD 제외) 확정: +0.374 (3-seed)** — 이전 +0.149의 2.5배. LPIPS −0.011 일관.
스택 구성 = boost(TTT×loop novel, +0.06) + gates+film(조건화, +0.25 with optzone) + sup + optzone.
다음: nlcond 3-seed(s96/97), qkv_route 부활(마지막 억압 knob), lr_mult 16 스윕. 목표 +1.0까지 +0.63.

### Wave 19b (2026-07-21)

| exp | PSNR | dPSNR | LPIPS | 판정 |
|---|---|---|---|---|
| r19_bgsfon_s96 | 22.482 | +0.394 | 0.2758 | ✓ |
| r19_bgsfon_s97 | 22.454 | +0.394 | 0.2802 | ✓ |
| **bgsfon 3-seed** | — | **+0.405** (0.428/0.394/0.394) | — | **확정** |
| r19_bgsfon_qkvr | 22.638 | +0.434 | **0.2743** | PSNR 무효, LPIPS 최고 |
| **r19_bgsfon_lr16** | **22.724** | **+0.520** | **0.2702** | **s95 신기록** |

lr_mult 8→16이 +0.09 추가 — 조건화 파라미터는 더 큰 자유를 원함 (아직 상승 중).
확정 스택: boost+gates+film+nlcond+sup+optzone(×16) = 조건화 축 해방으로 +0.4~0.5대 진입.
검증 중: lr16 3-seed, lr32 스윕, qkvr+lr16.

### Wave 19c (2026-07-21) — lr 스윕 계속 상승

| exp | PSNR | dPSNR | LPIPS | 판정 |
|---|---|---|---|---|
| r19_bgsfon_lr16_s96 | 22.642 | +0.554 | 0.2697 | ✓ |
| r19_bgsfon_lr16_s97 | 22.597 | +0.537 | 0.2738 | ✓ |
| **bgsfon+lr16 3-seed** | — | **+0.537** (0.520/0.554/0.537, t≥35) | — | **확정** |
| **r19_bgsfon_lr32_s95** | **22.870** | **+0.666** | **0.2649** | **s95 신기록, 스윕 상승 중** |
| r19_bgsfon_qkvr_lr16 | 22.729 | +0.525 | 0.2697 | qkvr 무효 확정(탈락) |

**3-seed 확정 +0.537** — naive loop의 seed 분산(±0.1)을 감안해도 압도적(t 35+).
lr_mult 8→16→32 = +0.40→+0.52→+0.67: 조건화 다이얼은 여전히 "더 돌아가고 싶은" 상태.
검증 중: lr32 3-seed, lr64, no-boost@lr32 attribution.

### Wave 19d (2026-07-21) — +1.0 사정권

| exp | PSNR | dPSNR | LPIPS | 판정 |
|---|---|---|---|---|
| r19_bgsfon_lr32_s96 | 22.708 | +0.620 | 0.2670 | ✓ |
| r19_bgsfon_lr32_s97 | 22.779 | +0.719 | 0.2679 | ✓ |
| **bgsfon+lr32 3-seed** | — | **+0.668** (0.666/0.620/0.719) | — | **확정** |
| **r19_bgsfon_lr64_s95** | **23.059** | **+0.855** | **0.2587** | 스윕 계속 상승 |
| r19_gfson_lr32 (no boost) | 22.877 | +0.673 | 0.2666 | **boost 기여 ≈ 0** @ 강조건화 |

**핵심**: (1) 3-seed 확정 +0.668 — 원래 목표 +0.5 돌파. (2) lr64 s95 +0.855 — 정점 미도달.
(3) 강한 조건화 하에서 boost 한계기여 소멸(+0.673 vs +0.666) — 조건화가 boost의 역할
(pass별 다른 일)을 흡수. 방법론 미니멀화 가능(film-only ablation 진행 중).
진행: lr128 스윕, lr64 s96, film-only(sup 유/무).

### Wave 19e/20 (2026-07-21) — 스윕 정점 + 미니멀 ablation

| exp | dPSNR (s95) | LPIPS | 해석 |
|---|---|---|---|
| r20_film (film만, sup無) | +0.277 | 0.2741 | 미니멀 코어: 다이얼 1종 = +0.28 |
| r20_film_sup | +0.448 | 0.2741 | +sup +0.17 |
| r19_bgsfon_lr128 | **+0.879** | **0.2550** | 64→128 +0.024 — 정점 근처 |
| r19_bgsfon_lr64_s96 | +0.685 | 0.2628 | ✓ |

미니멀 레시피 수렴: **loop + per-loop 조건화(film+gates+nlcond) + sup + optzone(lr≈64-128)**.
boost 불필요(기여 0), qkvr 불필요. film 단독은 1/3 — 해방 상태에선 gates/nlcond도 실질 기여.
검증 중: lr64 s97, lr128 s96/s97 (정점 3-seed), gfson@lr64 (boost-free 정점).

### Wave 19f (2026-07-21) — +1.0 첫 돌파 (sup 시대 최종 수치)

| exp | PSNR | dPSNR | LPIPS |
|---|---|---|---|
| r19_bgsfon_lr64_s97 | 22.990 | +0.930 | 0.2615 |
| **r19_bgsfon_lr128_s97** | **23.113** | **+1.053** | **0.2549** |
| r19_bgsfon_lr128_s96 | 22.857 | +0.769 | 0.2611 |
| r20_gfson_lr64_s95 (boost-free) | 23.057 | +0.853 | 0.2595 |

**3-seed: lr64 = +0.823 / lr128 = +0.900** (전 seed t≥40, LPIPS −0.028~−0.037).
단일 seed 최고 +1.053 — **+1.0 목표 첫 돌파** (구성: gates+film+nlcond+sup+optzone lr128; boost는
lr64에서 기여 0 확인 — boost-free와 동급). 주의: 이 수치들은 sup 포함(제외 결정 전 투입분).
현재 진행: wave-21 no-sup 미니멀 4종 (사용자 확정 제약 하의 헤드라인 결정용).

### Wave 21 (no-sup 미니멀, 2026-07-21, s95)

| exp | PSNR | dPSNR | LPIPS | 판정 |
|---|---|---|---|---|
| r21_film_lr64 (film만) | 22.543 | +0.339 | 0.2712 | 미니멀 코어 |
| **r21_gfn_lr64 (gates+film+nlcond)** | **22.773** | **+0.569** | **0.2648** | **no-sup 승자** |
| r21_gfna_lr64 (+affine shift) | 22.732 | +0.528 | 0.2697 | shift 무효 (no-sup 체제) |
| r21_gfna_lr128 | 22.745 | +0.541 | 0.2670 | lr64≈128 |

no-sup 미니멀 헤드라인 = 다이얼 3종+optzone = **+0.569 (s95)**. sup 기여 정량화: ~+0.3
(같은 구성+sup = +0.855~1.053). affine-shift는 no-sup에서 무효 → gfn이 최종 미니멀 후보.
3-seed 검증 투입(s96/s97). LM 이식 검증(lm_naive vs lm_affine) 병행 시작.

### LLM 이식 검증 — TASK-AGNOSTIC 실증 (2026-07-21)

WikiText-103, GPT-2 BPE, 16M 파라미터, dim256 2블록×4loop(chunk-causal TTT), 12k step ×
16k tok = 196M 토큰, 동일 seed/데이터/val 윈도우 paired.

| exp | val loss | ppl |
|---|---|---|
| lm_naive (naive loop) | 4.310 | 74.45 |
| **lm_affine (다이얼+optzone lr×64)** | **4.080** | **59.14** |

**Δ = −0.230 nats, perplexity −20.6%** (파라미터 +0.2%, FLOPs ~0). NVS(+0.569 s95 no-sup)와
동일 레시피가 LM에서 그대로 작동 — **task-agnostic 주장 양 태스크 실증 완료.**

### Wave 21 최종 + config 오류 정정 (2026-07-21 저녁)

**정정**: r21 "gfn" config에 nl_cond가 누락돼 있었음(생성 replace 실패). 따라서:
- **no-sup 미니멀 3-seed 확정 = +0.546 (gates+film 2다이얼 + optzone)** (+0.569/+0.528/+0.540, t≥25)
  — 3다이얼이 아니라 **2다이얼**의 성과. 더 미니멀.
- r21_gf ≡ r21_gfn(bit-identical) → 훈련 결정론적 → paired 비교 신뢰도 극상.
- nl_cond의 no-sup 추가 기여는 미검증 (config 수정 완료, 대형 LM 후 1런 예정).
- sup-시대 bgsfon(+0.855/+1.053) 및 LM affine config는 nl_cond 정상 포함.

### LM 16M seed-96 재현 (2026-07-21)

| exp | val loss | ppl |
|---|---|---|
| lm_naive_s96 | 4.3395 | 76.67 |
| lm_affine_s96 | 4.0661 | 58.33 |

**2-seed 일관: ppl −20.6% / −23.9%** — LM 이식 견고 확인.

### 대형 LM 78M (fineweb-edu 500M tok, DDP×2, 2026-07-21 밤)

| exp | val loss | ppl | Δ |
|---|---|---|---|
| lm_big_naive (3블록×4loop, 78.3M) | 3.7822 | 43.91 | — |
| **lm_big_affine (+다이얼+optzone×64)** | **3.7044** | **40.63** | **−0.078 nats, ppl −7.5%** |

스케일 추세: 16M −21~24% → 78M −7.5%. 방법 유효하나 상대효과 축소(다이얼 lr 미조정/토큰예산/
대형모델의 자체 분화 가능성). 진행: 비루프 원본 앵커(l12 195M compute-match, l8, l2).
후속 후보: 78M에서 lr_mult 스윕(16/32), nl_cond 진짜 포함 시험.

### TTT-rope 규모 LLM (lact_llm 파이프라인, seq4096, fineweb-edu, 2026-07-22)

앵커(기존 런 재활용): 12층 원본 0.5B ppl 27.92 / 3B ppl 18.40.

| 모델 (0.5B) | ppl | Δ vs naive |
|---|---|---|
| loop_naive 3×4 | 31.15 | — |
| **loop_ours (+다이얼+optzone×64)** | **30.46** | **−0.69 (−2.2%)** |

**스케일 추세 (정직)**: naive→원본 격차 회복률 16M 66% → 78M 50% → TTT-rope급 20%.
강한 TTT층(muon+momentum+scale)일수록 다이얼 여지 축소 또는 lr_mult 미조정.
진행: lr_mult 16/128 스윕(0.5B), 3B pair(~5h, 1일급 헤드라인).

### TTT-rope 규모 lr_mult 스윕 (0.5B, 2026-07-22)

| lr_mult | ppl | vs naive 31.15 |
|---|---|---|
| 4 | (진행) | |
| **16** | **30.25** | **−0.90 (−2.9%), 격차회복 27%** |
| 64 | 30.46 | −0.69 |
| 128 | 30.52 | −0.63 |

**스케일 법칙 발견**: 다이얼 최적 lr이 스케일에 반비례 (NVS d256: ×128↑ / TTT-rope급: ×16).
3B ours@lr16 추가 투입 (기존 @64도 완주해 비교).

### TTT-rope 규모 3B (1일급) 최종 (2026-07-22)

| 모델 | ppl | vs naive |
|---|---|---|
| 12층 원본 (anchor) | 18.40 | — |
| loop_naive 3×4 | 21.32 | — |
| ours @lr64 | 21.13 | −0.19 |
| **ours @lr16** | **20.85** | **−0.47 (−2.2%)** |

lr 스윕 완전 bracket: 4(30.63) < **16(30.25)** > 64(30.46) > 128(30.52) @0.5B — lr16 정점.

**LM 격차 회복률 최종 사다리**: 16M 66% → 78M 50% → 0.5B 27% → 3B 15%.
다이얼 효과는 전 스케일 실재(paired, iso-compute, +0.2% params)하나 스케일·훈련길이에 따라
상대 기여 축소. LM에선 원본(비루프)이 여전히 우위 — NVS(loop가 depth를 이김)와 대비되는
태스크 의존성. 다이얼 최적 lr은 스케일 반비례(실용 지침).

### 반복 순서 ablation (TTT-rope급 0.5B, 2026-07-22)

| 순서 | naive | +다이얼@16 | 다이얼 Δ |
|---|---|---|---|
| 스택 반복 123×4 | 31.15 | 30.25 | −0.90 |
| 층별 반복 1111 2222 3333 | 32.36 | 32.29 | **−0.07 (무효)** |

1) 스택 반복 > 층별 반복 (−1.2 ppl): 전체-스택 출력의 재정제가 진짜 반복.
2) **다이얼은 스택-loop의 단계 구조가 있어야 작동** — 층별 반복(연속 반복=같은 입력분포)에선
   분화시킬 단계가 없어 무효. 방법론 작동원리의 통제 실험.

### NVS 진짜 nl_cond ablation (2026-07-22)

| 구성 (no-sup, lr×64, s95) | dPSNR | LPIPS |
|---|---|---|
| gates+film (2다이얼) | +0.569 | 0.2648 |
| +nl_cond (3다이얼, 수정 config) | +0.592 | 0.2635 |

nl_cond 한계기여 +0.023 (노이즈 범위, sup-시대 +0.03과 일관). **미니멀 방법론 = 2다이얼 확정**;
nl_cond는 선택적 3번째 다이얼(TTT 고유 서사 가치, 성능 기여는 미미).

### TTT-rope급 3B 완전 사다리 (2026-07-22 최종)

| 모델 | 블록 파라미터 | 계산 | ppl |
|---|---|---|---|
| 3층 비루프 (iso-param) | 1× | 1× | 25.08 |
| loop naive 3×4 | 1× | 4× | 21.32 |
| **ours (+다이얼@lr16)** | 1.002× | 4× | **20.85** |
| 12층 원본 (iso-compute) | 2.4× | 4× | 18.40 |

**LM 2축 결론**: iso-param에선 loop+다이얼이 −17% ppl 대승(25.08→20.85); iso-compute에선
2.4×-param 원본이 우위 — loop+다이얼 = 파라미터 2.4× 압축을 +13% ppl에 사는 압축 기법,
다이얼이 그 비용의 15%를 무료로 회수. (NVS는 loop가 iso-compute에서도 depth를 이김 — 태스크 의존.)

### NVS d512 스케일업 (s95, 2026-07-22) — 방법이 스케일에서 성장

| 모델 (d512) | PSNR | dPSNR | LPIPS | params |
|---|---|---|---|---|
| naive loop 2×4 | 23.784 | — | 0.2178 | 13.5M |
| **+2다이얼 @lr64** | **24.420** | **+0.637** | **0.2051** | 13.6M |
| +2다이얼 @lr16 | 24.073 | +0.290 | 0.2136 | 13.6M |
| L8 unique | 24.119 | +0.335 | 0.2065 | 48.1M |

1) 다이얼 효과 d256 +0.569 → d512 +0.637 (성장; LM 축소 추세와 대조 — 태스크 의존).
2) NVS lr 법칙 유지 (64≫16) — "역스케일 lr"은 LM 파이프라인 특성.
3) **헤드라인**: d512에선 naive loop < L8인데 다이얼이 역전 — ours(13.6M)가 L8(48.1M, 3.5×)을
   +0.30 상회. "다이얼이 있어야 looping이 depth를 이긴다." 3-seed는 node2 W1 진행 중.

#### d512 3-seed 확정 (node2 W1, 2026-07-22) — seed-matched paired vs 동일-seed naive loop

| 구성 (d512) | s95 | s96 | s97 | **3-seed 평균** | LPIPS(평균) |
|---|---|---|---|---|---|
| naive loop 2×4 (baseline) | 23.784 | 23.756 | 23.789 | 23.776 | 0.2183 |
| +2다이얼 @lr64 (paired ΔPSNR) | +0.637 | +0.529 | +0.596 | **+0.587** | −0.012 |
| +2다이얼 @lr16 (paired ΔPSNR) | +0.290 | +0.273 | +0.297 | **+0.287** | −0.007 |

- 3/3 seed 전부 강한 양성 (paired t: lr64 20~25, lr16 14~17 → 노이즈 아님). d256 다이얼 효과
  (+0.569)와 일관되게 d512에서도 성장 유지 → **"다이얼 이득은 스케일에서 커진다" 3-seed 확정.**
- NVS lr 법칙(64 ≫ 16) 3-seed에서 재확인. naive PSNR 자체는 seed 간 ±0.02로 매우 안정.

### 기존방법 baseline 정면 비교 (d256, s95, 각 원저 레시피, 2026-07-22)

| 방법 | dPSNR | LPIPS | 비고 |
|---|---|---|---|
| **ours (직접 2다이얼+optzone)** | **+0.569** | 0.2648 | 유일 양성 |
| DVLT time-gates (생성형 scale) | −0.092 | 0.2852 | 무효 |
| UT timestep-embed | −0.166 | 0.2893 | 음성 |
| adaLN-zero (DiT 생성형) | −0.857 | 0.3295 | gate-off 시작, 30k 예산 회복 불가 |
| LayerScale per-loop (1e-5) | −1.609 | 0.3713 | branch-off 시작, 참사 |

**형태가 아니라 처방**: 기존 조건화 형태 전부 ≤0. (주의: 각 원저의 원 세팅 반박이 아니라
"그 형태를 이 세팅에 이식"한 비교임.) W2(+optzone 구제 시험, node2)가 분해 완성 예정.

#### W2 optzone 분해 (node2, 2026-07-22) — "형태 vs 옵티마이저" 인과 분리

기존 조건화 형태에 **우리 옵티마이저 처방(wd=0 + lr×64, zero-init 조건화 파라미터 전용 그룹)**을
그대로 적용해 재측정. 표준-옵티(위 표) 대비 순수 옵티마이저 효과를 분리. (paired vs naive loop 22.204, s95)

| 방법 (d256, s95) | 표준-옵티 dPSNR | **+optzone dPSNR** | optzone 회복폭 | 판정 |
|---|---|---|---|---|
| DVLT time-gates | −0.092 | **−0.011** (t=−0.65) | +0.08 | 처방이 ≈0으로 회복하나 형태 자체 무효 |
| adaLN-zero (DiT 생성형) | −0.857 | **+0.620** (t=33.1) | **+1.48** | 처방이 참사→최상위로 역전 — 형태는 멀쩡, 옵티가 죽였던 것 |
| LayerScale per-loop | −1.609 | (node1 r23_layerscale_oz 진행) | — | — |

- **핵심**: adaLN-zero는 −0.86(표준-옵티)에서 +0.620(optzone)으로 **+1.48dB 스윙** — 우리 2다이얼
  gates+film(+0.569)과 동급/약간 상회. zero-init 조건화 파라미터를 weight-decay가 identity로
  끌어당겨 죽이던 것이 사인(死因)이었고, 처방(wd=0+고lr)이 형태와 무관하게 되살림.
- **"형태가 아니라 처방" 강화**: DVLT는 형태 자체가 looping에 무효(처방으로도 0), adaLN은 형태는
  유효하고 옵티마이저만 문제였음 — 두 실패의 원인이 다름을 optzone 분해가 드러냄.
