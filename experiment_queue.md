# 실험 큐

## WAVE 18 (라운드4 — TTT×loop NOVELTY, IDEAS_R4.md, 2026-07-20)

방향 전환: ep2/KD 신규 금지. boost 계열 TTT×loop 고유 메커니즘만. DESIGN THEOREM A∩B∩C.

큐 (우선순위순, 전부 boost 22.303 / naive 22.204 대비 s95 paired):
1. **r18_slice_boost_s95** ★ — carry + loop마다 disjoint hidden slice(d_h/n) delta write, apply full.
   orbit 구조적 불가+용량×n+무료 ensemble read. naive보다 쌈. 10에이전트 만장일치 1순위. (커널 구현)
2. **r18_proj_boost_s95** — boost 전체빼기→방향 deflation v−(v·f̂_prev)f̂_prev. 3줄. ~1.0×.
3. **r18_detach_boost_s95** — boost pred_prev detach(proper stagewise). 1줄.
4. **r18_ensemble_read_s95** — 최종 apply가 모든 per-loop 메모리를 현재 q로 읽음(read측 boost). ~1.03×.
5. **r18_multiboost_s95** — v−Σ_{j<ℓ}γ_j f_{W_j}(k) (모든 이전 재평가, stagewise). ~1.02×.
6. **r18_decorloss_s95** — train-only Σcos²(f_{W_j}(k),f_{W_ℓ}(k)). strict iso-inference.
7. **r18_boostgain_s95** — per-token boost 강도 s(e_t) (이전메모리 fit gating).

PROBE (GPU 틈새, slice-boost와 병행):
- probe-memory-overlap: boost 메모리 function-space 직교도 → diversity 투자 결정.
- probe-boost-traj: boost 궤적 재배속 여부 → 메모리 stage=궤적 currency 여부.

진행 중(KD 방향 마지막 데이터, 완주 후 슬롯 반납): warm6_kd(g0), optzone(g1), anneal654(g2 곧 완료), lpips-KD(g3).

# 실험 큐

## WAVE 17 (라운드3 선정 — IDEAS_R3.md, 2026-07-20)

실행 중:
- g0: r16_loop_ep2gs_kd6_s96 (KD 스택 seed 검증)
- g2: r16_loop_ep2gs_kd6_s97 (KD 스택 seed 검증)
- g1: r16_loop_l2x8_sup_s95 (L2×8+sup teacher, eval 직전)

큐 (우선순위순):
1. **r17_ep2gs_kdtraj_s95** — ep2gs+KD + trajectory waypoint KD (student ℓ ← teacher t(ℓ)={2,3,5,6}
   렌더, w=0.3). `--kd_traj` 신설. → g3 즉시.
2. **r17_ep2gs_kd8_s95** — teacher를 L2×8+sup로 교체 (teacher 품질 스케일링). 코드 0줄. → g1.
3. **r17_warm6_kd_s95** — `--init_from` (L2×6+sup teacher weights, shape 동일 검증됨) + KD + lr 5e-5.
4. **r17_anneal654_s95** — n_loops 6(0–6k)→5(–12k)→4(–30k) 결정적 anneal, 배포 4. 1.15× train.
5. **r17_ep2gs_optzone_s95** — per-loop 파라미터(wd 억압 발견) wd=0 + lr×8 존 분리, ep2gs 위.
6. **r17_ep2gs_loopnoise_s95** — pass별 σ=[.12,.06,.03,0] feature noise (train만), ep2gs 위.
7. **r17_micropass_s95** — [B1B2,B1,B1B2,B1,B1B2] 5-pass @1.00×.
8. probe_traj (×6 vs ×4 per-loop 곡선) — GPU 틈새 30분.
모든 신규 런에 `--ema` 동봉 (eval 이중 측정, attribution 무비용).

2차 대기: feat-KD, bpb, state2gate, TokenPyramid, ensemble-teacher, shortcut-consistency,
sup-γ curriculum, c2f-sup. (상세 IDEAS_R3.md)

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

## WAVE R2 — 라운드-2 100아이디어 (task-agnostic, IDEAS_R2.md) 선별 큐

geometry 제외(task-specific). 통합이론(agent 9): 이득은 비선형/출력측/cross-pass 다양성에서만.
- 실행중(WAVE-A): Attn-OutGate(g0,LT2 +1.43), QKV-Route(g2,transform), RotBag(g3,diversity/이론test).
- WAVE-B: NL-Cond(SwiGLU preact bias, 이론#1), StochDepth(loop수 샘플링, 5-6에이전트 수렴),
  distill(deep-teacher, +0.44 iso-inference), hep(ep2 innovation 2nd step, ~free upgrade).
- WAVE-C: StreamNorm/CoreNorm(RMSNorm@depth), Fused-Readout/lob(output composition), DropLoop,
  ConcatInject(Huginn concat), phalt(per-token halting), SCW(target self-write).
양성이면 ep2+sup+gates(+0.283)에 스택 → 3-seed.

## WAVE 10 — 100-아이디어(10 subagent) 선별 큐 (상세: IDEAS_100.md)

선별 기준: 다수 에이전트 수렴 + 직교축(stacking law) + 미니멀·0~tiny param·zero-init + ≤1.1× + 실패군 무관.
확정 winner(iso): boost+0.099, ep2+0.077, sup+0.073, gates+0.033 → ep2+sup+gates 스택 +0.187.

### TIER 1 — 다수 에이전트 flagship (최우선 구현·실행)
1. **precond_w1 (Gauss-Newton/RLS readout)** [agents 1,10,4]: w1 update를 `(HᵀH+λI)⁻¹`로 좌
   preconditioning(Neumann/Richardson J항, solve 없이). NS는 scale-invariant→방향만 살아 lr-trap 회피.
   binding underfit(cos 0.3~0.5) 직격 + carry에 고정점. 축=conditioning(신규). **구현 중.**
2. **cumboost (cumulative-residual boost)** [agents 9,5,2,6]: boost 이중계산 버그 수정 — 토큰공간 잔차 r을
   carry, `r←r−h_{W_ℓ}(k)` 단조감소(v_0 anchor). 진짜 gradient boosting. 축=capacity(boost upgrade). +0.15~0.25 추정.
3. **feat_mom (feature Anderson/Nesterov)** [agents 4,6,7]: `x += β[loop]·(x−x_prev)`, β zero-init.
   loop 자체 수렴방향 외삽 = 공짜 유효깊이. Muon 무관(feature축). ~0 FLOPs. 축=feature-trajectory(신규).
4. **epavg (Polyak iterate averaging)** [agents 1,2,7]: ep 반복의 w1 iterate 평균을 apply에. ep3 포화(궤도)를
   fit으로 전환. 축=fit(ep2 확장/대체).

### TIER 2 — 강력·직교축
5. **c2f_muon (spectral coarse-to-fine NS steps)** [agents 8,10]: muon steps loop마다 {2,4,6,8}(총20=baseline).
   NS는 update 모양만 바꿈→lr-trap 회피. 축=update-spectrum(신규). LPIPS.
6. **loop_temp (per-loop key/query temperature τ)** [agents 5,8]: q,k에 per-loop τ(softplus, init 1).
   forward silu 모양 변경→lr-trap 회피. 축=bandwidth/radial(신규).
7. **key_center (DC decorrelation)** [agent 5]: `k −= α·k.mean(chunk)`, α zero-init. key crosstalk 제거→
   단일 인스턴스 유효용량↑. 축=address-decorrelation(신규).
8. **plucker_addr (reciprocal-product addressing)** [agent 3]: epipolar bilinear `d_t·m_i+m_t·d_i`를 k/q에
   주입(γ zero-init). 이전 프로젝트 camera +1.7dB의 TTT판. 축=addressing(신규, geometry).
9. **opcompose (deeper fast-net)** [agent 2]: 연산자 합성 `f_{W_n}∘…∘f_{W_1}`, γ zero-init blend. 축=depth(신규). 高분산.
10. **readout_agg (learned per-loop readout)** [agents 2,4,10]: decode를 `Σw_ℓ x_ℓ`(softmax, one-hot init)에서. 축=readout(신규).

### TIER 3 — train-only (eval FLOPs 0)
11. **self_distill** [agents 4,7]: `Σ MSE(render_ℓ, sg(render_{ℓ+1}))`. 축=loss.
12. **fit_sup** [agent 6]: `Σ(1−cos(f_W(k),v))` aux loss. 축=loss/memory-space.

### 실행 원칙: 각 단독 s95 측정(vs naive 22.204) → 양성이면 ep2+sup+gates 스택에 추가 → 3-seed.
### 최종 스택 = 각 직교축 승자 + sup + gates. 목표 누적 +0.5.

## WAVE 7 (TTT×loop 고유 물리 — 코드 완료, 축 지도 완성)

목표: 동일 architecture·~1.0× compute로 naive loop(+delta+sup 대비) 순증. 각 메커니즘을
단독(vs naive 22.204)과 sup 스택(vs sup 22.287)으로 이중 측정.

- **ep2** (`loop_l2x4_ep2`): inner update epochs=2 — underfit 직접 공격, drift-free 다중스텝. ~1.05×.
- **boost** (`loop_l2x4_boost`): carry+boost — 신선 메모리에 이전 loop 잔차만 기록,
  유효 용량 ×n_loops (binding 제약 직격). ~1.06×.
- 스모크 통과 후 각각 s95 단독 + s95 sup스택. 유망하면 3-seed + momentum/rotation 파생.
- 미구현 후보(결과 보고 투자): cross-loop momentum(Muon Eq.20 복원), per-loop 직교회전 addressing.

## WAVE 5 (big-swing: 목표 naive loop +0.5dB @ iso-compute) — 완료(전멸/평탄, RESULTS.md)

- GPU2 실행 중: **r5_loop_l2x5_lj_delta_s95** (late-join, 0.99×, +sup) ← L2×6 스케일링 발견 기반 1순위
- 다음 슬롯: **r5_loop_l2x4_rfb_s95** (render feedback + delta, `--loop_sup_weight 0.5`)
- 다음 슬롯: **r5_loop_l2x4_gates_s95** (Déjà View branch/state gates + LT2 rho)
- 다음 슬롯: **r5_loop_l2x4_chunk_s95** (4-chunk 순차 delta write, muon 5→1)
- wave 6 후보: latejoin×gates×chunk 승자 조합, L2×7 strided-coarse (agent 제안), SfM-v2,
  analysis-by-synthesis final pass, sfm(이미 config 있음), L2×6-lj(1.16×, 상한 확인용)

## QUEUE (구 라운드 — 참고용)

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
