# IDEAS_R4 — 라운드 4 (TTT×loop NOVELTY ONLY, 10 에이전트 × 10, 2026-07-20)

방향(사용자 재설정): ep2·KD 제외(계산량 증가, generic). sup/gates 시큰둥(generic). **boost가 방향** —
fast weight=활성값(TTT) × loop가 메모리 인스턴스 다수 생성(looping) 둘 다 활용. 목표 naive +1.0dB.

## DESIGN THEOREM (이론 에이전트, autopsy에서 도출) — 생존 3조건 A∩B∩C
- **(A) 단일-스텝**: 모든 fast-weight가 fresh init에서 1 NS-step 이내. (multi-step=orbit; carry/momentum 위반)
- **(B) function 전송**: cross-loop 상태는 weight/gradient(function)로만 전송, 사용 시 **현재특징 재평가**.
  (boost +0.099 vs cumboost −1.52 = function vs frozen-vector 전송)
- **(C) WHAT not HOW-WELL**: coupling은 target/subspace/readout에 작용, lr/step/precond(전부 dead)엔 안 함.
boost = A∩B∩C 유일 시도점. 교집합 6 클래스 → 아래.

## 10 에이전트 강한 수렴 (독립 도출 = 강한 신호)

**★★★ TIER 1 — slice-boost / block-partition-boost (에이전트 1,3,4,6,8,10 독립 1순위)**
carry 1텐서, loop ℓ이 **disjoint hidden slice(d_h/n)만** delta write, apply는 full 읽음.
- orbit 구조적 불가(각 slice 정확히 1 NS step=A), delta 현재키 재평가(B), disjoint subspace 보완(C).
- 용량 무간섭 ×n (용량 binding −0.84), apply full=**무료 ensemble read**.
- **naive보다 쌈 (~0.93-1.0×, NS가 좁은 행렬에)**. 최우선 구현.

**★★ TIER 2 (에이전트 3-5개)**
- **ensemble-read** (1,2,4,8,10): 최종 apply=f_{W_n}(q)+Σγ_j f_{W_j}(q_current). boosting predictor는
  앙상블 SUM인데 마지막 잔차전문가만 읽힘. read측 boost(미검증축). ~1.03×.
- **proj-boost** (1,4,8,9,10): boost 전체빼기→방향 deflation v−(v·f̂_prev)f̂_prev. NS가 magnitude 못 살림→
  투영이 magnitude-immune. 3줄 drop-in, ~1.0×. 최저비용 A/B.
- **multiboost/ensemble-boost** (1,10): v−Σ_{j<ℓ}γ_j f_{W_j}(k) (모든 이전 재평가, stagewise 정합). ~1.02×.
- **memory-decorrelation-loss** (4,5,9,10): train-only Σcos²(f_{W_j}(k),f_{W_ℓ}(k)). **유일 strict iso-inference**.
  update rule 불변→모든 것과 조합. vanilla 불가.

**★ TIER 3**
- **detach-boost** (1,5): boost pred_prev를 detach(proper stagewise). 1줄, <1.0×.
- **boost-gain-feedback** (7): per-token boost 강도 s(e_t), e_t=이전메모리 fit. boost 오차나는 곳 안 뺌.
- **grad-orthogonal-diversity** (4,10): reset, raw gradient Gram-Schmidt(NS 전). 방향다양성.
- **transductive-write** (6): 최종 loop이 target-query에도 write(covariate-shift 수정).
- **misfit-echo** (10): 잔차 r=v−f_W(k)를 zero-init proj로 stream 주입(feature-mediated boosting).

**PROBE (싼 결정용, 에이전트 10)**
- **probe-memory-overlap**: boost 메모리들이 이미 function-space 직교인가? → diversity류 투자 여부 결정.
- **probe-boost-traj**: boost가 deeper model처럼 궤적 재배속하나? → 메모리 stage=궤적 currency 여부.

## 실행 선정 (우선순위)
1. **slice-boost** — TIER1 만장일치. 커널 구현(slice+carry+renorm). 최우선.
2. **proj-boost** — 3줄 boost 변형, ~0 cost. slice-boost 대기 중 즉시.
3. **detach-boost** — 1줄. 병행.
4. **ensemble-read** — read측 boost. 커널 apply 확장.
5. **multiboost** — stagewise 정합.
6. **memory-decorrelation-loss** — train-only strict iso.
7. **boost-gain-feedback** — state→loop.
프로브 2개는 GPU 틈새(각 30분)로 slice-boost와 병행 — diversity/trajectory 축 투자 결정.
판정: 전부 boost(22.303)와 naive(22.204) 대비 s95 paired. +0.15↑ → 3-seed.
## 부록: 에이전트별 원본

# R4 Agent 1: Boosting extensions
substrate: heads=1,d=256,d_h=512,n_loops=4. boost-style extra SwiGLU fwd ≈+10% TTT ≈~0% wall.
1. block-partition-boost★ — loop ℓ가 hidden block B_ℓ=[ℓ·128,(ℓ+1)·128) (w0/w2 cols, w1 rows)만 write(carry+delta); apply는 full 읽음. orbit 구조적 불가(좌표당 1회=LaCT 단일스텝 설계), delta로 각 block이 이전 전체 잔차 fit=진짜 multi-memory boosting 1텐서에. 용량×4 무간섭. NS도 [256,128] slice라 4× 저렴 ≤1.0×. Risk: w1 weight-norm이 block 걸쳐 rescale(scale-only 무해 추정).
2. multiboost — loop ℓ target = v − Σ_{j<ℓ} f_{W_j}(k_ℓ) (모든 이전 fresh 메모리 현재키 재평가). boost의 last-only double-count 수정=각 메모리 진짜 disjoint layer. cumboost 수정판(현재키 재평가). +6 fwd ≈+2%. Risk: drift된 초기메모리 노이즈 누적→boost로 collapse.
3. ensread — 최종 loop apply가 모든 메모리를 현재(최고정제) query로: o=f_{W_n}(q)+Σγ_ℓf_{W_ℓ}(q). boost가 stagewise 앙상블 만들지만 joint로 안 읽음. read-side 미개척축, boost에 스택. +3 apply@1loop≈+1%. γ zero-init.
4. orthboost — vi = v−(v·p̂)p̂, p̂=pred_prev/‖‖ (벡터 빼기 대신 방향 투영). rule1: Muon이 방향만 남김→boost 전체벡터 빼기는 오/과소 차감, 투영은 idempotent 과차감 불가. ~0 cost. Risk: pred_prev 방향 자체가 틀리면 참성분 삭제; rank 1dim/token만 감소→효과 작을수도.
5. detach-boost — pred_prev를 wp*.detach()로(proper stagewise). 외부 loss가 이전 loop을 "쉬운 잔차 만들기"로 훈련하는 collapse 채널 차단. 1줄, <1.0× train. Risk: joint refinement 이득 상실 가능(pli 교훈). 무료 시험.
6. gramboost — NS 전 w1_raw를 이전 메모리 w1 rows에 직교화(row-wise). value공간(boost) 아닌 FUNCTION공간 비상관. NS가 제약 anisotropy 보존(방향만 산다). ~0. Risk: weight직교≠output직교(nonlinear); β-gate zero-init.
7. shrink-boost — vi=v−ν_ℓ·pred_prev, ν_ℓ 학습(init1). Friedman shrinkage=noisy boosting 표준수정. target 방향 rotate(pre-grad)라 Muon 생존. ~0. Risk: ν≈1 collapse(neutral); multiboost의 rider로 최적.
8. adaboost-lr — per-token write lr ×= m=1−cos(pred_prev,v) detached. AdaBoost example reweighting(boost의 residual targeting과 직교). rho와 달리 훈련된 메모리 실패신호. matmul 내부라 방향 변경(Muon 생존). ~0. Risk: rho계열 ≈0이라 약한 축(neutral).
9. decorboost — train-only aux Σcos²(f_{W_ℓ}(k),f_{W_{ℓ−1}}(k)) 현재키. boost 비상관 명시 강제(미분가능 활성). inference 1.0×. Risk: 저rank 잔차 시 primary와 충돌; λ~0.01.
10. leakboost — W=weight_norm((1−α)⊙W_init+α⊙W_{ℓ−1}), per-unit α=sigmoid(a) init−4. reset↔carry unit단위 보간(느린 unit 발견). Risk: α→1 unit은 orbit(−0.45) 노출; 최후 시험.
스택: 1,2 배타(같은 multi-memory축; block이 더 싸고 orbit-proof). 3/4·6/5/9 상호직교. 7/8 rider.

# R4 Agent 2: Memory-ensemble READOUT (read-side boost)
공통 plumbing: model.py block loop이 per-loop {w0,w1,w2}를 per-block mem_bank에 저장 (carry 패턴 재사용), apply branch가 앙상블 항 추가. 전부 γ zero-init. 비용: target-only apply 1개≈1.1%, 대부분 ~1.07×. 결정적 과학질문: boost는 WRITE측에서 "옛 메모리를 현재특징 재평가" 검증 → READ측은 미검증(update/apply 비대칭의 마지막 절반).
1. final-ensemble-read — 최종 apply = f_{W_L}(q) + Σγ_ℓ f_{W_ℓ}(q_final). 각 메모리를 최종 refined query로 재조회. boost의 read-side dual. 1.07×. Risk: W_1 drift(3 loop 차이) → γ taper + 미분가능 메모리 공적응.
2. boost-sum-read — boost와 함께: o_ℓ = f_{W_ℓ}(q_ℓ)+γ f_{W_{ℓ−1}}(q_ℓ). boost가 write하지만 안 읽는 stagewise 합 복원(write/read 불일치 수정). 1 loop차라 drift 최소. boost wp 경로 재사용, 추가 plumbing 0. 1.07×.
3. mem-softmax-mixture — per-token gated mixture: α=softmax(q·κ_ℓ/τ), o+=γΣα_ℓ o_ℓ. 토큰별로 가장 잘 설명하는 메모리로 라우팅(rule3 토큰단위). 1.07×. Risk: W_L로 collapse=baseline(safe).
4. gate-confidence-read — 각 메모리 readout을 SwiGLU gate 에너지 ‖silu(q@w0_ℓ)‖로 가중(TTT 내부 활성=retrieval confidence, vanilla 불가). ~0 params. 1.07×. Risk: query norm 추종→q L2norm+per-token softmax가 상쇄.
5. drift-translator-read — 옛 메모리를 번역된 query q̃_ℓ=norm(q+B_ℓA_ℓq)로 조회(per-loop zero-init rank16 adapter). drift 정면 해결(rule2 최강). read_refine/qkv_route와 구분. 1.07×.
6. decor-ensemble-read — idea1 + train-only aux λΣcos²(o_ℓ,o_j)(detached). 메모리 비상관 명시 최적화; 미분가능 활성이라 gradient가 write chain으로 역류. inference=idea1. loss축+capacity축 동시.
7. prev-read — 매 loop ℓ>0에서 o=f_{W_ℓ}(q_ℓ)+γf_{W_{ℓ−1}}(q_ℓ). refinement 과정 자체에 cross-memory 주입(decoder뿐 아니라). reset모드 second opinion. 1 loop차. boost와 스택하면 idea2로.
8. orth-complement-read — 옛 readout의 최종read 직교성분만 추가: o+=Σγ_ℓ(o_ℓ−(ô·o_ℓ)ô). rule1 역이용(메모리는 주로 방향으로 다름), double-count 구조적 방지. idea1과 head-to-head=깨끗한 probe.
9. innovation-read — 연속 메모리 차분 읽기: o+=Σγ_ℓ(f_{W_ℓ}(q)−f_{W_{ℓ−1}}(q)). reset모드에서 per-pass 신규내용 추출, 각 쌍 1 loop차(drift 최소). Risk: boost write와는 이중차분되니 reset 전용.
10. mem-skip-decode — 최후 block TTT만 Σγ_ℓ f_{W_ℓ}(q_final)를 decoder 입력에 zero-init skip 주입(residual 망각 우회). fused_readout(feature 융합, vanilla, null +0.035)의 메모리판=깨끗한 판별. 최저비용 1.03×.
※ 1/3/4/5/8/9는 plumbing 1개 공유+apply combination rule만 다름 → read_mode 플래그로 A/B. 2/7은 기존 boost wp만.

# R4 Agent 3: Subspace/partition physics
KERNEL FACTS: (a) w0/w2 weight-norm은 dim=1 per hidden-column → untouched column 정확히 보존. w1은 output dim column이 hidden row 섞음 → block-row write가 old row 축소(grouped per-(block,col) renorm으로 수정, 0 FLOP). (b) NS는 zero row/col block 보존: masking pre-NS ≡ NS-on-slice → partition update는 strictly 더 쌈(grad+NS ×1/n, update ~15-20% 모델FLOP → 순 ~0.93×).
1. block-boost-carry★★ — carry W, loop ℓ가 hidden col [ℓ·d_h/n,(ℓ+1)·d_h/n)만 write(delta=v−f_W(k) on FULL); apply는 full 읽음. 좌표당 NS 1회→orbit 구조적 불가(rule1); delta 현재키 재평가(rule2); block이 disjoint 잔차층 특화(rule3) + boost와 달리 apply가 전체 앙상블 읽음(무료 ensemble readout). 용량 binding(−0.84 half)→×n 무간섭. ~0.93× naive보다 쌈. Risk: block0가 loop0 crude 특징으로 frozen→loop3 query로 읽힘(q/k L2norm→drift 제한, boost 1-loop gap 생존).
2. ring-refresh — 2 block 4 loop: loop ℓ에 block ℓmod2를 init 리셋+1스텝 재작성(delta vs 나머지 지속 block). 모든 write single-step-from-init(orbit無), 살아남는 block은 항상 late/refined 특징으로 작성→#1의 stale-crude 위험 제거. ~0.93×. Risk: 용량 ×2만; block 수명 2loop 짧을수도.
3. slow-fast-split — slow half는 loop0에 1회 write후 frozen carry; fast half는 매 loop init리셋+delta(v−f_slow(k)). slow 1 NS스텝, fast 매loop 리셋→orbit無. fast-reset=known-good baseline에 지속 메모리 추가(매 pass가 현재키로 boost). ~0.96×. Risk: 용량 절반이 loop0 특징에 pin.
4. ortho-grad-accretion — carry W, 각 loop post-NS update를 이전 모든 update에 trace-직교화(Gram-Schmidt) 후 add. "방향만 산다" 정면(rule1): ‖ΣG‖²=Σ‖G‖² 단조증가, 재방문 금지. ~1.0×. Risk: weight-norm이 직교성 약간 깸; 과거 G n−1개 저장(activation 메모리).
5. grow-active-set — reset, loop ℓ가 col [0,(ℓ+1)d_h/n) refit(init에서 1스텝): width 25→100% 성장. single-step(orbit無), 초기 coarse/late 완전용량(c2f_muon과 달리 width schedule). ~0.95×. Risk: 초기 1/4 메모리(write-depth −0.92 우려).
6. soft-leak-partition — #1 hard block을 완화: post-NS col scale = m_ℓ+α(1−m_ℓ), α per-loop zero-init. α=0=#1(무해), 학습이 간섭수준 선택. ~1.0×. Risk: α 크면 carry 병리(−0.45) 누출; weight-decay 필요.
7. value-band-blocks — #1 + output-channel partition: block ℓ이 residual의 output band ℓ만 회귀→w1 block-diagonal, 양 공간 비간섭. rule3 이중구조(주파수밴드 boosting). ~0.92×. Risk: 에너지 작은 band 기아(d/n=64 aggressive).
8. head-round-robin — per-head partition: loop ℓ가 head subset ℓ만 update(weight-norm/NS 이미 per-head→zero 혼합), 전 head apply. ~0.93×. Risk: 현 config num_heads=1 → ≥4-head baseline 필요(iso지만 arch 변경 confound).
9. block-gated-readout — #1 위 apply가 block b 기여를 g_{ℓ,b}로 스케일(ones-init). 메모리 BLOCK 조건화(vanilla 불가→portability 통과, gates와 달리). #1 staleness 완화. ~0 FLOP. Risk: gates축과 sub-additive; #1 선착 필요.
10. key-frame-slices — rank-structured: loop ℓ의 w0/w2가 input-dim row [ℓ·d/n,...)만 update(각 loop이 다른 key subspace 청취). 누적 W가 n loop후 full rank(rank staircase). ~0.95×. Risk: raw 좌표 key 에너지 불균등→죽은 slice; fallback 고정 random orthonormal frame.
※ 1/2/3 같은 family(refresh/persistence trade). #1 먼저, 2/3 de-risk fallback. 4/10 다른축. 6/7/9 #1 add-on.

# R4 Agent 4: Exploit NS-orbit (only direction survives Muon)
KERNEL: NS Frobenius-normalize(steps=0에도); renorm per-column(w0/w2 per hidden unit, w1 per OUTPUT channel); pre-renorm column norm(정렬/크기 정보 유일 장소)은 계산 후 폐기; o_norm은 per-token RMS라 read측 scalar gain은 지워지나 per-channel(anisotropic)은 생존.
1. ortho-residual-boost★ — boost target을 pred_prev 방향 직교성분으로: vi−=β⟨vi,p̂⟩p̂. boost 잔차의 collinear-magnitude 부분은 어차피 write 불가(Muon)→메모리 낭비. hard-orthogonal이 새 방향 보장. ~1.0×. β zero-init=boost. lr knob 아님(pre-NS 방향).
2. ensemble-read-boost — boost하 apply가 stagewise 앙상블 읽음: o=f_{W_ℓ}(q)+γf_{W_{ℓ−1}}(q). read측 magnitude/per-channel 생존. read_refine(≈0)은 같은 메모리; 이건 다른 메모리. ~1.04×.
3. slice-carry — carry, loop ℓ가 d_h/n column slice만 write(NS slice, renorm slice만). 좌표당 1 fixed-angle step=LaCT 검증 regime→carry −0.45 구조적 불가. read는 union(용량×n, boost last-only 능가). chunk(−1.27)과 달리 모든 메모리가 전 토큰 봄(param공간만 분할). ~1.0×(slice NS 더 쌈).
4. norm-shadow-read-gain — 폐기되는 pre-renorm 성장비 r_j=‖W+NS(G)‖/‖W‖(정렬/흡수량 유일 encode)를 per-channel read gain으로: w1_eff=w1·r₁^α. Muon이 못 지우는 흡수량을 read측(anisotropic 생존)으로 환류. ~1.0×(이미 계산됨). Risk: r 거의 상수일수도.
5. grad-deflate — 이전 loop post-NS update Ĝ_{ℓ−1} 저장, 현재를 deflate: G_ℓ−=β⟨G_ℓ,Ĝ⟩Ĝ. anti-momentum(momentum −0.18은 증폭, 이건 제거)→n update가 n차원 subspace span=방향 다양성. weight공간 carry frame-consistent. ~1.0×. Risk: 공유방향이 유용할수도; β zero-init.
6. loop-norm-profile — renorm TARGET을 per-loop learnable per-column: w*_norm·exp(s_ℓ). Muon이 안 건드리는 magnitude 채널(sphere 반경). anisotropic profile=per-loop hidden unit 강조 재배분. ~1.0×. Risk: uniform이면 rho2 dead; anisotropy가 베팅(zero-init).
7. diag-gain-read — update후 per-channel LS gain g_c=Σf(k)v/Σf(k)²를 read에 적용(+boost target 보정). underfit cos 0.3-0.5, 최적 대각 rescale은 closed-form, per-channel이라 o_norm 생존. precond(dead)와 달리 read 수리. ~1.03×. Risk: g≈1 null; clamp+detach.
8. transductive-write — update 토큰을 read 분포 관련성으로 pre-NS 가중: s_i=softmax(k̂_i·q̄). rho(neutral)는 magnitude(misfit) 실었음; relevance는 새 정보(read 분포)로 update를 query subspace로 회전. ~1.0×. Risk: q̄ rank-1 crude.
9. loop-decorrel-aux — train-only λΣcos²(ô_ℓ,ô_{ℓ−1}) per-token read 출력. 보완적 일을 직접 최적화(fast-weight output판, generic feature 아님). inference iso. Risk: λ 크면 task와 충돌; detached+small.
10. norm-carry — per-column norm target만 carry: w*_norm_ℓ=w*_norm_{ℓ−1}·r_j^γ, 방향은 fresh reset. NS-orbit 면역 채널(scalar, NS/rotation 없음)에 누적상태. slow/fast 2-timescale을 폐기 통계로. ~1.0×. Risk: 반경 drift가 operating point 이동; γ zero-init+clamp.
※ 1+2(boost write/read측, 쌍이 stagewise 완성), 3 단독(carry 대체), 4/6/10 norm채널(하나), 5/7/8/9 새 직교축 후보.

# R4 Agent 5: fast-weights-as-activations (train-only, inference bit-identical)
1. probe-decor★ — 모든 W_ℓ를 같은 최종 query q̂로 재적용, aux=Σcos²(f_{W_ℓ}(q̂),f_{W_m}(q̂)) per token. rule3(boost 메커니즘을 미분가능 목적함수화)+rule2(현재특징 재평가). train+4-6%, inf 1.0×. Risk: boost와 sub-additive(같은 complementarity축)→단독+on-boost 둘다; sign/rotation 트릭 회피(centered cos, λ~0.01-0.1).
2. readout-decor — in-stream apply 출력 o_ℓ(pre-c_proj) 수집, aux=Σcos²(o_ℓ,o_m). 최저비용 rule3, sup과 달리 STATE readout에 작용. train~0%. Risk: query drift와 혼동(약한 신호); residual이 aligned 출력 원할수도.
3. boost-stagewise — boost wp path detach A/B((a)frozen stage (b)joint cascade). 미분가능 활성이라 가능. 1줄. Risk: knob(mechanism 아님), 무료.
4. util-partition — hidden 사용프로파일 u_ℓ=mean|silu(k@w0)·(k@w2)|, aux=Σcos²(u_ℓ,u_m)→soft disjoint hidden units. subspace-partition의 train-only판(hard partition/inference변경 없이). train~0%. Risk: util 비상관≠function 비상관(약한 proxy).
5. mem-orth — aux=Σcos²(vec(W_ℓ−W_init),vec(W_m−W_init)) per head/weight. rule1역: Muon이 방향만 남김→per-loop 방향 직교화가 최순수 앙상블 정규화. Risk: weight직교≠function 비상관(silu 혼합).
6. anytime-state — 부분앙상블 S_m=Σ_{ℓ≤m}f_{W_ℓ}(q̂), penalize 1−cos(S_m,o_final.detach()) discount. state수준 deep sup(sup은 render축; 이건 메모리 readout=다른축, TTT고유). #1과 substrate 공유. Risk: sup과 sub-additive.
7. loop-role-contrastive — d_ℓ=norm(vec(W_ℓ−W_init)), InfoNCE(positive=타 scene 같은 loop idx). loop 위치가 data-독립 ROLE 학습(coarse→fine). 0 params. Risk: bs16 작아 noisy; gates와 sub-additive.
8. innovation-forecast — aux 1−cos(f_{W_ℓ}(k_ℓ),(v_{ℓ+1}−v_ℓ).detach()). 메모리가 다음 pass innovation 인코딩(forward-looking scratchpad). fit loss 아님(target=innovation). Risk: drift 느리면 target→0(normalize, small-norm skip).
9. ensemble-coverage — 최종키 k_n, aux=1−cos(Σ_ℓf_{W_ℓ}(k_n),v_n). SUM이 최종 value 설명(개별 underfit 허용). Risk: fit-quality loss에 가장 가까움→rule3면 null; boost와 강sub-additive.
10. inner-label-detach — update 내부만 vt.detach() 라벨화(v는 딴데 live). 메모리 콘텐츠가 실제 제약. base kernel(naive 포함) 적용. 1줄. Risk: TTT e2e가 through-update grad 필요할수도(음성 가능); 무료 진단.
※ #1/#6 substrate 공유; #2/4/5/7 near-0비용 collector; #3/#10 1줄 detach. 전부 train-only inference bit동일.

# R4 Agent 6: Update/apply asymmetry (write vs read across loops)
COST: update seg on 8 views ≈0.31 block; +1 update seg 양 block 1 loop=+7.8%; apply seg≈0.11=+2.8%. pre-NS lr reweight는 방향 변경(Muon 생존). 전부 context 유지(write set 성장/floor reweight만, 제거 없음).
1. transductive-final-write★ — 최종 loop op order=[U(in),A(in),U(tgt: key=q_t,target=v_t−f_W(q_t)),A(tgt)]. 메모리는 input-key로 fit되나 target-query로 probe(covariate shift); read 분포에 delta write=보완적. γ zero-init. +7.8%(apply 분할). Risk: self-echo→delta+zero-init gate.
2. write-once-slices — carry, loop ℓ가 hidden slice ℓ(d_h/n col)만 write, apply full. 좌표당 1 LaCT step→carry 합법(rule1). slice-carry/block-boost와 동일 계열. ~1.0×(slice NS 더 쌈). Risk: zero-init degrade 없음(구조적); w1 renorm은 written row만.
3. query-matched-write-weights — 0-FLOP: input write lr을 target-query 근접도로 pre-NS 가중 lr_i*=softplus(a(k_i·q̄_tgt)+b). 방향-only(Muon 생존), q̄ 현재특징. +0%. Risk: rho(≈0)는 fresh 메모리 self-misfit(uniform); 이건 외부신호(read 분포).
4. coverage-novelty-writes — loop≥2가 input write lr을 이전 메모리 미응답도로: c_i=‖silu(k@wp0)⊙(k@wp2)‖, lr*=(1+a·z(−c)). boost는 VALUE공간 보완, 이건 KEY공간(coverage) 보완=다른 축, boost와 스택. boost시 +0%. Risk: coverage가 residual과 상관→sub-additive.
5. disagreement-routed-target-writes — 최종 loop: read 후 U(tgt) lr∝(1−cos(o_t,v_t)) pre-NS, 재read blend. 다음 write가 현재 read의 비선형함수(apply→update 결합, TTT고유). +10.6%(상한). Risk: 초기 cos 균일→γ zero-init.
6. soft-write-rotation — per-loop 회전 soft emphasis: cohort-ℓ(view mod n) lr*(1+β_ℓ), floor 1. n 메모리가 다른 콘텐츠 특화. chunk(−1.27, 순차 hard 제거)와 달리 1스텝 soft+full context. +0%. Risk: 너무 약함; cohort 임의(fallback 학습 scoring).
7. staggered-transductive-growth — write set 성장: loop 3,4가 disjoint target half를 transductive write(q-key delta gated). query 적응을 2 pass에 분산. write-once. read-heavy/late-join death 회피(write set만 성장, read 불변). +7.8%.
8. write-value-routing — WRITE value만 per-loop channel gate v'=v⊙(1+g_ℓ) pre-NS, read/output 불변. 각 메모리가 다르게 mix된 value 저장. vanilla 불가(memorize vs emit 분리 없음). +0%. Risk: gates축 sub-additive.
9. density-equalized-write — key 중복(sky/wall)이 chunk-gradient 방향 지배→r=4 random proj로 over-represented 방향 downweight, per-loop anneal. 저장 effective rank↑. +0.1%. Risk: key_center(+0.017) 인접, 약함.
10. learned-complementary-write-keys — WRITE key만 low-rank delta k+=(k@A)B + train-only aux Σcos²(f_{W_ℓ}(q̂),f_{W_ℓ'}(q̂)). rot_bag/qkv_route는 objective 부재로 실패; 이건 명시적 complementarity 목적. inference +0.5%. Risk: addressing graveyard, aux가 유일 근거.

# R4 Agent 7: State→loop feedback (pass ℓ+1을 pass ℓ의 fast-weight STATE로 조건화)
전부 boost처럼 state dict로 carry, 항상 현재 k/q로 재평가(rule2). zero-init=baseline.
1. boost-gain-feedback★ — per-token boost 강도: v'=v−s(e_t)·f_{W_{ℓ−1}}(k_t), e_t=1−cos(pred_prev,v_t), s zero-init(s≡0 baseline, s≡1 boost). boost(+0.099)가 틀린 곳에서도 빼는 것 수정→신뢰가능 잔차만 target(rule3). pred_prev 재사용+cos 1개 ~1.0×. Risk: prev-fit 균일하면 상수 collapse(=boost); 단 fit이 0.3-0.7 실측 변동(rho와 달리 informative).
2. prevfit-lr — 이전 메모리 misfit e_t로 다음 write lr 재배분: lr←lr·(1+g(e_t−ē)) zero-init, mean-preserving. NS 전 진입→방향 변경 Muon 생존(rule1); 앙상블 실패 토큰에 1스텝 집중(rule3). ~1.02×. Risk: rho(−0)는 현재 fresh-init(상수) 사용; 여기선 훈련된 converged 메모리(변동)+mean-center.
3. drift-orthogonal-update — NS 전 grad를 이전 loop weight변위 ΔW_{ℓ−1}에 직교화. rule1: 방향만 산다→각 fresh 메모리를 새 방향으로. weight공간이라 boost function공간과 조합. ~1.001×. Risk: 자연 grad가 이미 직교면 no-op→probe로 cos(ΔW_ℓ,ΔW_{ℓ+1})>0.3 확인.
4. agreement-damp — a_t=cos(f_{W_ℓ}(k),f_{W_{ℓ−1}}(k)) 합의로 수렴토큰 lr 감쇠. 2-instance 불일치=단일메모리 불가 variance 추정. pre-NS(rule1). ~1.03×. Risk: 초기 불일치/후기 합의→per-loop 상수; mean-center+per-loop g.
5. stored-direction-novelty — ΔW1의 top 특이방향 u, key가 u에 novel한 토큰 lr 상향. 지배 저장방향은 2번째가 재저장 안 할 것(rule3), pre-NS(rule1), 현재키(rule2). ~1.005×. Risk: top-1이 chunk DC(하늘/벽)면 key_center로 퇴화; ΔW(scene-specific write) 사용→informative.
6. read-confidence-film — read 크기 m_t=‖f_{W_ℓ}(q_t)‖(read측 magnitude 생존)→zero-init token gate. 메모리 confidence/coverage 신호로 다음 pass focus. gates(index만)와 달리 data-dependent. ~1.0×. Risk: o_norm이 하류 둔감화; probe로 read norm vs PSNR anti-correl 확인.
7. head-recruit-read — per-head util로 under-used head의 SwiGLU gate에 zero-init bias(gate_bias 훅). 용량 binding→idle 용량 recruit. ~1.0×. Risk: nl_cond −0.059 + util 이미 균등화 가능; head variance probe.
8. drift-gate — per-head write-load d_h=1−cos(vec(W_ℓ),vec(W_init))(weight-norm하 유일 informative stat)→zero-init read gate. ~1.0×. Risk: 최고 dead-signal 위험(NS 고정norm 회전→scene간 d 거의 동일); head간 상대패턴 사용+probe.
9. state-stat-lr-input — [e_t,a_t,m_t] detached를 lr_fc에 추가입력(zero-init col)→feedback law 메타학습. pre-NS 방향채널(rule1), 어느 stat 죽어도 robust. ~1.03×. Risk: fp32 plumbing; loop0 DDP unused-param.
10. agreement-blend-read — apply측: o←o+g(1−a_t)(f_{W_{ℓ−1}}(q)−o), 불일치 토큰서 2-instance 앙상블 평균=variance 감소. read측 magnitude 생존(rule1), 현재query 재표현(rule2). read-depth 최중요(−1.66). ~1.04×. Risk: read_refine(≈0)은 같은 메모리; 여기선 다른 instance(진짜 앙상블). 합의시 self-disable.
※ 1/2/4/10 boost pred_prev plumbing 공유(boost와 스택). 3/5/8은 trained boost ckpt에서 30분 probe 먼저.

# R4 Agent 8: Multi-timescale memory
COST: Block≈20GF=TTT 12(kernel 10: update 6.8 incl NS 3.0, apply 3.2)+attn 3.4+MLP 4.3. 추가 full apply=+16%/block; half/target-only 반감; final-loop-only ÷4.
1. mosaic-writes — loop마다 다른 hidden quarter update(loop0 fresh), 4loop후 모자이크(모든 col 정확히 1 NS step). orbit 불가+누적. slice-carry와 동일 계열. NS quarter라 ≤1.0×. Risk: loop0 col이 loop3 drift query로 읽힘; quarter 용량(−0.84)→half 권장.
2. split-residual-pair — 매 loop 내 메모리 2개 half-width 순차: A fits v, B fits v−f_A(k)(현재키), apply=f_A(q)+f_B(q). 둘다 1스텝 fresh, intra-pass boosting → cross-loop boost와 2-level 계층. ~1.0×. Risk: half-width SwiGLU 표현력; ep2 인접(방어: fresh+새target).
3. low-rank-slow-delta — d_h/8 slow 메모리 carry+delta(target v−f_slow(k), 고정점), full fast는 매loop fresh도 v−f_slow(k). delta가 carry 구제(+0.23), tiny width가 damage 제한. slow=gist, fast=detail 명시 2-timescale. ~1.06×. Risk: d_h/8 너무 약함; f_s에 zero-init read-gate.
4. consensus-carry-columns — per column, 현재 NS방향이 이전과 일치(cos>0)하는 곳만 carry, 불일치는 init reset. 방향안정성=timescale 선택자. orbit=방향 flip-flop을 cos test가 제거. state→loop feedback. ~1.0×. Risk: 일치 col 적으면 reset 퇴화(safe null); DC col 선택 위험.
5. ensemble-final-read — boost n개 fresh 잔차메모리, 최종 loop서 진짜 앙상블 o=Σsoftmax(a)_ℓ f_{W_ℓ}(q_target). read측 boost(+0.099 trick를 read에). boost는 last stage만 읽음→앙상블 합이 수학적 정답. final/target-only ≈+6%. Risk: boost와 sub-additive; softmax가 stale 무시 학습→fallback.
6. stagger-phase-columns — hidden col 2그룹 phase-shift reset: F 매loop reset(1-step), S 짝수loop만(2-loop window). 한 텐서 한 read. S 2step(4-step orbit보다 짧음). ~1.0×. Risk: S 2-step mini-orbit; F free-ride.
7. pair-read-boost — 매loop apply가 현재+이전 gated read: o=f_{W_ℓ}(q)+tanh(g_ℓ)f_{W_{ℓ−1}}(q). 롤링 2-age readout, 중간 loop이 잔차-only 메모리로 안 읽힘. boost wp 재사용. ~1.08%. Risk: boost/#5와 같은 축(하나 선택).
8. ortho-boost — boost target이 이전 예측 방향만 제거: vt=v−(v·f̂_prev)f̂_prev. direction-only가 Muon magnitude-blind와 부합(boost의 magnitude 잔차는 renorm과 싸움). ~1.04×. Risk: pred magnitude가 informative면 under-correct; boost 대체.
9. stagger-AB — full-width 메모리 2개 엄격교대: A loop0,2 / B loop1,3. 매loop 1 update(baseline cost), apply=f_active(q)+tanh(g)f_other(q). 각 2 비연속 step(orbit 절반). ~1.08×. Risk: 2-step mini-carry; #6 col판 먼저.
10. commit-refit — half-width slow S를 재-fit(copy 아님): loop 1,3서 target=f_{W_ℓ}(k) 현재키(anti-cumboost commit). S=pass앙상블 압축요약(계층). ~1.10×(상한). Risk: S가 W 중복일수도; 최고비용.

# R4 Agent 9: Cross-loop key/query geometry (보완적 addressing)
precedent: key_center +0.017, rot_bag −0.15, qkv_route ~0, loop_temp −0.01. q/k L2-norm(방향만).
1. residual-steered-keys★ — loop ℓ write-key를 이전 메모리 잔차로 nudge: k←norm(k+(v−f_{W_{ℓ−1}}(k))@P) P zero-init. boost는 target 보완, 이건 ADDRESS 보완(같은 키 다른 잔차 토큰 충돌 분리)=2번째 용량×n 포트. boost pred_prev 재사용 ~1.01×. rot_bag과 달리 per-token data-dependent misfit-targeted zero-init.
2. write-occupancy-deflation — 이전 loop pre-NS w0-grad top-r 방향(write mass 착지처)에 현재 키 deflate: k←norm(k−γU_{ℓ−1}U^T k). fresh 메모리를 미점유 키방향으로→n 메모리가 키공간 타일링. pre-NS grad(NS는 spectrum flatten). ~1.005×. Risk: grad row space≈key cov→#3로 collapse.
3. crossloop-key-gram-schmidt — loop 간 write-key Gram-Schmidt(이전 loop top-r 직교기저 누적 QR). key_center(+0.017 rank-1)의 full 확장. 스펙트럼 겹치면 ×n 복원. ~1.01×. Risk: 이전특징 기저 staleness(방향은 느리게 drift).
4. transductive-gap-write — write-key를 (읽을 query subspace ∧ 미커버) gap으로: (I−U_{<ℓ})Q_rQ_r^T. read 수요 향한 coverage addressing(rule2+3). update/apply 비대칭 활용. ~1.01×. Risk: q,k 같은 to_qkv→gap 작을수도(safe no-op).
5. coarse2fine-key-rank — loop별 key를 top-r_ℓ PCA로 blend, r 성장(32→64→128→full). 초기 coarse prototype, 후기 fine. carried state 0(drift-free 최강 rule2). boost와 스택=coarse-to-fine boosting. ~1.01×. Risk: c2f_muon(−0.15)은 optimization축; 이건 addressing coverage축.
6. banded-write-read — #3의 대칭: write-key와 read-query 둘다 U_{<ℓ}에 deflate(별도 gate). 메모리 ℓ이 자기 band에서만 write&read(filter bank). #3의 read mismatch(SwiGLU extrapolate garbage) 수정. ~1.01×.
7. cold-cell-gate-steer — per-hidden-unit write mass a_h 추적, loop ℓ+1 gate를 cold unit으로 bias. 키공간 아닌 STORAGE-CELL 공간 보완(subspace-partition addressing판). unit identity loop-stable(drift 없음). ~1.0×. Risk: nl_cond −0.059(단 이건 per-sample coverage 신호); blunt(토큰 동일 shift).
8. rotor-with-overlap-loss — per-loop Householder rotor(q,k, β zero-init)+train-only aux ‖Ū_ℓ^T Ū_{ℓ−1}‖². rot_bag(−0.15 랜덤)/qkv_route(~0 무제약)의 결손=complementarity 압력. rotor=방향-only actuator. ~1.0× inference. Risk: null precedent; aux 약하면 identity(safe).
9. keyspan-div-loss — arch 변경 0, train-only aux로 per-loop write-key 공분산 비상관: λΣtr(C_ℓC_{ℓ−1})/‖‖‖‖. loop stack이 이미 키 회전 가능, incentive만 부여. inference=baseline(iso 구조적). Risk: nuisance 방향으로 loss 지불(PSNR 무변); portable 이의(방어: persistent 콘텐츠 분할).
10. channel-partition-masks — per-loop soft channel mask k←norm(k⊙(1+tanh(θ_ℓ))) zero-init. k renorm이라 mask가 방향 변경(loop_temp scalar와 달리). ~1.0×. Risk: loop_temp/qkv_route null→압력없이 uniform 유지; portability 실패→#9 loss의 actuator로만.
※ 1-2 boost recipe를 addressing에 포팅. 3-6 band-partition 계열(key_center 확장). 7 storage-cell 공간. 8-10 learned/loss(rot_bag 실패이유=complementarity 압력 부재 수정, null precedent 상속).

# R4 Agent 10: Theory + autopsy synthesis
DESIGN THEOREM: cross-loop TTT 메커니즘은 3조건 동시 만족해야 생존:
(A) 모든 fast-weight 텐서가 fresh init에서 1 NS-step 이내 유지(LaCT 단일스텝 regime; multi-step=orbit).
(B) cross-loop 상태는 FUNCTION(weight/gradient 방향)으로만 전송+사용시 현재특징 재평가(boost +0.099 vs cumboost −1.52 = function전송 vs vector전송).
(C) coupling이 WHAT(target/subspace/readout)에 작용, HOW WELL(lr/step/precond/count=전부 dead)엔 안 함.
audit: carry&momentum=A위반; cumboost=B위반; rho/lrs/rho2/precond/ep3/muon-sched=C위반; pli=전송無+shared-init 파괴; input transform(rot_bag/loop_temp/qkv)=meta-learned to_qkv가 재흡수. boost=A∩B∩C 유일 시도점. 교집합=6 클래스: ensemble-residual targeting, partitioned single-step writes, ensemble readout, direction-diversity, fresh state→feature feedback, train-only complementarity loss.
1. ensemble-boost — pass ℓ이 v−Σ_{j<ℓ}γ_j f_{W_j}(k_current) fit(모든 이전 메모리 현재키 재평가, γ shrinkage). boost는 ℓ−1만 빼서 pass3가 pass1 재저장. 용량 strictly×n. Friedman stagewise. ~1.02×. Risk: capacity 포화(sup과 sub-additive).
2. slice-boost★★★ — carry 1텐서, loop ℓ이 hidden slice ℓ(d_h/n)만 delta write(v−f_W_full(k)), apply full=무료 ensemble read. 용량 binding(−0.84)→무간섭×n, carry 부활 orbit위험0(slice당 1 NS step). MoM router없이. ≤1.0×(NS 좁아 쌈). Risk: loop0가 untrained init slice 3개 읽음; d_h/4 약할수도(fit 병목 아님-rule3).
3. ensemble-read — 최종 pass apply=f_{W_n}(q)+Σγ_j f_{W_j}(q_current). boosting predictor=앙상블 SUM인데 현재 마지막 잔차전문가만 읽음. read_refine(≈0)은 같은메모리→다른메모리 읽기는 미검증. ~1.03×. Risk: stream이 암묵 구현(γ~0).
4. PROBE probe-memory-overlap — 모든 per-loop 메모리를 공유 key로 평가, (i)pairwise cos (ii)잔차norm 감쇠. boost 메모리가 이미 직교면 7/8 무의미→1/2/3 투자; 강겹침이면 diversity가 최저비용. rule3 직접측정. 50 LOC read-only.
5. PROBE probe-boost-traj — boost vs naive vs L2×6-sup 궤적: boost가 deeper model처럼 재배속하나(loop3 덜 수렴, 마지막 큰 gain, LPIPS late drop)? 메모리 stage가 궤적 currency인지 결정→1/2/10 투자 여부. 코드 0(--ckpt 지원).
6. misfit-echo — update후 잔차 r=v−f_W(k)(현재 fresh)를 zero-init d×d proj로 update-token 출력에 주입. r=두 양성(delta+0.23, boost target)의 핵심량→다음 pass 특징이 메모리 실패처로 라우팅(feature-mediated boosting). Titans surprise 유사. ~1.01×.
7. grad-orthogonal-diversity — reset모드, 각 pass raw gradient를 이전들에 Gram-Schmidt(NS 전). "방향만 산다" 정면→직교방향=비중복 메모리. Muon-OGD(continual). ~1.0×. Risk: weight직교≠function 보완(probe4가 판정).
8. memory-decorrelation-loss — train-only Σcos²(f_{W_j}(k),f_{W_ℓ}(k)). rule3 직접, update rule 불변→모든 것과 조합, inference 정확히 1.0×(유일 strict iso). vanilla 불가. train+3-5%. Risk: λ 튜닝.
9. proj-boost — boost 전체빼기를 방향 deflation으로: v−(v·f̂_prev)f̂_prev. NS가 magnitude 신뢰불가→전체빼기는 magnitude 오차 주입, 투영은 magnitude-immune. dot-product inner objective의 loss-consistent 잔차. 3줄, ~1.0×. 최저비용 A/B(subtraction vs projection 판별).
10. stage2-boost — pass마다 fresh 메모리 2개, B가 A의 same-pass 잔차 v−f_A(k) fit, apply=f_A(q)+f_B(q). 2n stage in n pass=2n-loop 궤적. 잔차가 정확히 fresh(drift 0). ep2와 달리 2 weak learner(boost 물리). ~1.07×. Risk: PI가 ep2 compute-adder로 볼수도→probe5 확인 후.
