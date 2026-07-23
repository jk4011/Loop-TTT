# NODE_QUEUE — 노드 간 실험 작업 큐 (Claude ↔ Claude 협업)

## 프로토콜 (두 노드의 Claude 모두 준수)

- **역할**: node1 Claude(주 세션)가 실험을 설계해 여기에 추가. node2 Claude는 이 파일을 보고
  `[PENDING]` 항목을 위에서부터 claim해 실행한다.
- **Claim**: 실행 시작 전에 해당 항목을 `[RUNNING node2 gpu<i> <시각>]`으로 수정-저장.
  (출력 dir `outputs/<exp>`가 이미 존재하고 train.log가 갱신 중이면 남이 돌리는 것 — 건드리지 말 것.)
- **완료**: eval 결과가 나오면 `[DONE ppl/psnr=...]`로 갱신하고, RESULTS.md에 항목 규격대로 추가.
- **git**: 커밋 전 반드시 `git pull --rebase`. 충돌 시 RESULTS/NODE_QUEUE는 양쪽 내용 보존(append 성격).
  체크포인트(*.pth,*.pt)는 절대 커밋 금지(.gitignore 됨).
- **노드 준비(node2 최초 1회)**: NVS 실험 전 `/tmp/re10k` reshard 필요:
  `bash lact/lact_nvs/node2_batch.sh`의 1단계 참조 또는:
  ```
  PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
  $PY lact/lact_nvs/data_preprocess/reshard_re10k.py --src /NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k/train --odir /tmp/re10k/train --index /tmp/re10k/train_index.json --workers 16
  $PY lact/lact_nvs/data_preprocess/reshard_re10k.py --src /NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k/test  --odir /tmp/re10k/test  --index /tmp/re10k/test_index.json  --workers 16
  ```
- **실행 관례**: NVS는 `lact/lact_nvs`에서 `bash chain_run.sh <gpu> <exp> <config> <seed> [args]`
  (훈련→eval 자동). LLM은 `lact/lact_llm_loop`에서 `./run_loop.sh <gpu> <exp> [args]`.
  node2는 GPU 0-5 (6개). 실험명은 아래 큐에 적힌 그대로 사용(노드 간 충돌 방지의 핵심).
- **판정 기준**: NVS는 `python compare_evals.py outputs/r1_loop_l2x4_s<seed>/eval.json outputs/<exp>/eval.json`
  (동일 seed paired). LLM은 train.log 마지막 VAL ppl.
- **노드 사망/가변 자원 대응** (node2는 Slurm이라 언제든 꺼질 수 있고 GPU 수도 달라질 수 있음):
  - 커맨드의 gpu 인덱스는 **제안**일 뿐 — 실행 노드가 시작 시 `nvidia-smi`로 가용 GPU 수를 세고
    그 수만큼만 claim, 인덱스는 자유 배정 (RUNNING 표기에 실제 gpu 기록).
  - **stale 회수**: `[RUNNING ...]`인데 `outputs/<exp>/train.log`가 **30분 이상 미갱신**이면 고아로
    간주 — 어느 노드든 같은 실험명·같은 커맨드로 재실행해 이어받고(auto-resume), RUNNING 줄의
    노드/시각을 갱신한다. eval.json이 이미 있으면 실행 없이 DONE 처리만 한다.
  - 노드가 살아나면 자신의 RUNNING 항목부터 위 규칙으로 자가 복구.

## 큐

### W1. d512 스케일업 3-seed 완성 (6런 = GPU 0-5에 병렬)
- [DONE PSNR=23.756 LPIPS=0.2187] r22_d512_naive_s96 — `bash chain_run.sh 0 r22_d512_naive_s96 config/loop_l2x4_d512_p16.yaml 96`
- [DONE PSNR=23.789 LPIPS=0.2183] r22_d512_naive_s97 — `bash chain_run.sh 1 r22_d512_naive_s97 config/loop_l2x4_d512_p16.yaml 97`
- [DONE PSNR=24.285 Δ+0.529] r22_d512_gf_lr64_s96 — `bash chain_run.sh 2 r22_d512_gf_lr64_s96 config/loop_l2x4_gates_film_d512_p16.yaml 96 --loop_param_lr_mult 64`
- [DONE PSNR=24.385 Δ+0.596] r22_d512_gf_lr64_s97 — `bash chain_run.sh 3 r22_d512_gf_lr64_s97 config/loop_l2x4_gates_film_d512_p16.yaml 97 --loop_param_lr_mult 64`
- [DONE PSNR=24.029 Δ+0.273] r22_d512_gf_lr16_s96 — `bash chain_run.sh 4 r22_d512_gf_lr16_s96 config/loop_l2x4_gates_film_d512_p16.yaml 96 --loop_param_lr_mult 16`
- [DONE PSNR=24.087 Δ+0.297] r22_d512_gf_lr16_s97 — `bash chain_run.sh 5 r22_d512_gf_lr16_s97 config/loop_l2x4_gates_film_d512_p16.yaml 97 --loop_param_lr_mult 16`
- 기록: **W1 완료. d512 3-seed 확정: gf@lr64 평균 +0.587, gf@lr16 평균 +0.287 (3/3 양성). RESULTS.md 기록됨.**

### W2. 기존방법 baseline + optzone 분해 (W1 끝나는 GPU부터)
- [DONE PSNR=22.193 Δ-0.011] r23_dvlt_oz_s95 — `bash chain_run.sh 0 r23_dvlt_oz_s95 config/loop_l2x4_dvlt_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (LOOP_PARAM_KEYS에 dvlt_mlp 추가 완료: commit e1f064d. dvlt_temb은 buffer라 제외. optzone가 표준-옵티 -0.09를 ≈0으로 회복, 그러나 gates+film +0.569엔 못 미침 — 형태 문제.)
- [DONE PSNR=22.824 Δ+0.620] r23_adaln_oz_s95 — `bash chain_run.sh 1 r23_adaln_oz_s95 config/loop_l2x4_adaln_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (adaln_emb/adaln_mlp 키 추가 완료: commit e1f064d. **표준-옵티 -0.857 → optzone +0.620, +1.48dB 스윙!** 형태는 유효, 옵티가 죽였던 것. RESULTS.md 기록됨.)
- [DONE PSNR=22.284 Δ+0.080] r23_layerscale_oz_s95 — `bash chain_run.sh <g> r23_layerscale_oz_s95 config/loop_l2x4_layerscale_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (표준-옵티 -1.609 → +0.080, +1.69 회복. scale-단독 형태는 이득 작음. RESULTS.md 분해표 3/3 완성.)
- 기록: **W2 완료.** 분해 결론: 처방(optzone)은 필요조건, 형태 요건은 shift+입력측 변조. RESULTS.md 참조.

### W3. LM 백로그 (여유 GPU 시)
- [PENDING] perlayer 3B pair는 보류(0.5B에서 이미 결론). 대신:
- [DONE ppl=21.05] lm loop_ours_3b_lr8 — `./run_loop.sh 2 loop_ours_3b_lr8 --num_hidden_layers 3 --n_loops 4 --loop_dials true --loop_param_lr_mult 8 --bs 8 --token_budget 3000000000` (lr 정점 미세화: 8 vs 16)
  (**ppl 21.05 (3B, 91552 steps). lr16 앵커 20.854 < lr8 21.05 → LM 3B 정점은 lr16. HF 429는 지연 재시도(attempt 2)로 극복. RESULTS.md 3B 사다리에 기록됨.**)


### W4. inner-affine "사이사이" (사용자 아이디어: 공유 행렬 사이 전 이음새에 loop별 affine)
- [DONE PSNR=22.864 Δ+0.660] r24_gf_inner_lr64_s95 — `bash chain_run.sh 3 r24_gf_inner_lr64_s95 config/loop_l2x4_gf_inner_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (t=32.0. **gf+inner full +0.660 > gf 2다이얼 +0.569 (inner가 +0.091 가산), > inner_only +0.320.** 스택 최상, RESULTS.md 기록됨.)
- [DONE PSNR=22.523 Δ+0.320] r24_inner_only_lr64_s95 — `bash chain_run.sh 4 r24_inner_only_lr64_s95 config/loop_l2x4_inner_only_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (t=27.7. NVS inner-affine 단독 +0.320 — 양성이나 gf 2다이얼 +0.569보다 약함(LM에선 inner_only가 3다이얼 격파 → 태스크 의존). gf_inner full 결과 나오면 RESULTS 함께 기록.)

#### W4b. NVS inner 사이트 분해 (LM W5 결과가 attn/MLP-이음새 우세를 시사 → NVS에서 사이트 귀속)
- [DONE PSNR=22.831 Δ+0.627] r24_gf_inner_ttt_lr64_s95 — `bash chain_run.sh 0 r24_gf_inner_ttt_lr64_s95 config/loop_l2x4_gf_inner_ttt_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + TTT 이음새만)
- [DONE PSNR=22.844 Δ+0.640] r24_gf_inner_attn_lr64_s95 — `bash chain_run.sh 1 r24_gf_inner_attn_lr64_s95 config/loop_l2x4_gf_inner_attn_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + attn 이음새만)
- [DONE PSNR=22.839 Δ+0.635] r24_gf_inner_mlp_lr64_s95 — `bash chain_run.sh 2 r24_gf_inner_mlp_lr64_s95 config/loop_l2x4_gf_inner_mlp_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + MLP 은닉만)
- [DONE PSNR=22.850 Δ+0.646] r24_gf_inner_qkv_lr64_s95 — `bash chain_run.sh 3 r24_gf_inner_qkv_lr64_s95 config/loop_l2x4_gf_inner_qkv_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + qkv측만, attn+TTT)
- [DONE PSNR=22.763 Δ+0.559] r24_gf_inner_out_lr64_s95 — `bash chain_run.sh 0 r24_gf_inner_out_lr64_s95 config/loop_l2x4_gf_inner_out_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + 출력측(c_proj직전)만, attn+TTT)
  (node2: t=31.6. 출력측 inner만 = +0.559 ≈ gf 2다이얼 +0.569. W4b 사이트 분해표는 node1이 조립.)
- 판정 프레임: 전부 vs gf@64 +0.569. full(진행중) vs 단일사이트 합 → 가산성; qkv vs out → 생성부/출력부; TTT vs attn/MLP → LM(W5)의 "TTT-이음새 무효" 재현 여부.


### W5. inner-affine LM 검증 (양 태스크 검증의 LM쪽 — 최경량 16M/WikiText, 기존 anchor 재활용:
naive 74.45 / 3다이얼 59.14. 각 1 GPU ~1.5h. lact/lact_nvs에서 실행)
- [DONE ppl=55.20] lm_affine_inner_s95 — `CUDA_VISIBLE_DEVICES=<g> /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python train_lm.py --config config/lm_loop_l2x4_affine_inner.yaml --expname lm_affine_inner_s95 --data_dir /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/wikitext103_gpt2 --steps 12000 --bs 16 --seed 95 --val_every 3000 --loop_param_lr_mult 64 > outputs_lm_affine_inner.log 2>&1` (3다이얼+inner full; vs 59.14)
  (run_lm_w5.sh 래퍼로 실행. **ppl 55.20 vs 3다이얼 59.14 = −3.94, naive 74.45 대비 −19.25.** RESULTS는 W5 3종 완료 후 표로.)
- [DONE ppl=59.45] lm_affine_innerttt_s95 — 같은 형식, config `lm_loop_l2x4_affine_innerttt.yaml`, expname `lm_affine_innerttt_s95`, 로그 `outputs_lm_affine_innerttt.log` (TTT inner만) — run_lm_w5.sh 래퍼 사용
  (**ppl 59.45 ≈ 3다이얼 59.14 — TTT 이음새 inner만으론 무효. LM 이득은 attn/MLP 이음새에서 온다(affine_inner 55.20과 대조).**)
- [DONE ppl=57.68] lm_inner_only_s95 — 같은 형식, config `lm_loop_l2x4_inner_only.yaml`, expname `lm_inner_only_s95`, 로그 `outputs_lm_inner_only.log` (inner만; vs naive 74.45) — run_lm_w5.sh 래퍼 사용
  (**ppl 57.68 — inner-affine 단독이 3다이얼 59.14를 이김(−1.46)! W5 3종 완료, RESULTS.md 표 기록됨.**)
- 주의: 실행 dir은 lact/lact_nvs (train_lm.py 위치). launch_exp의 TRITON/INDUCTOR 캐시 export 복사할 것.
- 판정: outputs/<exp>/eval_lm.json의 val_loss/ppl.

### W7. ★최우선★ 3다이얼+inner full — large LM (d768 3L×4, 3B tokens, ~1일/GPU)
- [DONE ppl=20.85] lm loop_oursinner_3b_lr16 — `./run_loop.sh <g> loop_oursinner_3b_lr16 --num_hidden_layers 3 --n_loops 4 --loop_dials true --loop_inner full --loop_param_lr_mult 16 --bs 8 --token_budget 3000000000`
  (**★W7 완료★ VAL ppl 20.85 (3B, 91552 steps). vs ours(3다이얼만) 20.854 ≈ 동일 — 3B/d768 대규모에선
  inner-affine이 3다이얼 위에 이득 없음(16M에선 −3.94 도움됐던 것과 대조 → inner 이득은 스케일 의존,
  대규모서 소멸). naive_3b 21.322 대비 −0.47은 다이얼 몫. dtype 재착수 성공. RESULTS.md 기록됨.**)
  (node2 재착수: node1 dtype 수정(layer_lact_swiglu.py:590/1056, modeling_lact.py:51 .to(dtype)) pull 확인, 스테일 outputs 삭제 후 w7_retry.sh 래퍼로 gpu0 시작. 스텝 진입 검증 완료(FlashAttention 통과).)
  (**dtype 버그 수정 완료(node1): fp32 다이얼×bf16 activation 승격이 원인 — qkv/out/MLP 3사이트에
  `.to(activation.dtype)` 캐스트 추가. FlashAttention 경로 통과하는 재현 스크립트로 크래시 재현→수정→
  autocast fwd/bwd 스모크 통과 확인. outputs/loop_oursinner_3b_lr16 잔해는 지우고 새로 시작할 것.
  429 시 node2의 지연-재시도 래퍼 관례.**)
  (**node2 진단: init 직후 `RuntimeError: FlashAttention only support fp16 and bf16 data type`
  (flash_attn_interface.py:91 _flash_attn_forward). loop_inner="full" 경로에서 attention q/k/v가
  fp32로 FlashAttention에 들어감 — 새 inner-affine(loop_qkv_s/b 등) 파라미터가 fp32라 autocast
  밖에서 q/k/v dtype을 fp32로 올리거나, 그 경로가 bf16 autocast에 안 감싸인 것으로 추정.
  스모크가 통과한 건 짧은 seq/eager-attn 경로로 FlashAttention을 안 탔기 때문으로 보임.
  429 아님·일시적 아님 → 재시도 무의미. W3 스트림은 무사(2.33B/3B 계속). **node1 코드 수정 필요:**
  loop_inner qkv/o affine 적용 후 q/k/v를 .to(bf16) 하거나 해당 블록을 autocast로 감쌀 것.
  gpu0 반납.**)
  (구현 완료: layer_lact_swiglu qkv직후/o_proj직전 + LoopInnerMLP 은닉 affine, 스모크 통과.
  앵커: naive_3b 21.322 / ours(3다이얼)_3b_lr16 20.854 / orig_3l 25.085. **node1 권장** — node2는
  3B fineweb 스트림이 이미 1개 돌고 있어 HF 429 재발 위험. 429 시 지연 재시도 관례대로.)

### W6. seed 승격 (단일-seed 발견의 3-seed 확정; 여유 GPU부터 위에서 순서대로)
- [DONE ppl=55.42] lm_affine_inner_s96 — W5 첫 항목과 같은 형식(run_lm_w5.sh 래퍼 가능), `--seed 96`, expname `lm_affine_inner_s96` (LM 최고치 55.20의 seed 재현 — 양태스크 주장 보강, ~1.5h)
  (**ppl 55.42 (s95 55.20) — LM 최고치 2-seed 평균 55.31. affine_inner 스택 최상 확정. RESULTS.md 표 seed 반영됨.**)
- [DONE ppl=57.13] lm_inner_only_s96 — W5 셋째 항목과 같은 형식, `--seed 96`, expname `lm_inner_only_s96` ("inner 단독 > 3다이얼" 재현, ~1.5h)
  (**ppl 57.13 (s95 57.68과 일관) — inner_only가 3다이얼 59.14를 2-seed 모두 격파. affine_inner_s96 나오면 LM seed-promo 표로.**)
- [DONE PSNR=22.686 Δ+0.598] r23_adaln_oz_s96 — `bash chain_run.sh 4 r23_adaln_oz_s96 config/loop_l2x4_adaln_d256_p16.yaml 96 --loop_param_lr_mult 64` (optzone-구제 +0.620의 seed 확인; paired 기준 r1_loop_l2x4_s96)
  (**t=25.3. s95 +0.620과 일관 — adaln optzone 구제 seed 안정. s97 나오면 RESULTS에 3-seed 확인.**)
- [DONE PSNR=24.338 Δ+0.582] r22_d512_gf_lr128_s96 — `bash chain_run.sh 3 r22_d512_gf_lr128_s96 config/loop_l2x4_gates_film_d512_p16.yaml 96 --loop_param_lr_mult 128` (d512 lr 정점 128 vs 64 판별)
  (**t=25.9. lr128 +0.582 > lr64 +0.529 (s96, +0.053). s97 나오면 lr128-vs-64 판별 기록.**)
- [DONE PSNR=24.435 Δ+0.646] r22_d512_gf_lr128_s97 — 같은 형식, seed 97, expname `r22_d512_gf_lr128_s97`
  (**t=21.3. d512 lr128 2-seed avg +0.614 > lr64 +0.563 — 정점 lr128 쪽 약간 이동. RESULTS.md 기록됨. W6 완료.**)
- [DONE PSNR=22.443 Δ+0.383] r23_adaln_oz_s97 — `bash chain_run.sh 1 r23_adaln_oz_s97 config/loop_l2x4_adaln_d256_p16.yaml 97 --loop_param_lr_mult 64`
  (**t=19.6. adaln optzone 3-seed 확정: +0.620/+0.598/+0.383 → 평균 +0.534, 3/3 유의. RESULTS.md 기록됨.**)

### W8. 오버헤드 벤치 + NVS 최고구성 seed 승격 (node2 유휴 GPU용; W7 다음 순위)
- [DONE 아래 6줄] bench_overhead — `bash bench_overhead.sh <g>` (lact/lact_nvs에서. naive/gf/gf+inner ×
  eager/compile 6종, 600스텝씩 ~1.5h. /tmp/re10k 필요. 결과 6줄("name: it/s")을 이 항목 DONE 노트에
  그대로 기록 — "compile이 다이얼 오버헤드를 얼마나 회수하는가" 측정이 목적. 체크포인트는 스크립트가
  자동 삭제, eval 없음.)
  ```
  naive_eager: 7.403
  gf_eager: 6.582
  gfinner_eager: 5.122
  naive_compile: 15.330
  gf_compile: 14.153
  gfinner_compile: 13.957
  ```
  (**해석: eager에선 gf+inner가 naive 대비 −31%(7.40→5.12), gf만은 −11%. compile 하에선 gf+inner
  −9.0%(15.33→13.96), gf만 −7.7% — compile이 다이얼/inner 오버헤드의 대부분을 회수(−31%→−9%),
  절대 처리량도 ~2배. 실사용은 compile 필수, 그때 다이얼 비용은 한 자릿수 %.**)
- [DONE PSNR=22.639 Δ+0.551] r24_gf_inner_lr64_s96 — `bash chain_run.sh 3 r24_gf_inner_lr64_s96 config/loop_l2x4_gf_inner_d256_p16.yaml 96 --loop_param_lr_mult 64`
  (NVS 최고 스택(s95 +0.660) seed 승격. paired 기준 r1_loop_l2x4_s96. t=30.6.)
- [DONE PSNR=22.694 Δ+0.634] r24_gf_inner_lr64_s97 — 같은 형식, seed 97, expname `r24_gf_inner_lr64_s97` (paired 기준 r1_loop_l2x4_s97)
  (**t=25.7. NVS gf+inner full 3-seed 확정: +0.660/+0.551/+0.634 → 평균 +0.615, 3/3 유의. RESULTS.md 기록됨.**)

### W9. 후속 (node1 실행 중, 2026-07-22 22:4x)
- [DONE ppl=55.81] lm_affine_innerqkv_s95 — `./run_lm_w5.sh 0 config/lm_loop_l2x4_affine_innerqkv.yaml lm_affine_innerqkv_s95 outputs_lm_affine_innerqkv.log` (**55.81 ≈ full 55.20, 3다이얼 59.14 대비 −3.3 — qkv측 2사이트가 full 이득의 ~85%. 교차태스크 미니멀 처방 성립. RESULTS.md 기록.**)
- [DONE PSNR=24.551 Δ+0.767] r25_d512_gfinner_lr128_s95 — `bash chain_run.sh 1 r25_d512_gfinner_lr128_s95 config/loop_l2x4_gf_inner_d512_p16.yaml 95 --loop_param_lr_mult 128` (풀스택 d512 스케일 확인; 비교: d512 gf lr128 s95 +0.703)
- [DONE PSNR=22.684 Δ+0.596] r24_gf_inner_qkv_lr64_s96 — `bash chain_run.sh 2 r24_gf_inner_qkv_lr64_s96 config/loop_l2x4_gf_inner_qkv_d256_p16.yaml 96 --loop_param_lr_mult 64` (미니멀 후보 gf+qkv 승격)
- [DONE PSNR=22.512 Δ+0.452] r24_gf_inner_qkv_lr64_s97 — `bash chain_run.sh 3 r24_gf_inner_qkv_lr64_s97 config/loop_l2x4_gf_inner_qkv_d256_p16.yaml 97 --loop_param_lr_mult 64`

### W10. 절대 처리량 — 알맞은 baseline 포함 벤치 (사용자 요청: 비루프 모델 대비)
- [DONE 아래 8줄] bench_baselines_d512 — `bash bench_baselines.sh <g> d512` (lact/lact_nvs에서, 단독 GPU ~1.3h.
  L8(고유깊이 iso-compute)/naive L2x4/gf/gf+inner × eager/compile 8줄 — 결과를 이 항목 DONE 노트에 그대로 기록.)
  ```
  d512_l8_eager: 4.570
  d512_naive_eager: 4.615
  d512_gf_eager: 4.008
  d512_gfinner_eager: 3.112
  d512_l8_compile: 9.587
  d512_naive_compile: 9.630
  d512_gf_compile: 9.410
  d512_gfinner_compile: 8.288
  ```
  (**해석(compile): naive L2×4 9.630 ≈ L8 9.587 → naive loop와 고유깊이 L8이 iso-compute(처리량 동일)
  임을 실측 확인. 다이얼 오버헤드: gf −2.3%, gf+inner −13.9%(8.288). d256(gf+inner −9%)보다 d512에서
  약간 큼. 요지: gf+inner는 L8과 동급 계산으로 W1의 +0.66 품질을 얻음 — 오버헤드 한 자릿수~10%대.**)
- [RUNNING node1 gpu0 2026-07-23 00:0x] bench_baselines_d256 — node1 직접 실행 중 (10런 ~1h)

### W11. ★L1×8 극한 케이스★ — 고유 layer 1개, depth 다양성 전부 다이얼 몫 (사용자 가설:
"layer=1이면 우리 장점이 더 두드러진다". iso-compute 유효깊이 8 유지. 전부 s95 1-seed 사다리 먼저)
- [RUNNING node1 gpu0 2026-07-23 00:5x] r26_loop_l1x8_s95 — `bash chain_run.sh 0 r26_loop_l1x8_s95 config/loop_l1x8_d256_p16.yaml 95` (naive L1×8 앵커. 비교: naive L2×4 22.204, L8 21.955)
- [DONE PSNR=22.492 Δ+0.484] r26_l1x8_gf_s95 — `bash chain_run.sh 1 r26_l1x8_gf_s95 config/loop_l1x8_gates_film_d256_p16.yaml 95 --loop_param_lr_mult 64` (2다이얼; 다이얼 슬롯 8개)
  (**t=26.8, paired vs naive L1×8 22.007(node1). NVS 다이얼 이득 +0.484 < L2×4 +0.569 — LM과 반대! NVS는 naive L1×8이 L2×4보다 −0.20만 나빠 다양성 손실 작음 → 복원 여지 작음. gfqkv 나오면 RESULTS 함께.**)
- [DONE PSNR=22.772 Δ+0.764] r26_l1x8_gfqkv_s95 — `bash chain_run.sh 2 r26_l1x8_gfqkv_s95 config/loop_l1x8_gf_qkv_d256_p16.yaml 95 --loop_param_lr_mult 64` (미니멀 처방: 다이얼+qkv측 inner)
  (**t=30.6. NVS L1×8 사다리: naive 22.007 → gf +0.484 → gf+qkv +0.764. 반전: 2다이얼은 L1×8에서 작지만(+0.484<L2×4 +0.569) qkv-inner가 크게 더함(+0.280 vs +0.077) → 전체 스택 L1×8 +0.764 > L2×4 +0.646. inner-affine이 극한 tying에서 빛남(LM과 일관). RESULTS.md 기록됨. W11 완료.**)
- [DONE ppl=86.73] lm_l1x8_naive_s95 — `./run_lm_w5.sh <g> config/lm_loop_l1x8.yaml lm_l1x8_naive_s95 outputs_lm_l1x8_naive.log` (LM 앵커. 비교: L2×4 naive 74.45)
  (**ppl 86.73 (14.57M) vs L2×4 naive 74.45 = +12.28 악화 — 고유 layer 1개의 다양성 손실 큼. affine/innerqkv가 이걸 얼마나 회복하는지가 L1×8 가설 검증.**)
- [DONE ppl=64.02] lm_l1x8_affine_s95 — `./run_lm_w5.sh 3 config/lm_loop_l1x8_affine.yaml lm_l1x8_affine_s95 outputs_lm_l1x8_affine.log` (3다이얼)
- [DONE ppl=61.43] lm_l1x8_innerqkv_s95 — `./run_lm_w5.sh 4 config/lm_loop_l1x8_innerqkv.yaml lm_l1x8_innerqkv_s95 outputs_lm_l1x8_innerqkv.log` (3다이얼+qkv측)
  (**L1×8 LM 사다리 완성: naive 86.73 → 3다이얼 64.02(회복 −22.71) → +qkv-inner 61.43(회복 −25.30).
  다이얼 회복이 L1×8에서 L2×4(−19.25/−18.64)보다 큼 → "다이얼 = 잃어버린 layer-다양성 압축 복원" 지지.
  RESULTS.md 기록됨. NVS L1×8(r26_gf/gfqkv) 진행 중.**)
- 판정 프레임: 각 태스크에서 (다이얼 이득 | L1×8) vs (다이얼 이득 | L2×4) — L1×8에서 이득이 더 크면
  "다이얼 = 잃어버린 layer-다양성의 파라미터 압축 복원" 서사 강화. naive L1×8이 L2×4보다 얼마나
  낮은지(다양성 손실 크기)도 그 자체로 데이터.

### W12. LM 미니멀 처방 2-seed
- [DONE ppl=55.79] lm_affine_innerqkv_s96 — `./run_lm_w5.sh 5 config/lm_loop_l2x4_affine_innerqkv.yaml lm_affine_innerqkv_s96 outputs_lm_affine_innerqkv_s96.log --seed 96` (s95 55.81의 재현)
  (**ppl 55.79 (s95 55.81) → 평균 55.80, ±0.02 매우 안정. LM 미니멀 처방(다이얼+qkv-inner) 2-seed 확정. RESULTS.md 기록됨.**)

### W13. ★대형 모델 절대 처리량★ (사용자 요청 — d256 벤치는 3.9M로 너무 작음)
- [DONE 아래 5줄] bench_llm_throughput — lact/lact_llm_loop에서:
  `CUDA_VISIBLE_DEVICES=<g> /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/.venv_llm/bin/python bench_throughput.py > bench_llm.log 2>&1`
  (d768/seq4096/bs8 실런 설정, 합성 토큰이라 HF 스트리밍 불필요. 12L고유(iso-compute 준거)/3L/naive
  3L×4/3다이얼/다이얼+inner 5종 순차, ~20분. 결과 5줄을 이 항목 DONE 노트에 기록.
  TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR는 run_loop.sh와 동일하게 export 후 실행.)
  (node2: run_bench_throughput.sh 래퍼(캐시 export)로 실행. eager, 30 timed steps:)
  ```
  orig_12L (iso-compute unique): 139.6M params, 160.6 ms/step, 203,990 tok/s
  l3 (1/4 compute):               71.8M params,  67.3 ms/step, 486,931 tok/s
  naive 3Lx4:                     71.8M params, 159.5 ms/step, 205,390 tok/s
  3Lx4 + 3dials:                  71.8M params, 179.4 ms/step, 182,674 tok/s
  3Lx4 + dials+inner:             72.0M params, 199.6 ms/step, 164,203 tok/s
  ```
  (**해석: naive 3L×4 205,390 ≈ orig_12L 203,990 tok/s → loop naive와 12L 고유깊이가 iso-compute
  (처리량 동일) 대형서도 확인. 다이얼 오버헤드(eager): 3다이얼 −11.1%, dials+inner −20.1%. 파라미터는
  다이얼 +0.05M/inner +0.2M로 사실상 무료. compile 시 오버헤드 대부분 회수됨(W8·W10 참조).**)
- [DONE — RESULTS.md W10-d512 절 참조. 핵심: compile서 gf −1.6%(공짜), gf+inner −12%] bench_baselines_d512 — `bash bench_baselines.sh 1 d512` (node1이 직접 실행; gpu0 훈련 1개 동시라 절대값에 소폭 경합 있음, 세션 내 상대비교 유효)

### W14. d512 풀스택 seed 승격 (+0.767 s95의 확정)
- [RUNNING node1 gpu2 2026-07-23 03:0x] r25_d512_gfinner_lr128_s96 — `bash chain_run.sh 2 r25_d512_gfinner_lr128_s96 config/loop_l2x4_gf_inner_d512_p16.yaml 96 --loop_param_lr_mult 128` (paired 기준 r22_d512_naive_s96)
- [RUNNING node1 gpu3 2026-07-23 03:0x] r25_d512_gfinner_lr128_s97 — 같은 형식, seed 97 (paired 기준 r22_d512_naive_s97)
- 큐 노트: qkv-only 가산이 NVS 3-seed에서 무너짐(+0.019, s97 음성) → NVS 대표 구성은 full inner로 확정,
  그 d512 스케일 검증이 본 웨이브. RESULTS.md W9 절 참조.

### W13b. LM compile 회수 검증 (사용자 요청 — "compile 이식 여지" 실측)
- [DONE] bench_llm_throughput+COMPILE — 3다이얼 −11.1%→**−3.7%**, dials+inner −20.1%→**−7.2%** (compile, naive 대비). 회수 검증 완료, RESULTS.md W13b 절.

### W15. L1×8 3-seed 승격 (s95 발견: inner 가산 3.6× 증폭 + 2.47M 극한압축 후보)
- [RUNNING node1 gpu0 2026-07-23] r26_loop_l1x8_s96 — `bash chain_run.sh 0 r26_loop_l1x8_s96 config/loop_l1x8_d256_p16.yaml 96` (앵커)
- [RUNNING node1 gpu1 2026-07-23] r26_loop_l1x8_s97 — 같은 형식, seed 97 (앵커)
- [RUNNING node1 gpu2 2026-07-23] r26_l1x8_gfqkv_s96 — `bash chain_run.sh 2 r26_l1x8_gfqkv_s96 config/loop_l1x8_gf_qkv_d256_p16.yaml 96 --loop_param_lr_mult 64`
- [RUNNING node1 gpu3 2026-07-23] r26_l1x8_gfqkv_s97 — 같은 형식, seed 97
- [DONE PSNR=22.604 Δ+0.451] r26_l1x8_gf_s96 — `bash chain_run.sh 1 r26_l1x8_gf_s96 config/loop_l1x8_gates_film_d256_p16.yaml 96 --loop_param_lr_mult 64` (gf-inner 분해 유지용)
  (**t=22.3. ★L1×8 gf 3-seed 확정: +0.484/+0.451/+0.608 → 평균 +0.514, 3/3 유의.★ inner 가산(gfqkv−gf) 3-seed는 node1의 gfqkv s96/s97 필요(s95는 +0.280). RESULTS.md 기록됨. W15 node2 몫 완료.**)
- [DONE PSNR=22.595 Δ+0.608] r26_l1x8_gf_s97 — 같은 형식, seed 97 (gpu2)
  (**t=30.5, paired vs naive L1×8 s97 21.986. L1×8 gf 3-seed: s95 +0.484 / s97 +0.608 / s96 대기.**)
- 판정: seed별 paired (naive_s{n} → gf_s{n} → gfqkv_s{n}). 관심량 = inner 가산(qkv−gf)의 3-seed 안정성.

## 완료 로그 (node2가 갱신)
- 2026-07-22 13:18 node2 시작 보고: B200×6 확인(전부 유휴), setup_node.sh 완료 상태, /tmp/re10k reshard 진행 중(~3분). W1 6런 GPU 0-5 claim, reshard 완료 즉시 투입.
- 2026-07-23 03:3x node2 큐 소진 보고: **W1~W13 + W7 전부 완료·기록·커밋.** 하이라이트 — W7(최우선
  large-LM) ppl 20.85, inner-affine 대규모서 무효(스케일 의존); W11 L1×8 양태스크 수렴(다이얼/inner =
  layer-다양성 복원); W2 adaln optzone +0.620(형태 아닌 처방). **자원 변화**: 현재 node2 GPU 1-4를
  외부 프로젝트(`gfm` env, loop_TTT 무관)가 점유 → node2 가용 GPU는 0,5 2개뿐. claim 가능 PENDING
  없음(W14는 node1). 신규 PENDING 감시하며 대기 중.
- 2026-07-23 03:4x **node2 가용 GPU 0개**: 외부 `gfm` 잡이 6 GPU 전부로 확장(방금 비운 gpu0/5 포함).
  **W15의 node2 몫(r26_l1x8_gf_s96/s97)을 claim 못 함** — GPU 해제 시 즉시 착수 예정. node1이 더
  급하면 node1 GPU로 가져가도 무방(그 경우 이 두 줄 RUNNING 표기 갱신 요망). GPU-해제 감시로 대기.
