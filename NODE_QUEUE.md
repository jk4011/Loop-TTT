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
- [RUNNING node2 gpu2 2026-07-22 15:52] lm loop_ours_3b_lr8 — `./run_loop.sh 2 loop_ours_3b_lr8 --num_hidden_layers 3 --n_loops 4 --loop_dials true --loop_param_lr_mult 8 --bs 8 --token_budget 3000000000` (lr 정점 미세화: 8 vs 16)
  (1차 시도 HF 429 rate-limit로 실패 — fineweb-edu 스트리밍이 IP-공유 창 초과. rate-limit 창 리셋 후 지연 재시도 중.)


### W4. inner-affine "사이사이" (사용자 아이디어: 공유 행렬 사이 전 이음새에 loop별 affine)
- [DONE PSNR=22.864 Δ+0.660] r24_gf_inner_lr64_s95 — `bash chain_run.sh 3 r24_gf_inner_lr64_s95 config/loop_l2x4_gf_inner_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (t=32.0. **gf+inner full +0.660 > gf 2다이얼 +0.569 (inner가 +0.091 가산), > inner_only +0.320.** 스택 최상, RESULTS.md 기록됨.)
- [DONE PSNR=22.523 Δ+0.320] r24_inner_only_lr64_s95 — `bash chain_run.sh 4 r24_inner_only_lr64_s95 config/loop_l2x4_inner_only_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (t=27.7. NVS inner-affine 단독 +0.320 — 양성이나 gf 2다이얼 +0.569보다 약함(LM에선 inner_only가 3다이얼 격파 → 태스크 의존). gf_inner full 결과 나오면 RESULTS 함께 기록.)

#### W4b. NVS inner 사이트 분해 (LM W5 결과가 attn/MLP-이음새 우세를 시사 → NVS에서 사이트 귀속)
- [RUNNING node1 gpu0 2026-07-22 18:20] r24_gf_inner_ttt_lr64_s95 — `bash chain_run.sh 0 r24_gf_inner_ttt_lr64_s95 config/loop_l2x4_gf_inner_ttt_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + TTT 이음새만)
- [RUNNING node1 gpu1 2026-07-22 18:20] r24_gf_inner_attn_lr64_s95 — `bash chain_run.sh 1 r24_gf_inner_attn_lr64_s95 config/loop_l2x4_gf_inner_attn_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + attn 이음새만)
- [RUNNING node1 gpu2 2026-07-22 18:20] r24_gf_inner_mlp_lr64_s95 — `bash chain_run.sh 2 r24_gf_inner_mlp_lr64_s95 config/loop_l2x4_gf_inner_mlp_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + MLP 은닉만)
- [RUNNING node1 gpu3 2026-07-22 18:20] r24_gf_inner_qkv_lr64_s95 — `bash chain_run.sh 3 r24_gf_inner_qkv_lr64_s95 config/loop_l2x4_gf_inner_qkv_d256_p16.yaml 95 --loop_param_lr_mult 64` (gf + qkv측만, attn+TTT)
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
- [FAILED 코드버그: FlashAttention fp32] lm loop_oursinner_3b_lr16 — `./run_loop.sh <g> loop_oursinner_3b_lr16 --num_hidden_layers 3 --n_loops 4 --loop_dials true --loop_inner full --loop_param_lr_mult 16 --bs 8 --token_budget 3000000000`
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
- [RUNNING node2 gpu3 2026-07-22 18:42] r22_d512_gf_lr128_s96 — `bash chain_run.sh 3 r22_d512_gf_lr128_s96 config/loop_l2x4_gates_film_d512_p16.yaml 96 --loop_param_lr_mult 128` (d512 lr 정점 128 vs 64 판별)
- [RUNNING node2 gpu5 2026-07-22 18:52] r22_d512_gf_lr128_s97 — 같은 형식, seed 97, expname `r22_d512_gf_lr128_s97`
- [DONE PSNR=22.443 Δ+0.383] r23_adaln_oz_s97 — `bash chain_run.sh 1 r23_adaln_oz_s97 config/loop_l2x4_adaln_d256_p16.yaml 97 --loop_param_lr_mult 64`
  (**t=19.6. adaln optzone 3-seed 확정: +0.620/+0.598/+0.383 → 평균 +0.534, 3/3 유의. RESULTS.md 기록됨.**)

## 완료 로그 (node2가 갱신)
- 2026-07-22 13:18 node2 시작 보고: B200×6 확인(전부 유휴), setup_node.sh 완료 상태, /tmp/re10k reshard 진행 중(~3분). W1 6런 GPU 0-5 claim, reshard 완료 즉시 투입.
