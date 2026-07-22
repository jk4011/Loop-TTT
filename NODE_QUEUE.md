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
- [RUNNING node2 gpu0 2026-07-22 15:38] r23_dvlt_oz_s95 — `bash chain_run.sh 0 r23_dvlt_oz_s95 config/loop_l2x4_dvlt_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (LOOP_PARAM_KEYS에 dvlt_mlp 추가 완료: commit e1f064d. dvlt_temb은 buffer라 제외.)
- [RUNNING node2 gpu1 2026-07-22 15:38] r23_adaln_oz_s95 — `bash chain_run.sh 1 r23_adaln_oz_s95 config/loop_l2x4_adaln_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (adaln_emb/adaln_mlp 키 추가 완료: commit e1f064d.)
- [RUNNING node1 gpu2 2026-07-22 15:46] r23_layerscale_oz_s95 — `bash chain_run.sh <g> r23_layerscale_oz_s95 config/loop_l2x4_layerscale_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (lscale 키 추가 완료: commit e1f064d. W1 gf 런 종료 GPU에 투입 예정.)
- 기록: 표준-옵티마이저 버전(r23_*, node1에서 실행 중)과 나란히 표로 — "형태 vs 옵티마이저" 분해가 목적.

### W3. LM 백로그 (여유 GPU 시)
- [PENDING] perlayer 3B pair는 보류(0.5B에서 이미 결론). 대신:
- [RUNNING node2 gpu2 2026-07-22 15:52] lm loop_ours_3b_lr8 — `./run_loop.sh 2 loop_ours_3b_lr8 --num_hidden_layers 3 --n_loops 4 --loop_dials true --loop_param_lr_mult 8 --bs 8 --token_budget 3000000000` (lr 정점 미세화: 8 vs 16)
  (1차 시도 HF 429 rate-limit로 실패 — fineweb-edu 스트리밍이 IP-공유 창 초과. rate-limit 창 리셋 후 지연 재시도 중.)


### W4. inner-affine "사이사이" (사용자 아이디어: 공유 행렬 사이 전 이음새에 loop별 affine)
- [RUNNING node2 gpu3 2026-07-22 16:44] r24_gf_inner_lr64_s95 — `bash chain_run.sh 3 r24_gf_inner_lr64_s95 config/loop_l2x4_gf_inner_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (gf 2다이얼 + attn/TTT qkv직후·c_proj직전 + MLP 은닉 affine. 비교: gf@64 +0.569)
- [RUNNING node2 gpu4 2026-07-22 16:44] r24_inner_only_lr64_s95 — `bash chain_run.sh 4 r24_inner_only_lr64_s95 config/loop_l2x4_inner_only_d256_p16.yaml 95 --loop_param_lr_mult 64`
  (config 존재 확인됨. gf 없이 inner만 — 분리측정)

## 완료 로그 (node2가 갱신)
- 2026-07-22 13:18 node2 시작 보고: B200×6 확인(전부 유휴), setup_node.sh 완료 상태, /tmp/re10k reshard 진행 중(~3분). W1 6런 GPU 0-5 claim, reshard 완료 즉시 투입.
