#!/usr/bin/env bash
# node2 병렬 실험 배치 (B200 x6 노드용) — Slurm GUI에서 이 스크립트를 실행으로 제출.
# 전제: lustre 공유(코드/데이터/env 동일). 이 스크립트가 하는 일:
#   1) 이 노드의 /tmp에 RE10K reshard (노드-로컬, ~3분)
#   2) d512 스케일업 3-seed 완성: {naive, gf@lr64, gf@lr16} x {s96, s97} = 6런 (GPU 0-5)
#   3) 전 런 완료까지 대기 (배치 잡 유지)
# 주의: node1과 실험 이름이 겹치지 않도록 여기서는 s96/s97만 돌린다.
set -uo pipefail
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python

echo "[node2] $(hostname) start $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

# 1) RE10K reshard (이미 있으면 스킵)
if [ ! -f /tmp/re10k/train_index.json ]; then
  echo "[node2] resharding RE10K to /tmp ..."
  $PY data_preprocess/reshard_re10k.py \
    --src /NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k/train \
    --odir /tmp/re10k/train --index /tmp/re10k/train_index.json --workers 16 &
  $PY data_preprocess/reshard_re10k.py \
    --src /NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k/test \
    --odir /tmp/re10k/test --index /tmp/re10k/test_index.json --workers 16 &
  wait
fi
echo "[node2] data ready: $(ls /tmp/re10k/*.json | wc -l) index files"

# 2) 6 실험 병렬 (GPU 0-5, 실험명은 node1과 disjoint)
bash chain_run.sh 0 r22_d512_naive_s96   config/loop_l2x4_d512_p16.yaml 96 &
sleep 3
bash chain_run.sh 1 r22_d512_naive_s97   config/loop_l2x4_d512_p16.yaml 97 &
sleep 3
bash chain_run.sh 2 r22_d512_gf_lr64_s96 config/loop_l2x4_gates_film_d512_p16.yaml 96 --loop_param_lr_mult 64 &
sleep 3
bash chain_run.sh 3 r22_d512_gf_lr64_s97 config/loop_l2x4_gates_film_d512_p16.yaml 97 --loop_param_lr_mult 64 &
sleep 3
bash chain_run.sh 4 r22_d512_gf_lr16_s96 config/loop_l2x4_gates_film_d512_p16.yaml 96 --loop_param_lr_mult 16 &
sleep 3
bash chain_run.sh 5 r22_d512_gf_lr16_s97 config/loop_l2x4_gates_film_d512_p16.yaml 97 --loop_param_lr_mult 16 &

# 3) 완료 대기 (chain_run들이 전부 끝날 때까지 배치 잡 유지)
wait
echo "[node2] ALL DONE $(date)"
