#!/usr/bin/env bash
# W5 LM inner-affine launcher: train_lm.py with lustre compile caches (/tmp is noexec).
# Usage: ./run_lm_w5.sh <gpu> <config_yaml> <expname> <logfile> [extra args...]
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
GPU=$1; CFG=$2; EXP=$3; LOG=$4; shift 4
CUDA_VISIBLE_DEVICES=$GPU "$PY" train_lm.py \
  --config "$CFG" --expname "$EXP" \
  --data_dir /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/wikitext103_gpt2 \
  --steps 12000 --bs 16 --seed 95 --val_every 3000 --loop_param_lr_mult 64 \
  "$@" > "$LOG" 2>&1
