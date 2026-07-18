#!/bin/bash
# Usage: chain_run.sh <gpu> <expname> <config> [seed] [extra train.py args...]
# Full experiment: 30k-iter training (launch_exp.sh protocol) then standard eval
# (256 held-out RE10K scenes, 8 input / 4 target views) -> outputs/<exp>/eval.json
# NOTE: seed is required when passing extra args.
set -u
GPU=$1
EXP=$2
CONFIG=$3
SEED=${4:-95}

cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python

bash launch_exp.sh $GPU $EXP $CONFIG $SEED "${@:5}"

CKPT=outputs/$EXP/model_0030000.pth
if [ ! -f "$CKPT" ]; then
  echo "CHAIN_FAIL $EXP: no final checkpoint" >> outputs/exp_status.log
  exit 1
fi

REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load $CKPT --config $CONFIG \
  --out outputs/$EXP/eval.json > outputs/$EXP/eval.log 2>&1
echo "CHAIN_DONE $EXP $(grep -o 'PSNR: [0-9.]*' outputs/$EXP/eval.log | head -1)" >> outputs/exp_status.log
