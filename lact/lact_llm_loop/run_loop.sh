#!/usr/bin/env bash
# Usage: ./run_loop.sh <gpu> <out_name> [extra train_small.py args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/.venv_llm/bin/python
export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH=/usr/local/cuda/bin:$PATH
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_llm"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_llm"
mkdir -p "outputs/$2" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
CUDA_VISIBLE_DEVICES=$1 "$PYTHON" train_small.py --out_dir "outputs/$2" "${@:3}" > "outputs/$2/train.log" 2>&1
