#!/usr/bin/env bash
# W13 large-LM throughput bench: bench_throughput.py with lustre compile caches (/tmp noexec).
# Usage: ./run_bench_throughput.sh <gpu>
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_llm"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_llm"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/.venv_llm/bin/python
CUDA_VISIBLE_DEVICES=$1 "$PY" bench_throughput.py > bench_llm.log 2>&1
