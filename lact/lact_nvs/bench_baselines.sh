#!/usr/bin/env bash
# Absolute-throughput bench vs PROPER baselines (non-loop models included),
# all on ONE GPU in one session for self-consistent numbers.
#   d256 set: L2(iso-param, 1/4 compute) / L8(iso-compute unique depth) /
#             naive L2x4 loop / gf / gf+inner   x eager/compile
#   d512 set: L8 / naive L2x4 / gf / gf+inner   x eager/compile
# Usage: bash bench_baselines.sh <gpu> <d256|d512>
set -u
G=$1; SET=${2:-d256}
cd "$(dirname "$0")"
PY_ENV=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

run_one () {  # name config extra
  local name=$1 cfg=$2 extra=$3
  local exp="bench_$name"
  rm -rf "outputs/$exp"; mkdir -p "outputs/$exp"
  CUDA_VISIBLE_DEVICES=$G $PY_ENV/torchrun \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "$cfg" \
    --data_path /tmp/re10k/train_index.json --dataset re10k --scene_pose_normalize \
    --expname "$exp" --steps 600 --warmup 100 --lr 1e-4 --lpips_start 5000 --seed 95 \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 --save_every 100000 --log_every 50 \
    --actckpt --loop_param_lr_mult 64 $extra > "outputs/$exp/train.log" 2>&1
  local v=$(grep -o "it/s: [0-9.]*" "outputs/$exp/train.log" | tail -6 | awk '{s+=$2;n++} END {if(n) printf "%.3f", s/n}')
  echo "$name: ${v:-FAILED}"
  rm -rf "outputs/$exp"
}

echo "=== BASELINE BENCH set=$SET gpu$G (600 steps, steady=last 6 it/s prints) ==="
if [ "$SET" = "d256" ]; then
  MODELS="l2:lact_l2_d256_p16.yaml l8:lact_l8_d256_p16.yaml naive:loop_l2x4_d256_p16.yaml gf:loop_l2x4_gates_film_d256_p16.yaml gfinner:loop_l2x4_gf_inner_d256_p16.yaml"
else
  MODELS="l8:lact_l8_d512_p16.yaml naive:loop_l2x4_d512_p16.yaml gf:loop_l2x4_gates_film_d512_p16.yaml gfinner:loop_l2x4_gf_inner_d512_p16.yaml"
fi
for mode in eager compile; do
  [ "$mode" = "compile" ] && CFLAG="--compile" || CFLAG=""
  for m in $MODELS; do
    name="${m%%:*}"; cfg="config/${m#*:}"
    run_one "${SET}_${name}_${mode}" "$cfg" "$CFLAG"
  done
done
echo "=== BASELINE BENCH DONE ==="
