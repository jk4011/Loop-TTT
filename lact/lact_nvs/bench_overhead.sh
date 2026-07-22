#!/usr/bin/env bash
# Clean-GPU wall-clock benchmark: naive / gf / gf+inner, eager vs torch.compile.
# Uses launch_exp.sh protocol but 600 steps, log_every 50; steady it/s = last 6 prints.
# Requires /tmp/re10k reshard. Usage: bash bench_overhead.sh <gpu>
set -u
G=$1
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
  rm -f "outputs/$exp"/model_*.pth 2>/dev/null
}

echo "=== OVERHEAD BENCH (gpu$G, 600 steps, steady=last 6 prints of it/s) ==="
run_one naive_eager     config/loop_l2x4_d256_p16.yaml            ""
run_one gf_eager        config/loop_l2x4_gates_film_d256_p16.yaml ""
run_one gfinner_eager   config/loop_l2x4_gf_inner_d256_p16.yaml   ""
run_one naive_compile   config/loop_l2x4_d256_p16.yaml            "--compile"
run_one gf_compile      config/loop_l2x4_gates_film_d256_p16.yaml "--compile"
run_one gfinner_compile config/loop_l2x4_gf_inner_d256_p16.yaml   "--compile"
echo "=== BENCH DONE ==="
