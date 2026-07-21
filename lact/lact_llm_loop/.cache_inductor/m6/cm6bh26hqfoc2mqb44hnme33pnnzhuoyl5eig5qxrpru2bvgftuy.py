
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'out_ptr7': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 56623104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp24 = tl.load(in_ptr6 + (x2), None).to(tl.float32)
    tmp31 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 0.0
    tmp8 = tmp7 * tmp6
    tmp9 = tmp3 + tmp8
    tmp11 = (tmp10 / tmp5)
    tmp12 = tmp9 * tmp11
    tmp13 = tmp1 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = 1e-07
    tmp19 = tmp17 + tmp18
    tmp20 = (tmp15 / tmp19)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp8
    tmp27 = tmp26 * tmp11
    tmp28 = tmp23 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp32 = libdevice.sqrt(tmp31)
    tmp33 = tmp32 + tmp18
    tmp34 = (tmp30 / tmp33)
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
    tl.store(out_ptr3 + (x2), tmp21, None)
    tl.store(out_ptr4 + (x2), tmp34, None)
    tl.store(out_ptr5 + (x2), tmp35, None)
    tl.store(out_ptr6 + (x2), tmp35, None)
    tl.store(out_ptr7 + (x2), tmp35, None)
