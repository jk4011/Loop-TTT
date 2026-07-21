
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 45613056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (1024 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (1024 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = -tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 - tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 2048, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr2 + (1024 + 4096*x1 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp23 = -tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (1024 + 4096*x1 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp24 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp17, tmp29, tmp30)
    tmp32 = tl.where(tmp4, tmp16, tmp31)
    tl.store(out_ptr0 + (x3), tmp32, None)
