
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64487424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (3072 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (3072 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 - tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 2048, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + (3072 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.sigmoid(tmp42)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tmp45 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr2 + (3072 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp40 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x3), tmp54, None)
