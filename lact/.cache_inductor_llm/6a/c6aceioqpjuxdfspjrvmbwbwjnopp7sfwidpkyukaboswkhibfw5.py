
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 294912, 'r0_': 84934656}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 18432
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = (tmp5 / tmp8)
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tmp4 * tmp15
    tmp17 = -tmp16
    tmp18 = (tmp9 / tmp8)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp25 = (tmp16 / tmp8)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp6 == tmp27
    tmp29 = (tmp5 / tmp6)
    tmp30 = tl.where(tmp28, tmp27, tmp29)
    tmp31 = tmp23 * tmp30
    tmp32 = tmp26 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp32, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp33, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
