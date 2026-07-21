
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 25952256}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6144
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 192)
    x1 = xindex // 192
    x3 = xindex
    tmp6 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 192*r0_2 + 36864*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp7 = 1e-05
        tmp8 = tmp6 + tmp7
        tmp9 = (tmp5 / tmp8)
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp15 = tmp4 * tmp14
        tmp16 = -tmp15
        tmp17 = (tmp9 / tmp8)
        tmp18 = tmp16 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp22 = tl.load(in_out_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr0 + (x0 + 192*r0_2 + 36864*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27 * tmp14
        tmp29 = 1e-05
        tmp30 = tmp6 + tmp29
        tmp31 = (tmp28 / tmp30)
        tmp32 = tmp22 + tmp31
        tmp33 = 0.0
        tmp34 = tmp6 == tmp33
        tmp36 = (tmp35 / tmp6)
        tmp37 = tl.where(tmp34, tmp33, tmp36)
        tmp38 = tmp20 * tmp37
        tmp39 = tmp32 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_2 + 192*x3), tmp39, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_2 + 192*x3), tmp40, r0_mask & xmask)
