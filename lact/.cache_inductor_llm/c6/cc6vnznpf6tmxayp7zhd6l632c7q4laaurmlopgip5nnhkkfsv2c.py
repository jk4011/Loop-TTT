
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 23593600}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp60 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 4.0848
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp12 = tl.load(in_ptr3 + (192*(((r0_2 + 7373*x0) % 192)) + 36864*x1 + ((((r0_2 + 7373*x0) // 192) % 192))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = -tmp14
        tmp16 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.load(in_ptr5 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 0.0
        tmp20 = tmp19 * tmp18
        tmp21 = tmp17 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.load(in_ptr6 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = 1e-07
        tmp26 = tmp24 + tmp25
        tmp27 = (tmp23 / tmp26)
        tmp28 = (tmp27 / tmp26)
        tmp29 = tmp15 * tmp28
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
        tmp35 = tl.load(in_ptr7 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tl.load(in_ptr8 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tmp37 * tmp6
        tmp39 = tmp36 + tmp38
        tmp40 = tl.load(in_ptr9 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp39 + tmp41
        tmp43 = tl.load(in_ptr10 + (192*(((r0_2 + 7373*x0) % 192)) + 36864*x1 + ((((r0_2 + 7373*x0) // 192) % 192))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp42 + tmp44
        tmp46 = -tmp45
        tmp47 = tl.load(in_ptr11 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp48 + tmp20
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tl.load(in_ptr12 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tmp52 + tmp25
        tmp54 = (tmp51 / tmp53)
        tmp55 = (tmp54 / tmp53)
        tmp56 = tmp46 * tmp55
        tmp57 = tl.full(tmp56.shape, 0, tmp56.dtype)
        tmp58 = tl.where(tmp2, tmp56, tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, R0_BLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(r0_mask & xmask, tmp61, _tmp60)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp33, xmask)
    tl.store(out_ptr1 + (x3), tmp60, xmask)
