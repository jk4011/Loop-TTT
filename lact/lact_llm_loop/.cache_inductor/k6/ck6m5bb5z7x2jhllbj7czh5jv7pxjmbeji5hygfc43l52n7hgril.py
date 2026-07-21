
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 9437440}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 1024.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 0.0
        tmp11 = tmp10 * tmp9
        tmp12 = tmp6 + tmp11
        tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = (tmp13 / tmp8)
        tmp15 = tmp12 * tmp14
        tmp16 = tmp4 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(r0_mask & xmask, tmp24, _tmp23)
        tmp25 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tl.load(in_ptr5 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 + tmp11
        tmp30 = tmp29 * tmp14
        tmp31 = tmp26 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 * tmp33
        tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
        tmp36 = tl.where(tmp2, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask & xmask, tmp39, _tmp38)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tl.store(out_ptr1 + (x3), tmp38, xmask)
