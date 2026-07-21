
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr6': '*bf16', 'out_ptr7': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 3, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 51904512}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6144
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp6 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp37 = tl.load(in_ptr5 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = 3.7418
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 2.8769
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = 2.8366
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp0 + tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(r0_mask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = (tmp22 / tmp30)
    tmp32 = libdevice.sqrt(tmp5)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp35 * tmp17
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 + tmp38
    tmp40 = tmp22 + tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
    tmp44 = tl.where(r0_mask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None].to(tl.float32)
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tmp46 + tmp29
    tmp48 = (tmp40 / tmp47)
    tmp49 = tmp48 * tmp32
    tmp50 = tmp49.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp22, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp34, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp34, r0_mask & xmask)
    tl.store(out_ptr6 + (r0_1 + 192*x0), tmp50, r0_mask & xmask)
    tl.store(out_ptr7 + (r0_1 + 192*x0), tmp50, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
