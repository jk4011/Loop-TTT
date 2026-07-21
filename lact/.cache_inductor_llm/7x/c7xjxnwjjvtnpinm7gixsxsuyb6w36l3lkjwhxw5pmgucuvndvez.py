
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 8257536}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 2.8769
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = 2.8366
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp0 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp12 / tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp12, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp24, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp24, r0_mask & xmask)
