
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331648, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp23 = tl.load(in_ptr3 + (3072 + 3*y0 + 12288*y1), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 1024*(x2) + 98304*y1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1, 1], 192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp8, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp25, xmask)
