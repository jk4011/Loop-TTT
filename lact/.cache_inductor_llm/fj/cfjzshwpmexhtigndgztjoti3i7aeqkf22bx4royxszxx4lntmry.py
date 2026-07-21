
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 75497472, 'x': 125829120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 4096)
    x2 = xindex
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 3072, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-393216) + x2 + 192*y0 + 196608*y1), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.load(in_ptr2 + ((-393216) + x2 + 192*y0 + 196608*y1), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 + tmp13
    tmp15 = tl.load(in_ptr4 + ((-2048) + y0 + 1024*x2 + 196608*y1), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tl.load(in_ptr5 + ((-2048) + y0 + 1024*x2 + 196608*y1), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = 0.0
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tl.full([1, 1], 1024, tl.int64)
    tmp26 = tmp0 >= tmp25
    tmp27 = tmp0 < tmp1
    tmp28 = tmp26 & tmp27
    tmp29 = tl.load(in_ptr6 + ((-196608) + x2 + 192*y0 + 196608*y1), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr7 + ((-196608) + x2 + 192*y0 + 196608*y1), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.load(in_ptr3 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 + tmp36
    tmp38 = tl.load(in_ptr8 + ((-1024) + y0 + 1024*x2 + 196608*y1), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 + tmp39
    tmp41 = tl.load(in_ptr9 + ((-1024) + y0 + 1024*x2 + 196608*y1), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tl.where(tmp28, tmp45, tmp23)
    tmp47 = tmp24 + tmp46
    tmp48 = tmp0 < tmp25
    tmp49 = tl.load(in_ptr10 + (x2 + 192*y0 + 196608*y1), tmp48 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.load(in_ptr1 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 * tmp51
    tmp53 = tl.load(in_ptr11 + (x2 + 192*y0 + 196608*y1), tmp48 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.load(in_ptr3 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp52 + tmp56
    tmp58 = tl.load(in_ptr12 + (y0 + 1024*x2 + 196608*y1), tmp48 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp57 + tmp59
    tmp61 = tl.load(in_ptr13 + (y0 + 1024*x2 + 196608*y1), tmp48 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp60 + tmp62
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp48, tmp63, tmp64)
    tmp66 = tl.where(tmp48, tmp65, tmp23)
    tmp67 = tmp47 + tmp66
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp67, xmask)
