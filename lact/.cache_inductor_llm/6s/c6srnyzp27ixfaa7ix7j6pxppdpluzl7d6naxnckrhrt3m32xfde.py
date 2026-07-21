
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*bf16', 'in_ptr15': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_zeros_like_33', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4718592, 'x': 51904512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_zeros_like_33(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_out_ptr0 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (y0 + 192*x2 + 36864*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr6 + (y1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (y1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_out_ptr1 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp40 = tl.load(in_ptr9 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp46 = tl.load(in_ptr11 + (y0 + 192*x2 + 36864*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr12 + (y1), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (y1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp65 = tl.load(in_ptr15 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 4.0848
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = 1e-07
    tmp14 = tmp12 + tmp13
    tmp15 = (tmp11 / tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp18 = 0.0
    tmp19 = tmp12 == tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp18 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = (tmp26 / tmp12)
    tmp28 = tl.where(tmp19, tmp18, tmp27)
    tmp29 = tmp17 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp16 + tmp30
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31.to(tl.float32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tmp40 * tmp3
    tmp42 = tmp39 + tmp41
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 + tmp44
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 + tmp47
    tmp50 = tmp49 + tmp13
    tmp51 = (tmp48 / tmp50)
    tmp52 = tmp51.to(tl.float32)
    tmp54 = tmp49 == tmp18
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 + tmp23
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = (tmp59 / tmp49)
    tmp61 = tl.where(tmp54, tmp18, tmp60)
    tmp62 = tmp53 * tmp61
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp52 + tmp63
    tmp66 = tmp65 * tmp33
    tmp67 = tmp64.to(tl.float32)
    tmp68 = tmp66 + tmp67
    tmp69 = tmp68.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp31, xmask)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp37, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + 192*y3), tmp64, xmask)
    tl.store(out_ptr1 + (x2 + 192*y3), tmp69, xmask)
