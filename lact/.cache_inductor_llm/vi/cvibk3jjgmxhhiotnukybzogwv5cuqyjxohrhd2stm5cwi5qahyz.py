
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*bf16', 'in_ptr9': '*fp32', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 66060288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_24(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex // 36864
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 192)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x3), None)
    tmp34 = tl.load(in_out_ptr1 + (x3), None)
    tmp36 = tl.load(in_ptr8 + (x3), None).to(tl.float32)
    tmp38 = tl.load(in_ptr9 + (x3), None)
    tmp41 = tl.load(in_ptr10 + (x3), None).to(tl.float32)
    tmp44 = tl.load(in_ptr11 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tl.load(in_ptr12 + (x2), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr13 + (x2), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr14 + (x3), None)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp6 = 4.0848
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp16 = 1e-07
    tmp17 = tmp15 + tmp16
    tmp18 = (tmp14 / tmp17)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 0.0
    tmp22 = tmp15 == tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = (tmp25 / tmp15)
    tmp27 = tl.where(tmp22, tmp21, tmp26)
    tmp28 = tmp20 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp19 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp2 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp34 * tmp1
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp38 * tmp6
    tmp40 = tmp37 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp43 + tmp45
    tmp48 = tmp47 + tmp16
    tmp49 = (tmp46 / tmp48)
    tmp50 = tmp49.to(tl.float32)
    tmp52 = tmp47 == tmp21
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp54.to(tl.float32)
    tmp56 = (tmp55 / tmp47)
    tmp57 = tl.where(tmp52, tmp21, tmp56)
    tmp58 = tmp51 * tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp50 + tmp59
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp35 + tmp61
    tmp63 = tmp62.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr0 + (x3), tmp33, None)
    tl.store(in_out_ptr1 + (x3), tmp62, None)
    tl.store(out_ptr1 + (x3), tmp63, None)
