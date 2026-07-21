
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7667712}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
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
    tmp23 = tl.load(in_ptr7 + (x3), None).to(tl.float32)
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
    tmp25 = (tmp24 / tmp15)
    tmp26 = tl.where(tmp22, tmp21, tmp25)
    tmp27 = tmp20 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp19 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp2 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
    tl.store(out_ptr0 + (x3), tmp32, None)
