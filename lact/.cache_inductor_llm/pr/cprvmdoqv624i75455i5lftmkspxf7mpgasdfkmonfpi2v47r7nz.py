
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 188743680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 1024)
    x2 = ((xindex // 1024) % 192)
    x3 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp39 = tl.load(in_ptr5 + (x2 + 192*x1 + 196608*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp41 = tl.load(in_ptr6 + (3*x1 + 12288*x3), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 * tmp2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp1.to(tl.float32)
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp11 * tmp15
    tmp17 = tmp10 + tmp16
    tmp19 = tmp18 * tmp9
    tmp20 = tmp19 * tmp2
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp4
    tmp23 = tmp21 * tmp1
    tmp24 = -tmp23
    tmp25 = tmp7 * tmp19
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp2.to(tl.float32)
    tmp29 = tmp3 - tmp28
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp22 + tmp32
    tmp34 = tmp11 * tmp18
    tmp35 = tmp2 * tmp6
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tmp8 * tmp18
    tmp40 = tmp39.to(tl.float32)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp15
    tmp45 = tmp38 + tmp44
    tmp46 = tmp43 * tmp9
    tmp47 = tmp46 * tmp35
    tmp48 = tmp37 + tmp47
    tl.store(out_ptr0 + (x0), tmp17, None)
    tl.store(out_ptr1 + (x0), tmp45, None)
    tl.store(in_out_ptr0 + (x0), tmp48, None)
