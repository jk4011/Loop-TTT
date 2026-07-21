
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_slice_backward_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 603979776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_slice_backward_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 3072, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-3072) + x0 + 1024*x1), tmp2, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.load(in_ptr1 + ((-3072) + x0 + 1024*x1), tmp2, other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tl.full([1], 2048, tl.int64)
    tmp13 = tmp0 >= tmp12
    tmp14 = tmp0 < tmp1
    tmp15 = tmp13 & tmp14
    tmp16 = tl.load(in_ptr2 + ((-2048) + x0 + 1024*x1), tmp15, other=0.0).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.load(in_ptr3 + ((-2048) + x0 + 1024*x1), tmp15, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tl.where(tmp15, tmp22, tmp10)
    tmp24 = tmp11 + tmp23
    tmp25 = tl.full([1], 1024, tl.int64)
    tmp26 = tmp0 >= tmp25
    tmp27 = tmp0 < tmp12
    tmp28 = tmp26 & tmp27
    tmp29 = tl.load(in_ptr4 + ((-1024) + x0 + 1024*x1), tmp28, other=0.0).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.load(in_ptr5 + ((-1024) + x0 + 1024*x1), tmp28, other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp28, tmp33, tmp34)
    tmp36 = tl.where(tmp28, tmp35, tmp10)
    tmp37 = tmp24 + tmp36
    tmp38 = tmp0 < tmp25
    tmp39 = tl.load(in_ptr6 + (x0 + 1024*x1), tmp38, other=0.0).to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.load(in_ptr7 + (x0 + 1024*x1), tmp38, other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp38, tmp43, tmp44)
    tmp46 = tl.where(tmp38, tmp45, tmp10)
    tmp47 = tmp37 + tmp46
    tl.store(in_out_ptr0 + (x2), tmp47, None)
