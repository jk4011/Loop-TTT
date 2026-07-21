
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 402653184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_backward_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3072, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2048) + x0 + 1024*x1), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + ((-2048) + x0 + 1024*x1), tmp5, other=0.0).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tl.full([1], 1024, tl.int64)
    tmp14 = tmp0 >= tmp13
    tmp15 = tmp0 < tmp1
    tmp16 = tmp14 & tmp15
    tmp17 = tl.load(in_ptr2 + ((-1024) + x0 + 1024*x1), tmp16, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + ((-1024) + x0 + 1024*x1), tmp16, other=0.0).to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp16, tmp21, tmp11)
    tmp23 = tmp12 + tmp22
    tmp24 = tmp0 < tmp13
    tmp25 = tl.load(in_ptr4 + (x0 + 1024*x1), tmp24, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr5 + (x0 + 1024*x1), tmp24, other=0.0).to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp24, tmp27, tmp28)
    tmp30 = tl.where(tmp24, tmp29, tmp11)
    tmp31 = tmp23 + tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
