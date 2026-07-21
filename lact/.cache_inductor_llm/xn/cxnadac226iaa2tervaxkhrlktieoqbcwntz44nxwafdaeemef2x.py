# AOT ID: ['1_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/32/c32mlzegezd4wp672m2oube424ywphbntonvis6basvzwof547z2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %permute_61 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%tangents_1, [0, 2, 1]), kwargs = {})
#   %slice_39 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_61, 2, 3072, 4096), kwargs = {})
#   %clone_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_39,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_slice_transpose_0 = async_compile.triton('triton_poi_fused_clone_slice_transpose_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_transpose_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_transpose_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (589824 + x0 + 786432*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/n6/cn64nxoswt6cxog5apznmd7puubvyo3hg7h2zv26kppghd35bee4.py
# Topologically Sorted Source Nodes: [add_127, truediv_16], Original ATen: [aten._to_copy, aten.masked_fill, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
# Source node to ATen node mapping:
#   add_127 => add_127
#   truediv_16 => div_16
# Graph fragment:
#   %bmm_166 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_166]
#   %add_123 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_123]
#   %pow_40 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_40]
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_4]
#   %sum_25 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_25]
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_132]
#   %convert_element_type_549 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_166, torch.float32), kwargs = {})
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_127 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_40, 1e-05), kwargs = {})
#   %div_16 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_123, %add_127), kwargs = {})
#   %mul_203 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_549, %div_16), kwargs = {})
#   %mul_204 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_549, %pow_4), kwargs = {})
#   %sum_24 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_203, [2], True), kwargs = {dtype: torch.float32})
#   %div_23 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_16, %add_127), kwargs = {})
#   %neg_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_204,), kwargs = {})
#   %mul_205 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %div_23), kwargs = {})
#   %div_24 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_204, %add_127), kwargs = {})
#   %sum_25 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_205, [2], True), kwargs = {dtype: torch.float32})
#   %div_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_123, %pow_40), kwargs = {})
#   %eq_1 : Tensor "b8[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_40, 0), kwargs = {})
#   %where_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_6, %div_25), kwargs = {})
#   %mul_206 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_25, %where_1), kwargs = {})
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_24, %mul_206), kwargs = {})
#   %convert_element_type_728 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_132, torch.bfloat16), kwargs = {})
#   return %sum_24,%sum_25,%add_132,%convert_element_type_728
triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1 = async_compile.triton('triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 21233664}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp2 / tmp5)
    tmp7 = tmp1 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tmp1 * tmp12
    tmp14 = -tmp13
    tmp15 = (tmp6 / tmp5)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = (tmp13 / tmp5)
    tmp22 = 0.0
    tmp23 = tmp3 == tmp22
    tmp24 = (tmp2 / tmp3)
    tmp25 = tl.where(tmp23, tmp22, tmp24)
    tmp26 = tmp20 * tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp27, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp28, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/cr/ccrflcjdngpgth3zik4waesasalfqduq7w52s4wgxshhmpdof5va.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_233 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_233]
#   %bmm_234 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_234]
#   %bmm_232 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_232]
#   %mul_246 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_234, 1.2012), kwargs = {})
#   %add_187 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_233, %mul_246), kwargs = {})
#   %mul_247 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_232, -3.0525), kwargs = {})
#   %add_188 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_187, %mul_247), kwargs = {})
#   return %add_188
triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = 1.2012
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = -3.0525
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ee/ceerk6uh45j4cp6kysl5k5wfgtzeuge6ntadkp2tyum7lyg6642g.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_231 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_231]
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_132]
#   %bmm_236 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_236]
#   %bmm_235 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_235]
#   %convert_element_type_733 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_231, torch.float32), kwargs = {})
#   %mul_245 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_132, 2.8366), kwargs = {})
#   %add_186 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_733, %mul_245), kwargs = {})
#   %convert_element_type_742 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_236, torch.float32), kwargs = {})
#   %add_189 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_186, %convert_element_type_742), kwargs = {})
#   %convert_element_type_743 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_235, torch.float32), kwargs = {})
#   %permute_145 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_743, [0, 2, 1]), kwargs = {})
#   %add_190 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_189, %permute_145), kwargs = {})
#   %convert_element_type_744 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_190, torch.bfloat16), kwargs = {})
#   return %convert_element_type_744
triton_poi_fused__to_copy_add_mul_transpose_3 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 14155776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (y0 + 192*x2 + 36864*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 2.8366
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/li/cligh5imsd5grvwyeph3m6badbfu4zeo6qwyz54nmqarxw72qk2f.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_239 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_239]
#   %bmm_240 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_240]
#   %bmm_238 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_238]
#   %mul_249 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_240, 1.2046), kwargs = {})
#   %add_192 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_239, %mul_249), kwargs = {})
#   %mul_250 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_238, -3.1427), kwargs = {})
#   %add_193 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_192, %mul_250), kwargs = {})
#   return %add_193
triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = 1.2046
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = -3.1427
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/zv/czvyo5belxpecjka5iaxuopdcmemb5c2r5ov6s2uxgx4bojaa25w.py
# Topologically Sorted Source Nodes: [gate_3], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   gate_3 => convert_element_type_540, convert_element_type_541, mul_192, sigmoid_12
# Graph fragment:
#   %bmm_165 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_165]
#   %bmm_163 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_163]
#   %bmm_162 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_162]
#   %convert_element_type_540 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_163, torch.float32), kwargs = {})
#   %sigmoid_12 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_540,), kwargs = {})
#   %mul_192 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_540, %sigmoid_12), kwargs = {})
#   %convert_element_type_541 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_192, torch.bfloat16), kwargs = {})
#   %mul_194 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_165, %convert_element_type_541), kwargs = {})
#   %mul_195 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_165, %bmm_162), kwargs = {})
#   %sigmoid_13 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_163,), kwargs = {})
#   %full_default_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_4, %sigmoid_13), kwargs = {})
#   %mul_196 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_163, %sub_3), kwargs = {})
#   %add_129 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_196, 1), kwargs = {})
#   %mul_197 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_13, %add_129), kwargs = {})
#   %mul_198 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_195, %mul_197), kwargs = {})
#   return %mul_194,%mul_198
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 88080384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/7h/c7hncpgwvrmwnbbdo34w744fvkkmfpd6z7s3zxrdpfwl5onzbdhs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_237 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_237]
#   %bmm_231 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_231]
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_132]
#   %bmm_236 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_236]
#   %bmm_235 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_235]
#   %bmm_242 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_242]
#   %bmm_241 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_241]
#   %add_195 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_195]
#   %convert_element_type_733 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_231, torch.float32), kwargs = {})
#   %mul_245 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_132, 2.8366), kwargs = {})
#   %add_186 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_733, %mul_245), kwargs = {})
#   %convert_element_type_742 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_236, torch.float32), kwargs = {})
#   %add_189 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_186, %convert_element_type_742), kwargs = {})
#   %convert_element_type_743 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_235, torch.float32), kwargs = {})
#   %permute_145 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_743, [0, 2, 1]), kwargs = {})
#   %add_190 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_189, %permute_145), kwargs = {})
#   %convert_element_type_749 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_237, torch.float32), kwargs = {})
#   %mul_248 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_190, 2.8769), kwargs = {})
#   %add_191 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_749, %mul_248), kwargs = {})
#   %convert_element_type_758 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_242, torch.float32), kwargs = {})
#   %add_194 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_191, %convert_element_type_758), kwargs = {})
#   %convert_element_type_759 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_241, torch.float32), kwargs = {})
#   %permute_152 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_759, [0, 2, 1]), kwargs = {})
#   %add_195 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %permute_152), kwargs = {})
#   %convert_element_type_760 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_195, torch.bfloat16), kwargs = {})
#   return %add_195,%convert_element_type_760
triton_poi_fused__to_copy_add_mul_transpose_6 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28311552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 192)
    x2 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 2.8366
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = 2.8769
    tmp15 = tmp13 * tmp14
    tmp16 = tmp1 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/yy/cyyrgtqsbfmy53bwk6uitgugusi25py54h2etbsgow66w22xzexu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_245 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_245]
#   %bmm_246 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_246]
#   %bmm_244 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_244]
#   %mul_252 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_246, 2.3037), kwargs = {})
#   %add_197 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_245, %mul_252), kwargs = {})
#   %mul_253 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_244, -5.5913), kwargs = {})
#   %add_198 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_197, %mul_253), kwargs = {})
#   return %add_198
triton_poi_fused_add_mul_7 = async_compile.triton('triton_poi_fused_add_mul_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = 2.3037
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = -5.5913
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/62/c625rzu4wdfx447bqtljg3dobmqcopz2iuha7gymqpw7nkz6swzz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_183 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_183]
#   %add_143 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_143]
#   %bmm_188 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_188]
#   %bmm_187 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_187]
#   %convert_element_type_599 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_183, torch.float32), kwargs = {})
#   %mul_217 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_143, 3.7418), kwargs = {})
#   %add_144 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_599, %mul_217), kwargs = {})
#   %convert_element_type_608 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_188, torch.float32), kwargs = {})
#   %add_147 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %convert_element_type_608), kwargs = {})
#   %convert_element_type_609 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_187, torch.float32), kwargs = {})
#   %permute_89 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_609, [0, 2, 1]), kwargs = {})
#   %add_148 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_147, %permute_89), kwargs = {})
#   %convert_element_type_610 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_148, torch.bfloat16), kwargs = {})
#   return %convert_element_type_610
triton_poi_fused__to_copy_add_mul_transpose_8 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 14155776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (y0 + 192*x2 + 36864*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 3.7418
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/5c/c5cqqmqbkqgc3cat4lur6n3zh222vviss4ooonecbbjjgqnu36bf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_191 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_191]
#   %bmm_192 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_192]
#   %bmm_190 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_190]
#   %mul_221 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_192, 2.6377), kwargs = {})
#   %add_150 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_191, %mul_221), kwargs = {})
#   %mul_222 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_190, -6.3029), kwargs = {})
#   %add_151 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_150, %mul_222), kwargs = {})
#   return %add_151
triton_poi_fused_add_mul_9 = async_compile.triton('triton_poi_fused_add_mul_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = 2.6377
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = -6.3029
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/aj/cajvogpitohxcg6zuhtvmljlsvyj2ffhicbbfyi6d3xneqyo3tqp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_189 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_189]
#   %bmm_183 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_183]
#   %add_143 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_143]
#   %bmm_188 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_188]
#   %bmm_187 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_187]
#   %bmm_194 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_194]
#   %bmm_193 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_193]
#   %add_153 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_153]
#   %convert_element_type_599 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_183, torch.float32), kwargs = {})
#   %mul_217 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_143, 3.7418), kwargs = {})
#   %add_144 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_599, %mul_217), kwargs = {})
#   %convert_element_type_608 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_188, torch.float32), kwargs = {})
#   %add_147 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %convert_element_type_608), kwargs = {})
#   %convert_element_type_609 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_187, torch.float32), kwargs = {})
#   %permute_89 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_609, [0, 2, 1]), kwargs = {})
#   %add_148 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_147, %permute_89), kwargs = {})
#   %convert_element_type_615 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_189, torch.float32), kwargs = {})
#   %mul_220 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_148, 3.9505), kwargs = {})
#   %add_149 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_615, %mul_220), kwargs = {})
#   %convert_element_type_624 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_194, torch.float32), kwargs = {})
#   %add_152 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %convert_element_type_624), kwargs = {})
#   %convert_element_type_625 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_193, torch.float32), kwargs = {})
#   %permute_96 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_625, [0, 2, 1]), kwargs = {})
#   %add_153 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_152, %permute_96), kwargs = {})
#   %convert_element_type_626 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_153, torch.bfloat16), kwargs = {})
#   return %add_153,%convert_element_type_626
triton_poi_fused__to_copy_add_mul_transpose_10 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28311552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 192)
    x2 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp8 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr5 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 3.7418
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = 3.9505
    tmp15 = tmp13 * tmp14
    tmp16 = tmp1 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/de/cdealtkrtzhkku5xbgrueh7izgg3r5v5ogwsfpplhdetxed4c7aq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_197 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_197]
#   %bmm_198 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_198]
#   %bmm_196 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_196]
#   %mul_224 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_198, 2.927), kwargs = {})
#   %add_155 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_197, %mul_224), kwargs = {})
#   %mul_225 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_196, -6.8946), kwargs = {})
#   %add_156 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_155, %mul_225), kwargs = {})
#   return %add_156
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = 2.927
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = -6.8946
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ip/cipml6yiwntyvnhhdikqgt642yfrngpxf3undxaro7jxq4jx37by.py
# Topologically Sorted Source Nodes: [add_112, X_57], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
# Source node to ATen node mapping:
#   X_57 => div_14
#   add_112 => add_112
# Graph fragment:
#   %bmm_195 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_195]
#   %add_153 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_153]
#   %bmm_200 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_200]
#   %bmm_199 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_199]
#   %convert_element_type_485 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_485]
#   %pow_36 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_36]
#   %convert_element_type_631 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_195, torch.float32), kwargs = {})
#   %mul_223 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_153, 4.0848), kwargs = {})
#   %add_154 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_631, %mul_223), kwargs = {})
#   %convert_element_type_640 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_200, torch.float32), kwargs = {})
#   %add_157 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, %convert_element_type_640), kwargs = {})
#   %convert_element_type_641 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_199, torch.float32), kwargs = {})
#   %permute_103 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_641, [0, 2, 1]), kwargs = {})
#   %add_158 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_157, %permute_103), kwargs = {})
#   %add_112 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_36, 1e-07), kwargs = {})
#   %div_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_485, %add_112), kwargs = {})
#   %div_31 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_14, %add_112), kwargs = {})
#   %neg_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_158,), kwargs = {})
#   %mul_226 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_3, %div_31), kwargs = {})
#   %sum_28 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_226, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf60
triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 11796800}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 4.0848
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp12 = tl.load(in_ptr3 + (192*(((r0_2 + 7373*x0) % 192)) + 36864*x1 + ((((r0_2 + 7373*x0) // 192) % 192))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = -tmp14
        tmp16 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.load(in_ptr5 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 1e-07
        tmp20 = tmp18 + tmp19
        tmp21 = (tmp17 / tmp20)
        tmp22 = (tmp21 / tmp20)
        tmp23 = tmp15 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/a2/ca2bmys4k6y55ysoqzbgpfjojnel75cfq5ftzay6sp34qqiguf4c.py
# Topologically Sorted Source Nodes: [add_112, X_57], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
# Source node to ATen node mapping:
#   X_57 => div_14
#   add_112 => add_112
# Graph fragment:
#   %buf60 : Tensor "f32[32, 1, 1, 5][5, 160, 160, 1]cuda:0" = PlaceHolder[target=buf60]
#   %convert_element_type_631 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_195, torch.float32), kwargs = {})
#   %mul_223 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_153, 4.0848), kwargs = {})
#   %add_154 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_631, %mul_223), kwargs = {})
#   %convert_element_type_640 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_200, torch.float32), kwargs = {})
#   %add_157 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, %convert_element_type_640), kwargs = {})
#   %convert_element_type_641 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_199, torch.float32), kwargs = {})
#   %permute_103 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_641, [0, 2, 1]), kwargs = {})
#   %add_158 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_157, %permute_103), kwargs = {})
#   %add_112 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_36, 1e-07), kwargs = {})
#   %div_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_485, %add_112), kwargs = {})
#   %div_31 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_14, %add_112), kwargs = {})
#   %neg_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_158,), kwargs = {})
#   %mul_226 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_3, %div_31), kwargs = {})
#   %sum_28 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_226, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %sum_28
triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 400}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/lf/clfeo5zjtdpexsreh4aa4xhxkmer3s7nrtqlnvle3sopt4nddqbl.py
# Topologically Sorted Source Nodes: [add_112], Original ATen: [aten.masked_fill, aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.eq]
# Source node to ATen node mapping:
#   add_112 => add_112
# Graph fragment:
#   %bmm_195 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_195]
#   %add_153 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_153]
#   %bmm_200 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_200]
#   %bmm_199 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_199]
#   %pow_36 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_36]
#   %sum_28 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_28]
#   %convert_element_type_485 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_485]
#   %convert_element_type_644 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_644]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_631 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_195, torch.float32), kwargs = {})
#   %mul_223 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_153, 4.0848), kwargs = {})
#   %add_154 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_631, %mul_223), kwargs = {})
#   %convert_element_type_640 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_200, torch.float32), kwargs = {})
#   %add_157 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, %convert_element_type_640), kwargs = {})
#   %convert_element_type_641 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_199, torch.float32), kwargs = {})
#   %permute_103 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_641, [0, 2, 1]), kwargs = {})
#   %add_158 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_157, %permute_103), kwargs = {})
#   %add_112 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_36, 1e-07), kwargs = {})
#   %div_32 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_158, %add_112), kwargs = {})
#   %convert_element_type_642 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_32, torch.bfloat16), kwargs = {})
#   %div_33 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_485, %pow_36), kwargs = {})
#   %eq_3 : Tensor "b8[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_36, 0), kwargs = {})
#   %where_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_6, %div_33), kwargs = {})
#   %mul_227 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_28, %where_3), kwargs = {})
#   %convert_element_type_643 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_227, torch.bfloat16), kwargs = {})
#   %add_159 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_642, %convert_element_type_643), kwargs = {})
#   %convert_element_type_644 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_159, torch.float32), kwargs = {})
#   %convert_element_type_811 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_644, torch.bfloat16), kwargs = {})
#   return %convert_element_type_644,%convert_element_type_811
triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25952256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 192)
    x2 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (x3), None)
    tmp6 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None).to(tl.float32)
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
    tmp22 = (tmp21 / tmp12)
    tmp23 = tl.where(tmp19, tmp18, tmp22)
    tmp24 = tmp17 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp16 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp27, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/7e/c7e2wvkgqgfhygq522543xspcwwynyeljc2nugymkwu4dgiwtzsv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_644 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_644]
#   %add_46 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_46]
#   %mul_262 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_644, %add_46), kwargs = {})
#   %sum_31 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_262, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf153
triton_red_fused_mul_sum_15 = async_compile.triton('triton_red_fused_mul_sum_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 9437440}}
)
@triton.jit
def triton_red_fused_mul_sum_15(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ck/cckntppe4jmco4owmes4p4puyvk7w3jtqtb37stuow2dxct3ct56.py
# Topologically Sorted Source Nodes: [ki_2], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   ki_2 => slice_22
# Graph fragment:
#   %bmm_261 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_261]
#   %primals_7 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %bmm_263 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_263]
#   %convert_element_type_818 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_261, torch.float32), kwargs = {})
#   %slice_22 : Tensor "f32[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_7, 1, 2048, 3072), kwargs = {})
#   %mul_268 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_818, %slice_22), kwargs = {})
#   %sum_34 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_268, [2], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_823 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_263, torch.float32), kwargs = {})
#   %mul_270 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_823, %slice_22), kwargs = {})
#   %sum_35 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_270, [2], True), kwargs = {dtype: torch.float32})
#   return %sum_34,%sum_35
triton_red_fused__to_copy_mul_slice_sum_16 = async_compile.triton('triton_red_fused__to_copy_mul_slice_sum_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_slice_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 50331648}}
)
@triton.jit
def triton_red_fused__to_copy_mul_slice_sum_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (393216 + r0_2 + 192*x0 + 786432*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp2
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/2j/c2jr5yllqpguucwuhhrql6cmmk7gkn34t5agntvocnck3ettazi5.py
# Topologically Sorted Source Nodes: [silu_7, lr1i_2, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86], Original ATen: [aten.fill, aten._to_copy, aten.silu, aten.slice, aten.mul, aten.transpose, aten.sigmoid, aten.rsub, aten.add, aten.neg, aten.sigmoid_backward, aten.sub]
# Source node to ATen node mapping:
#   add_86 => add_86
#   dgate_2 => mul_134
#   lr1i_2 => slice_25
#   mul_126 => mul_135
#   mul_127 => mul_136
#   sigma_2 => sigmoid_11
#   silu_7 => convert_element_type_375, convert_element_type_376, mul_130, sigmoid_9
#   sub_2 => sub_2
# Graph fragment:
#   %bmm_264 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_264]
#   %bmm_111 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_111]
#   %bmm_112 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %bmm_262 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_262]
#   %bmm_113 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_113]
#   %bmm_265 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_265]
#   %primals_8 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %add_219 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=add_219]
#   %full_default_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_828 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_265, torch.float32), kwargs = {})
#   %convert_element_type_375 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_111, torch.float32), kwargs = {})
#   %sigmoid_9 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_375,), kwargs = {})
#   %mul_130 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_375, %sigmoid_9), kwargs = {})
#   %convert_element_type_376 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_130, torch.bfloat16), kwargs = {})
#   %slice_25 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_8, 1, 2048, 3072), kwargs = {})
#   %mul_273 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_828, %slice_25), kwargs = {})
#   %convert_element_type_829 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_273, torch.bfloat16), kwargs = {})
#   %permute_180 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_829, [0, 2, 1]), kwargs = {})
#   %mul_134 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_113, %bmm_112), kwargs = {})
#   %sigmoid_11 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_111,), kwargs = {})
#   %mul_135 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %sigmoid_11), kwargs = {})
#   %mul_274 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_264, %mul_135), kwargs = {})
#   %sub_2 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_11), kwargs = {})
#   %mul_136 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_111, %sub_2), kwargs = {})
#   %add_86 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, 1), kwargs = {})
#   %mul_275 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_264, %add_86), kwargs = {})
#   %mul_276 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %bmm_111), kwargs = {})
#   %mul_277 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %sub_2), kwargs = {})
#   %neg_6 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_276,), kwargs = {})
#   %mul_278 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %mul_134), kwargs = {})
#   %mul_279 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %sigmoid_11), kwargs = {})
#   %add_215 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_6, %mul_278), kwargs = {})
#   %convert_element_type_830 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_215, torch.float32), kwargs = {})
#   %convert_element_type_831 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_11, torch.float32), kwargs = {})
#   %sub_4 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_831), kwargs = {})
#   %mul_280 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_831, %sub_4), kwargs = {})
#   %mul_281 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_830, %mul_280), kwargs = {})
#   %convert_element_type_832 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_281, torch.bfloat16), kwargs = {})
#   %add_216 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_277, %convert_element_type_832), kwargs = {})
#   %mul_282 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_279, %bmm_113), kwargs = {})
#   %mul_283 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_279, %bmm_112), kwargs = {})
#   %mul_284 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_262, %bmm_113), kwargs = {})
#   %mul_285 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_262, %convert_element_type_376), kwargs = {})
#   %add_217 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %mul_285), kwargs = {})
#   %sub_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_4, %sigmoid_11), kwargs = {})
#   %mul_286 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_111, %sub_5), kwargs = {})
#   %add_218 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_286, 1), kwargs = {})
#   %mul_287 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_11, %add_218), kwargs = {})
#   %mul_288 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %mul_287), kwargs = {})
#   %add_219 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_216, %mul_288), kwargs = {})
#   %mul_289 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_180, %convert_element_type_376), kwargs = {})
#   %mul_290 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_180, %bmm_112), kwargs = {})
#   %add_221 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_282, %mul_289), kwargs = {})
#   %mul_293 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_290, %mul_287), kwargs = {})
#   %add_223 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_219, %mul_293), kwargs = {})
#   return %add_217,%add_219,%add_221,%add_223
triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_17 = async_compile.triton('triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 188743680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp41 = tl.load(in_ptr6 + (6144 + 3*x1 + 12288*x3), None, eviction_policy='evict_last')
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/zu/czufayohfnyywz6rkjxi44w6h4rdkngwjodanip6dizgms74prwq.py
# Topologically Sorted Source Nodes: [silu_7, hidden_2, transpose_43], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.transpose, aten.sum]
# Source node to ATen node mapping:
#   hidden_2 => mul_131
#   silu_7 => convert_element_type_375, convert_element_type_376, mul_130, sigmoid_9
#   transpose_43 => permute_44
# Graph fragment:
#   %bmm_265 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_265]
#   %bmm_111 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_111]
#   %bmm_112 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %convert_element_type_828 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_265, torch.float32), kwargs = {})
#   %convert_element_type_375 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_111, torch.float32), kwargs = {})
#   %sigmoid_9 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_375,), kwargs = {})
#   %mul_130 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_375, %sigmoid_9), kwargs = {})
#   %convert_element_type_376 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_130, torch.bfloat16), kwargs = {})
#   %mul_131 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_376, %bmm_112), kwargs = {})
#   %permute_44 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_131, [0, 2, 1]), kwargs = {})
#   %mul_272 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_828, %permute_44), kwargs = {})
#   %sum_36 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_272, [2], True), kwargs = {dtype: torch.float32})
#   return %sum_36
triton_red_fused__to_copy_mul_silu_sum_transpose_18 = async_compile.triton('triton_red_fused__to_copy_mul_silu_sum_transpose_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_silu_sum_transpose_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25427968, 'r0_': 12582912}}
)
@triton.jit
def triton_red_fused__to_copy_mul_silu_sum_transpose_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ez/cezro4w5kyvprm47ifebfzryk3lcucjwahfzue4fun5jfagbcsb2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %permute_61 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%tangents_1, [0, 2, 1]), kwargs = {})
#   %slice_39 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_61, 2, 3072, 4096), kwargs = {})
#   %full_default_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_39, %full_default_3), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_61, %copy_5, 2, 3072, 4096), kwargs = {})
#   %slice_42 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 2048, 3072), kwargs = {})
#   %clone_6 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_42,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
triton_poi_fused_clone_copy_slice_transpose_zeros_like_19 = async_compile.triton('triton_poi_fused_clone_copy_slice_transpose_zeros_like_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_slice_transpose_zeros_like_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_copy_slice_transpose_zeros_like_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 192) % 1024)
    x2 = xindex // 196608
    x3 = (xindex % 196608)
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (393216 + x3 + 786432*x2), None).to(tl.float32)
    tmp0 = 2048 + x1
    tmp1 = tl.full([1], 3072, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 0.0
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ka/ckaatkphzjqvfcofsv6w62fylxagxknndpu6jvcfgxu5fmtmwrcp.py
# Topologically Sorted Source Nodes: [add_84, truediv_10], Original ATen: [aten.masked_fill, aten._to_copy, aten.transpose, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
# Source node to ATen node mapping:
#   add_84 => add_84
#   truediv_10 => div_10
# Graph fragment:
#   %bmm_268 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_268]
#   %bmm_274 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_274]
#   %add_80 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_80]
#   %pow_28 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_28]
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_4]
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_132]
#   %sum_40 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_40]
#   %add_237 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_237]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_837 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_268, torch.float32), kwargs = {})
#   %permute_184 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_837, [0, 2, 1]), kwargs = {})
#   %convert_element_type_854 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_274, torch.float32), kwargs = {})
#   %add_226 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_184, %convert_element_type_854), kwargs = {})
#   %add_84 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_28, 1e-05), kwargs = {})
#   %div_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_80, %add_84), kwargs = {})
#   %mul_303 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_226, %div_10), kwargs = {})
#   %mul_304 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_226, %pow_4), kwargs = {})
#   %sum_39 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_303, [2], True), kwargs = {dtype: torch.float32})
#   %div_48 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_10, %add_84), kwargs = {})
#   %neg_8 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_304,), kwargs = {})
#   %mul_305 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_8, %div_48), kwargs = {})
#   %div_49 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_304, %add_84), kwargs = {})
#   %sum_40 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_305, [2], True), kwargs = {dtype: torch.float32})
#   %add_236 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_132, %div_49), kwargs = {})
#   %div_50 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_80, %pow_28), kwargs = {})
#   %eq_7 : Tensor "b8[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_28, 0), kwargs = {})
#   %where_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_7, %full_default_6, %div_50), kwargs = {})
#   %mul_306 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_40, %where_7), kwargs = {})
#   %add_237 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_236, %mul_306), kwargs = {})
#   %convert_element_type_1033 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_237, torch.bfloat16), kwargs = {})
#   return %sum_39,%sum_40,%add_237,%convert_element_type_1033
triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20 = async_compile.triton('triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 25952256}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6144
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 192)
    x1 = xindex // 192
    x3 = xindex
    tmp6 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 192*r0_2 + 36864*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp7 = 1e-05
        tmp8 = tmp6 + tmp7
        tmp9 = (tmp5 / tmp8)
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp15 = tmp4 * tmp14
        tmp16 = -tmp15
        tmp17 = (tmp9 / tmp8)
        tmp18 = tmp16 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp22 = tl.load(in_out_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr0 + (x0 + 192*r0_2 + 36864*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27 * tmp14
        tmp29 = 1e-05
        tmp30 = tmp6 + tmp29
        tmp31 = (tmp28 / tmp30)
        tmp32 = tmp22 + tmp31
        tmp33 = 0.0
        tmp34 = tmp6 == tmp33
        tmp36 = (tmp35 / tmp6)
        tmp37 = tl.where(tmp34, tmp33, tmp36)
        tmp38 = tmp20 * tmp37
        tmp39 = tmp32 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_2 + 192*x3), tmp39, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_2 + 192*x3), tmp40, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ef/cefif4durdz46umv6354ljguqywpm2huqvvfyavv55aqvu2sevdp.py
# Topologically Sorted Source Nodes: [add_85, truediv_11], Original ATen: [aten.masked_fill, aten._to_copy, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
# Source node to ATen node mapping:
#   add_85 => add_85
#   truediv_11 => div_11
# Graph fragment:
#   %bmm_270 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_270]
#   %bmm_278 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_278]
#   %add_82 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_82]
#   %pow_30 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_30]
#   %pow_6 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_6]
#   %add_131 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_131]
#   %sum_38 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_38]
#   %add_234 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_234]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_842 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_270, torch.float32), kwargs = {})
#   %convert_element_type_865 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_278, torch.float32), kwargs = {})
#   %add_229 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_842, %convert_element_type_865), kwargs = {})
#   %add_85 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_30, 1e-05), kwargs = {})
#   %div_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_82, %add_85), kwargs = {})
#   %mul_299 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_229, %div_11), kwargs = {})
#   %mul_300 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_229, %pow_6), kwargs = {})
#   %sum_37 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_299, [2], True), kwargs = {dtype: torch.float32})
#   %div_44 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_11, %add_85), kwargs = {})
#   %neg_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_300,), kwargs = {})
#   %mul_301 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_7, %div_44), kwargs = {})
#   %div_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_300, %add_85), kwargs = {})
#   %sum_38 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_301, [2], True), kwargs = {dtype: torch.float32})
#   %add_233 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_131, %div_45), kwargs = {})
#   %div_46 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_82, %pow_30), kwargs = {})
#   %eq_6 : Tensor "b8[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_30, 0), kwargs = {})
#   %where_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_6, %full_default_6, %div_46), kwargs = {})
#   %mul_302 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_38, %where_6), kwargs = {})
#   %add_234 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_233, %mul_302), kwargs = {})
#   %convert_element_type_867 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_234, torch.bfloat16), kwargs = {})
#   return %sum_37,%sum_38,%add_234,%convert_element_type_867
triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21 = async_compile.triton('triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 28311552}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = (tmp5 / tmp8)
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tmp4 * tmp15
    tmp17 = -tmp16
    tmp18 = (tmp9 / tmp8)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp25 = (tmp16 / tmp8)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp6 == tmp27
    tmp29 = (tmp5 / tmp6)
    tmp30 = tl.where(tmp28, tmp27, tmp29)
    tmp31 = tmp23 * tmp30
    tmp32 = tmp26 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp32, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp33, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/d6/cd6xavznwrcfvsrmleoj3leosv247gyh4ttoiiyfabfk3f5auylo.py
# Topologically Sorted Source Nodes: [add_69], Original ATen: [aten.masked_fill, aten.mul, aten._to_copy, aten.add, aten.transpose, aten.div, aten.eq]
# Source node to ATen node mapping:
#   add_69 => add_69
# Graph fragment:
#   %convert_element_type_644 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_644]
#   %mean_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_2]
#   %bmm_303 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_303]
#   %add_260 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_260]
#   %bmm_308 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_308]
#   %bmm_307 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_307]
#   %pow_24 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_24]
#   %sum_43 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_43]
#   %convert_element_type_307 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_307]
#   %add_267 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_267]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_263 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_644, %mean_2), kwargs = {})
#   %convert_element_type_936 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_303, torch.float32), kwargs = {})
#   %mul_323 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_260, 4.0848), kwargs = {})
#   %add_261 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_936, %mul_323), kwargs = {})
#   %convert_element_type_945 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_308, torch.float32), kwargs = {})
#   %add_264 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_261, %convert_element_type_945), kwargs = {})
#   %convert_element_type_946 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_307, torch.float32), kwargs = {})
#   %permute_233 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_946, [0, 2, 1]), kwargs = {})
#   %add_265 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_264, %permute_233), kwargs = {})
#   %add_69 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_24, 1e-07), kwargs = {})
#   %div_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_265, %add_69), kwargs = {})
#   %convert_element_type_947 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_57, torch.bfloat16), kwargs = {})
#   %div_58 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_307, %pow_24), kwargs = {})
#   %eq_9 : Tensor "b8[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_24, 0), kwargs = {})
#   %where_9 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_9, %full_default_6, %div_58), kwargs = {})
#   %mul_327 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_43, %where_9), kwargs = {})
#   %convert_element_type_948 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_327, torch.bfloat16), kwargs = {})
#   %add_266 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_947, %convert_element_type_948), kwargs = {})
#   %convert_element_type_949 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_266, torch.float32), kwargs = {})
#   %add_267 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %convert_element_type_949), kwargs = {})
#   %convert_element_type_1116 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_267, torch.bfloat16), kwargs = {})
#   return %add_267,%convert_element_type_1116
triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_22 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30670848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/en/cennipvy3w5dalipdantn3akrgslnbivcay7uxylhuyzmfcgv72k.py
# Topologically Sorted Source Nodes: [X_28, add_58, X_29], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
# Source node to ATen node mapping:
#   X_28 => convert_element_type_260
#   X_29 => div_7
#   add_58 => add_58
# Graph fragment:
#   %bmm_333 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_333]
#   %add_287 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_287]
#   %bmm_338 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_338]
#   %bmm_337 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_337]
#   %add_44 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_44]
#   %pow_22 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_22]
#   %convert_element_type_1019 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_333, torch.float32), kwargs = {})
#   %mul_340 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_287, 4.0848), kwargs = {})
#   %add_288 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1019, %mul_340), kwargs = {})
#   %convert_element_type_1028 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_338, torch.float32), kwargs = {})
#   %add_291 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_288, %convert_element_type_1028), kwargs = {})
#   %convert_element_type_1029 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_337, torch.float32), kwargs = {})
#   %permute_268 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1029, [0, 2, 1]), kwargs = {})
#   %add_292 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %permute_268), kwargs = {})
#   %convert_element_type_260 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_44, torch.bfloat16), kwargs = {})
#   %add_58 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_22, 1e-07), kwargs = {})
#   %div_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_260, %add_58), kwargs = {})
#   %div_60 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_7, %add_58), kwargs = {})
#   %neg_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_292,), kwargs = {})
#   %mul_343 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_11, %div_60), kwargs = {})
#   %sum_44 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_343, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf286
triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_23 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 14156160}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 4.0848
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp12 = tl.load(in_ptr3 + (192*(((r0_2 + 7373*x0) % 192)) + 36864*x1 + ((((r0_2 + 7373*x0) // 192) % 192))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = -tmp14
        tmp16 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.load(in_ptr5 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = 1e-07
        tmp21 = tmp19 + tmp20
        tmp22 = (tmp18 / tmp21)
        tmp23 = (tmp22 / tmp21)
        tmp24 = tmp15 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/vi/cvibk3jjgmxhhiotnukybzogwv5cuqyjxohrhd2stm5cwi5qahyz.py
# Topologically Sorted Source Nodes: [X_28, add_58, X_21, add_47], Original ATen: [aten.masked_fill, aten.mul, aten._to_copy, aten.add, aten.transpose, aten.div, aten.eq]
# Source node to ATen node mapping:
#   X_21 => convert_element_type_213
#   X_28 => convert_element_type_260
#   add_47 => add_47
#   add_58 => add_58
# Graph fragment:
#   %convert_element_type_727 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_727]
#   %mean_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_2]
#   %bmm_333 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_333]
#   %add_287 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_287]
#   %bmm_338 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_338]
#   %bmm_337 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_337]
#   %pow_22 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_22]
#   %sum_44 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_44]
#   %add_44 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_44]
#   %add_294 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_294]
#   %convert_element_type_810 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_810]
#   %bmm_363 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_363]
#   %add_314 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_314]
#   %bmm_368 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_368]
#   %bmm_367 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_367]
#   %pow_20 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_20]
#   %sum_45 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_45]
#   %add_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %add_321 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_321]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_265 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_810, %mean_2), kwargs = {})
#   %mul_267 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_727, %mean_2), kwargs = {})
#   %convert_element_type_1019 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_333, torch.float32), kwargs = {})
#   %mul_340 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_287, 4.0848), kwargs = {})
#   %add_288 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1019, %mul_340), kwargs = {})
#   %convert_element_type_1028 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_338, torch.float32), kwargs = {})
#   %add_291 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_288, %convert_element_type_1028), kwargs = {})
#   %convert_element_type_1029 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_337, torch.float32), kwargs = {})
#   %permute_268 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1029, [0, 2, 1]), kwargs = {})
#   %add_292 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %permute_268), kwargs = {})
#   %convert_element_type_260 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_44, torch.bfloat16), kwargs = {})
#   %add_58 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_22, 1e-07), kwargs = {})
#   %div_61 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_292, %add_58), kwargs = {})
#   %convert_element_type_1030 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_61, torch.bfloat16), kwargs = {})
#   %div_62 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_260, %pow_22), kwargs = {})
#   %eq_10 : Tensor "b8[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_22, 0), kwargs = {})
#   %where_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_10, %full_default_6, %div_62), kwargs = {})
#   %mul_344 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_44, %where_10), kwargs = {})
#   %convert_element_type_1031 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_344, torch.bfloat16), kwargs = {})
#   %add_293 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1030, %convert_element_type_1031), kwargs = {})
#   %convert_element_type_1032 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_293, torch.float32), kwargs = {})
#   %add_294 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %convert_element_type_1032), kwargs = {})
#   %convert_element_type_1102 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_363, torch.float32), kwargs = {})
#   %mul_357 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_314, 4.0848), kwargs = {})
#   %add_315 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1102, %mul_357), kwargs = {})
#   %convert_element_type_1111 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_368, torch.float32), kwargs = {})
#   %add_318 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_315, %convert_element_type_1111), kwargs = {})
#   %convert_element_type_1112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_367, torch.float32), kwargs = {})
#   %permute_303 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1112, [0, 2, 1]), kwargs = {})
#   %add_319 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_318, %permute_303), kwargs = {})
#   %convert_element_type_213 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %add_47 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_65 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_319, %add_47), kwargs = {})
#   %convert_element_type_1113 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_65, torch.bfloat16), kwargs = {})
#   %div_66 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_213, %pow_20), kwargs = {})
#   %eq_11 : Tensor "b8[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_20, 0), kwargs = {})
#   %where_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_11, %full_default_6, %div_66), kwargs = {})
#   %mul_361 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_45, %where_11), kwargs = {})
#   %convert_element_type_1114 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_361, torch.bfloat16), kwargs = {})
#   %add_320 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1113, %convert_element_type_1114), kwargs = {})
#   %convert_element_type_1115 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_320, torch.float32), kwargs = {})
#   %add_321 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_265, %convert_element_type_1115), kwargs = {})
#   %convert_element_type_1117 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_321, torch.bfloat16), kwargs = {})
#   %convert_element_type_1118 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_294, torch.bfloat16), kwargs = {})
#   return %add_294,%convert_element_type_1118,%add_321,%convert_element_type_1117
triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_24 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_24', '''
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/sq/csql4ki2eqwsbsx7j6k5tovbxatjr2eobzy7yzhfsvcrfx72ytxt.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw2_1, dw1_momentum, mul_11, dw1_1, dw0_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten.sum]
# Source node to ATen node mapping:
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   dw2_1 => add_3
#   mul_10 => mul_13
#   mul_11 => mul_14
# Graph fragment:
#   %add_267 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_267]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %add_321 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_321]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %add_294 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_294]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %full_default_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_13), kwargs = {})
#   %mul_362 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_267, %add_3), kwargs = {})
#   %sum_46 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_362, [1, 2], True), kwargs = {dtype: torch.float32})
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %mul_364 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_321, %add_2), kwargs = {})
#   %sum_47 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_364, [1, 2], True), kwargs = {dtype: torch.float32})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %mul_366 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_294, %add_1), kwargs = {})
#   %sum_48 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_366, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf334,%buf336,%buf338
triton_red_fused_add_mul_sum_zeros_like_25 = async_compile.triton('triton_red_fused_add_mul_sum_zeros_like_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_zeros_like_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 3, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3840, 'r0_': 21234240}}
)
@triton.jit
def triton_red_fused_add_mul_sum_zeros_like_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tmp7 * tmp6
        tmp9 = tmp5 + tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr3 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 + tmp8
        tmp20 = tmp16 * tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp26 = tl.load(in_ptr5 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr6 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 + tmp8
        tmp30 = tmp26 * tmp29
        tmp31 = tl.full(tmp30.shape, 0, tmp30.dtype)
        tmp32 = tl.where(tmp2, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, R0_BLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(r0_mask & xmask, tmp35, _tmp34)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
    tl.store(out_ptr2 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ba/cbaujfgqs2r4il3cv5hbwk3w2uw7vdx7l4ith2ylkikdfhomztbo.py
# Topologically Sorted Source Nodes: [ki_1], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   ki_1 => slice_11
# Graph fragment:
#   %bmm_369 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_369]
#   %primals_7 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %bmm_371 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_371]
#   %convert_element_type_1123 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_369, torch.float32), kwargs = {})
#   %slice_11 : Tensor "f32[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_7, 1, 1024, 2048), kwargs = {})
#   %mul_368 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1123, %slice_11), kwargs = {})
#   %sum_49 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_368, [2], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_1128 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_371, torch.float32), kwargs = {})
#   %mul_370 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1128, %slice_11), kwargs = {})
#   %sum_50 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_370, [2], True), kwargs = {dtype: torch.float32})
#   return %sum_49,%sum_50
triton_red_fused__to_copy_mul_slice_sum_26 = async_compile.triton('triton_red_fused__to_copy_mul_slice_sum_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_slice_sum_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 50331648}}
)
@triton.jit
def triton_red_fused__to_copy_mul_slice_sum_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (196608 + r0_2 + 192*x0 + 786432*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp2
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/f3/cf33qyqsbqvsnp3djuxvftznocorwbvuv3kyli7mvgiald54a277.py
# Topologically Sorted Source Nodes: [silu_4, lr1i_1, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43], Original ATen: [aten.fill, aten._to_copy, aten.silu, aten.slice, aten.mul, aten.transpose, aten.sigmoid, aten.rsub, aten.add, aten.neg, aten.sigmoid_backward, aten.sub]
# Source node to ATen node mapping:
#   add_43 => add_43
#   dgate_1 => mul_70
#   lr1i_1 => slice_14
#   mul_65 => mul_71
#   mul_66 => mul_72
#   sigma_1 => sigmoid_7
#   silu_4 => convert_element_type_197, convert_element_type_198, mul_66, sigmoid_5
#   sub_1 => sub_1
# Graph fragment:
#   %bmm_372 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_372]
#   %bmm_57 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %bmm_58 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_58]
#   %bmm_370 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_370]
#   %bmm_59 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %bmm_373 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_373]
#   %primals_8 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %add_330 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=add_330]
#   %full_default_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_1133 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_373, torch.float32), kwargs = {})
#   %convert_element_type_197 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_57, torch.float32), kwargs = {})
#   %sigmoid_5 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_197,), kwargs = {})
#   %mul_66 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_197, %sigmoid_5), kwargs = {})
#   %convert_element_type_198 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_66, torch.bfloat16), kwargs = {})
#   %slice_14 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_8, 1, 1024, 2048), kwargs = {})
#   %mul_373 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1133, %slice_14), kwargs = {})
#   %convert_element_type_1134 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_373, torch.bfloat16), kwargs = {})
#   %permute_310 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1134, [0, 2, 1]), kwargs = {})
#   %mul_70 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_59, %bmm_58), kwargs = {})
#   %sigmoid_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_57,), kwargs = {})
#   %mul_71 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %sigmoid_7), kwargs = {})
#   %mul_374 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_372, %mul_71), kwargs = {})
#   %sub_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_7), kwargs = {})
#   %mul_72 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_57, %sub_1), kwargs = {})
#   %add_43 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, 1), kwargs = {})
#   %mul_375 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_372, %add_43), kwargs = {})
#   %mul_376 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_374, %bmm_57), kwargs = {})
#   %mul_377 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_374, %sub_1), kwargs = {})
#   %neg_13 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_376,), kwargs = {})
#   %mul_378 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, %mul_70), kwargs = {})
#   %mul_379 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, %sigmoid_7), kwargs = {})
#   %add_326 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_13, %mul_378), kwargs = {})
#   %convert_element_type_1135 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_326, torch.float32), kwargs = {})
#   %convert_element_type_1136 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_7, torch.float32), kwargs = {})
#   %sub_8 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_1136), kwargs = {})
#   %mul_380 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1136, %sub_8), kwargs = {})
#   %mul_381 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1135, %mul_380), kwargs = {})
#   %convert_element_type_1137 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_381, torch.bfloat16), kwargs = {})
#   %add_327 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_377, %convert_element_type_1137), kwargs = {})
#   %mul_382 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %bmm_59), kwargs = {})
#   %mul_383 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %bmm_58), kwargs = {})
#   %mul_384 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_370, %bmm_59), kwargs = {})
#   %mul_385 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_370, %convert_element_type_198), kwargs = {})
#   %add_328 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %mul_385), kwargs = {})
#   %sub_9 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_4, %sigmoid_7), kwargs = {})
#   %mul_386 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_57, %sub_9), kwargs = {})
#   %add_329 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_386, 1), kwargs = {})
#   %mul_387 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_7, %add_329), kwargs = {})
#   %mul_388 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_384, %mul_387), kwargs = {})
#   %add_330 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_327, %mul_388), kwargs = {})
#   %mul_389 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_310, %convert_element_type_198), kwargs = {})
#   %mul_390 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_310, %bmm_58), kwargs = {})
#   %add_332 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_382, %mul_389), kwargs = {})
#   %mul_393 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_390, %mul_387), kwargs = {})
#   %add_334 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_330, %mul_393), kwargs = {})
#   return %add_328,%add_330,%add_332,%add_334
triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_27 = async_compile.triton('triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 188743680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp41 = tl.load(in_ptr6 + (3072 + 3*x1 + 12288*x3), None, eviction_policy='evict_last')
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/76/c76qv4u2mwqrmv7uzprulrerpwvlnu2bztkjdz3evr5v3yijetyr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %permute_61 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%tangents_1, [0, 2, 1]), kwargs = {})
#   %slice_39 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_61, 2, 3072, 4096), kwargs = {})
#   %full_default_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_39, %full_default_3), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_61, %copy_5, 2, 3072, 4096), kwargs = {})
#   %slice_42 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 2048, 3072), kwargs = {})
#   %copy_7 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_42, %full_default_3), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_7, 2, 2048, 3072), kwargs = {})
#   %slice_45 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_3, 2, 1024, 2048), kwargs = {})
#   %clone_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_45,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_7
triton_poi_fused_clone_copy_slice_transpose_zeros_like_28 = async_compile.triton('triton_poi_fused_clone_copy_slice_transpose_zeros_like_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_slice_transpose_zeros_like_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_copy_slice_transpose_zeros_like_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 192) % 1024)
    x2 = xindex // 196608
    x3 = (xindex % 196608)
    x4 = xindex
    tmp13 = tl.load(in_ptr0 + (196608 + x3 + 786432*x2), None).to(tl.float32)
    tmp0 = 1024 + x1
    tmp1 = tl.full([1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3072, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 0.0
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = 0.0
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/td/ctdk3lsek23jdbam3ps56kph5vhx3asomb3iboimd42r23kgdb7j.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %permute_61 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%tangents_1, [0, 2, 1]), kwargs = {})
#   %slice_39 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_61, 2, 3072, 4096), kwargs = {})
#   %full_default_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_39, %full_default_3), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_61, %copy_5, 2, 3072, 4096), kwargs = {})
#   %slice_42 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 2048, 3072), kwargs = {})
#   %copy_7 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_42, %full_default_3), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_7, 2, 2048, 3072), kwargs = {})
#   %slice_45 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_3, 2, 1024, 2048), kwargs = {})
#   %copy_9 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_45, %full_default_3), kwargs = {})
#   %slice_scatter_default_11 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %copy_9, 2, 1024, 2048), kwargs = {})
#   %slice_48 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_11, 2, 0, 1024), kwargs = {})
#   %clone_8 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_48,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_8
triton_poi_fused_clone_copy_slice_transpose_zeros_like_29 = async_compile.triton('triton_poi_fused_clone_copy_slice_transpose_zeros_like_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_slice_transpose_zeros_like_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_copy_slice_transpose_zeros_like_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 192) % 1024)
    x2 = xindex // 196608
    x3 = (xindex % 196608)
    x4 = xindex
    tmp20 = tl.load(in_ptr0 + (x3 + 786432*x2), None).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 0.0
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 3072, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = 0.0
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp0 >= tmp10
    tmp17 = 0.0
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp15, tmp21)
    tmp23 = tl.where(tmp5, tmp8, tmp22)
    tl.store(out_ptr0 + (x4), tmp23, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/no/cnojn7mlmtwtuytd7ehzjo4t4tlptwcewz33yddhobnkjnsh6qsg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_393 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_393]
#   %bmm_387 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_387]
#   %add_350 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_350]
#   %bmm_392 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_392]
#   %bmm_391 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_391]
#   %bmm_398 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_398]
#   %bmm_397 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_397]
#   %add_366 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_366]
#   %convert_element_type_1179 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_387, torch.float32), kwargs = {})
#   %mul_411 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_350, 2.8366), kwargs = {})
#   %add_357 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1179, %mul_411), kwargs = {})
#   %convert_element_type_1188 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_392, torch.float32), kwargs = {})
#   %add_360 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_357, %convert_element_type_1188), kwargs = {})
#   %convert_element_type_1189 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_391, torch.float32), kwargs = {})
#   %permute_335 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1189, [0, 2, 1]), kwargs = {})
#   %add_361 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_360, %permute_335), kwargs = {})
#   %convert_element_type_1195 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_393, torch.float32), kwargs = {})
#   %mul_414 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_361, 2.8769), kwargs = {})
#   %add_362 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1195, %mul_414), kwargs = {})
#   %convert_element_type_1204 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_398, torch.float32), kwargs = {})
#   %add_365 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_362, %convert_element_type_1204), kwargs = {})
#   %convert_element_type_1205 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_397, torch.float32), kwargs = {})
#   %permute_342 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1205, [0, 2, 1]), kwargs = {})
#   %add_366 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_365, %permute_342), kwargs = {})
#   %convert_element_type_1206 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_366, torch.bfloat16), kwargs = {})
#   return %add_366,%convert_element_type_1206
triton_poi_fused__to_copy_add_mul_transpose_30 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28311552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 192)
    x2 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp8 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr5 + (x1 + 192*x0 + 36864*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 2.8366
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = 2.8769
    tmp15 = tmp13 * tmp14
    tmp16 = tmp1 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/7a/c7axm3toocu7hqwde4emsewacqlb7oujve5y4xlign5seqqk4ti3.py
# Topologically Sorted Source Nodes: [dw0_momentum], Original ATen: [aten.zeros_like, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   dw0_momentum => full_default_1
# Graph fragment:
#   %add_383 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_383]
#   %full_default_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_462 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_383, %full_default_1), kwargs = {})
#   %sum_61 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_462, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf517
triton_red_fused_mul_sum_zeros_like_31 = async_compile.triton('triton_red_fused_mul_sum_zeros_like_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_zeros_like_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 4718720}}
)
@triton.jit
def triton_red_fused_mul_sum_zeros_like_31(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/c6/cc6vnznpf6tmxayp7zhd6l632c7q4laaurmlopgip5nnhkkfsv2c.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw1_momentum, mul_11, dw1_1, dw0_1, X_7, add_15, X_8, X, add_4, X_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.transpose, aten.div, aten.neg, aten.sum]
# Source node to ATen node mapping:
#   X => convert_element_type_35
#   X_1 => div
#   X_7 => convert_element_type_82
#   X_8 => div_1
#   add_15 => add_15
#   add_4 => add_4
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   mul_10 => mul_13
#   mul_11 => mul_14
# Graph fragment:
#   %bmm_441 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_441]
#   %add_403 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_403]
#   %bmm_446 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_446]
#   %bmm_445 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_445]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %pow_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_10]
#   %bmm_471 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_471]
#   %add_430 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_430]
#   %bmm_476 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_476]
#   %bmm_475 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_475]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %full_default_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %convert_element_type_1326 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_441, torch.float32), kwargs = {})
#   %mul_440 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_403, 4.0848), kwargs = {})
#   %add_404 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1326, %mul_440), kwargs = {})
#   %convert_element_type_1335 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_446, torch.float32), kwargs = {})
#   %add_407 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_404, %convert_element_type_1335), kwargs = {})
#   %convert_element_type_1336 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_445, torch.float32), kwargs = {})
#   %permute_398 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1336, [0, 2, 1]), kwargs = {})
#   %add_408 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_407, %permute_398), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %add_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_10, 1e-07), kwargs = {})
#   %div_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_82, %add_15), kwargs = {})
#   %div_85 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, %add_15), kwargs = {})
#   %neg_18 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_408,), kwargs = {})
#   %mul_443 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_18, %div_85), kwargs = {})
#   %sum_59 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_443, [1, 2], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_1409 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_471, torch.float32), kwargs = {})
#   %mul_457 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_430, 4.0848), kwargs = {})
#   %add_431 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1409, %mul_457), kwargs = {})
#   %convert_element_type_1418 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_476, torch.float32), kwargs = {})
#   %add_434 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_431, %convert_element_type_1418), kwargs = {})
#   %convert_element_type_1419 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_475, torch.float32), kwargs = {})
#   %permute_433 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1419, [0, 2, 1]), kwargs = {})
#   %add_435 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_434, %permute_433), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %add_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_35, %add_4), kwargs = {})
#   %div_89 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %add_4), kwargs = {})
#   %neg_19 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_435,), kwargs = {})
#   %mul_460 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_19, %div_89), kwargs = {})
#   %sum_60 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_460, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf469,%buf514
triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 23593600}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp60 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 4.0848
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp12 = tl.load(in_ptr3 + (192*(((r0_2 + 7373*x0) % 192)) + 36864*x1 + ((((r0_2 + 7373*x0) // 192) % 192))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = -tmp14
        tmp16 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.load(in_ptr5 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 0.0
        tmp20 = tmp19 * tmp18
        tmp21 = tmp17 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.load(in_ptr6 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = 1e-07
        tmp26 = tmp24 + tmp25
        tmp27 = (tmp23 / tmp26)
        tmp28 = (tmp27 / tmp26)
        tmp29 = tmp15 * tmp28
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
        tmp35 = tl.load(in_ptr7 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tl.load(in_ptr8 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tmp37 * tmp6
        tmp39 = tmp36 + tmp38
        tmp40 = tl.load(in_ptr9 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp39 + tmp41
        tmp43 = tl.load(in_ptr10 + (192*(((r0_2 + 7373*x0) % 192)) + 36864*x1 + ((((r0_2 + 7373*x0) // 192) % 192))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp42 + tmp44
        tmp46 = -tmp45
        tmp47 = tl.load(in_ptr11 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp48 + tmp20
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tl.load(in_ptr12 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tmp52 + tmp25
        tmp54 = (tmp51 / tmp53)
        tmp55 = (tmp54 / tmp53)
        tmp56 = tmp46 * tmp55
        tmp57 = tl.full(tmp56.shape, 0, tmp56.dtype)
        tmp58 = tl.where(tmp2, tmp56, tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, R0_BLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(r0_mask & xmask, tmp61, _tmp60)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp33, xmask)
    tl.store(out_ptr1 + (x3), tmp60, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/6s/c6srnyzp27ixfaa7ix7j6pxppdpluzl7d6naxnckrhrt3m32xfde.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw1_momentum, mul_11, dw1_1, dw0_1, X_7, add_15, X, add_4], Original ATen: [aten.masked_fill, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.transpose, aten.div, aten.eq]
# Source node to ATen node mapping:
#   X => convert_element_type_35
#   X_7 => convert_element_type_82
#   add_15 => add_15
#   add_4 => add_4
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   mul_10 => mul_13
#   mul_11 => mul_14
# Graph fragment:
#   %bmm_471 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_471]
#   %add_430 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_430]
#   %bmm_476 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_476]
#   %bmm_475 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_475]
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %sum_60 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %add_321 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_321]
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_1]
#   %add_436 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_436]
#   %bmm_441 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_441]
#   %add_403 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_403]
#   %bmm_446 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_446]
#   %bmm_445 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_445]
#   %pow_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_10]
#   %sum_59 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_59]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %add_294 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_294]
#   %add_409 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_409]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %mul_365 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_321, %mean_1), kwargs = {})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %mul_367 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_294, %mean_1), kwargs = {})
#   %convert_element_type_1326 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_441, torch.float32), kwargs = {})
#   %mul_440 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_403, 4.0848), kwargs = {})
#   %add_404 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1326, %mul_440), kwargs = {})
#   %convert_element_type_1335 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_446, torch.float32), kwargs = {})
#   %add_407 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_404, %convert_element_type_1335), kwargs = {})
#   %convert_element_type_1336 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_445, torch.float32), kwargs = {})
#   %permute_398 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1336, [0, 2, 1]), kwargs = {})
#   %add_408 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_407, %permute_398), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %add_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_10, 1e-07), kwargs = {})
#   %div_86 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_408, %add_15), kwargs = {})
#   %convert_element_type_1337 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_86, torch.bfloat16), kwargs = {})
#   %div_87 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_82, %pow_10), kwargs = {})
#   %eq_16 : Tensor "b8[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_10, 0), kwargs = {})
#   %where_16 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_16, %full_default_6, %div_87), kwargs = {})
#   %mul_444 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_59, %where_16), kwargs = {})
#   %convert_element_type_1338 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_444, torch.bfloat16), kwargs = {})
#   %add_409 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1337, %convert_element_type_1338), kwargs = {})
#   %convert_element_type_1339 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_409, torch.float32), kwargs = {})
#   %add_410 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %convert_element_type_1339), kwargs = {})
#   %convert_element_type_1409 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_471, torch.float32), kwargs = {})
#   %mul_457 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_430, 4.0848), kwargs = {})
#   %add_431 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1409, %mul_457), kwargs = {})
#   %convert_element_type_1418 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_476, torch.float32), kwargs = {})
#   %add_434 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_431, %convert_element_type_1418), kwargs = {})
#   %convert_element_type_1419 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_475, torch.float32), kwargs = {})
#   %permute_433 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1419, [0, 2, 1]), kwargs = {})
#   %add_435 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_434, %permute_433), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %add_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div_90 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_435, %add_4), kwargs = {})
#   %convert_element_type_1420 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_90, torch.bfloat16), kwargs = {})
#   %div_91 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_35, %pow_8), kwargs = {})
#   %eq_17 : Tensor "b8[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_8, 0), kwargs = {})
#   %where_17 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_17, %full_default_6, %div_91), kwargs = {})
#   %mul_461 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_60, %where_17), kwargs = {})
#   %convert_element_type_1421 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_461, torch.bfloat16), kwargs = {})
#   %add_436 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1420, %convert_element_type_1421), kwargs = {})
#   %convert_element_type_1422 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_436, torch.float32), kwargs = {})
#   %add_437 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_365, %convert_element_type_1422), kwargs = {})
#   %convert_element_type_1424 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_437, torch.bfloat16), kwargs = {})
#   %convert_element_type_1425 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_410, torch.bfloat16), kwargs = {})
#   return %add_436,%convert_element_type_1424,%add_409,%convert_element_type_1425
triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_zeros_like_33 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_zeros_like_33', '''
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/wg/cwgrpleq6w6cvjutf75fmsbw3sfr62qtgbzwpkxfjdscp5hw7xq7.py
# Topologically Sorted Source Nodes: [dw0_momentum, dw1_momentum], Original ATen: [aten.zeros_like, aten.mul, aten._to_copy, aten.add, aten.sum]
# Source node to ATen node mapping:
#   dw0_momentum => full_default_1
#   dw1_momentum => full_default
# Graph fragment:
#   %add_321 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_321]
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_1]
#   %add_436 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_436]
#   %add_294 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_294]
#   %add_409 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_409]
#   %full_default_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_365 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_321, %mean_1), kwargs = {})
#   %mul_367 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_294, %mean_1), kwargs = {})
#   %convert_element_type_1339 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_409, torch.float32), kwargs = {})
#   %add_410 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %convert_element_type_1339), kwargs = {})
#   %convert_element_type_1422 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_436, torch.float32), kwargs = {})
#   %add_437 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_365, %convert_element_type_1422), kwargs = {})
#   %mul_463 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_437, %full_default), kwargs = {})
#   %sum_62 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_463, [1, 2], True), kwargs = {dtype: torch.float32})
#   %mul_464 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_410, %full_default_1), kwargs = {})
#   %sum_63 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_464, [1, 2], True), kwargs = {dtype: torch.float32})
#   return %buf520,%buf523
triton_red_fused__to_copy_add_mul_sum_zeros_like_34 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_zeros_like_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_zeros_like_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 14156160}}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_zeros_like_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 7373
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 7373*x0
        tmp1 = tl.full([1, 1], 36864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
        tmp9 = 0.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr3 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16 * tmp4
        tmp18 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20 * tmp9
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp25, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/d6/cd6nihwqr66sr5yhkfj2a65rtzrp46hdijuvkef5fejfwuzpkv2d.py
# Topologically Sorted Source Nodes: [ki], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   ki => slice_1
# Graph fragment:
#   %bmm_477 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_477]
#   %primals_7 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %bmm_479 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_479]
#   %convert_element_type_1430 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_477, torch.float32), kwargs = {})
#   %slice_1 : Tensor "f32[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_7, 1, 0, 1024), kwargs = {})
#   %mul_465 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1430, %slice_1), kwargs = {})
#   %sum_64 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_465, [2], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_1435 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_479, torch.float32), kwargs = {})
#   %mul_467 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1435, %slice_1), kwargs = {})
#   %sum_65 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_467, [2], True), kwargs = {dtype: torch.float32})
#   return %sum_64,%sum_65
triton_red_fused__to_copy_mul_slice_sum_35 = async_compile.triton('triton_red_fused__to_copy_mul_slice_sum_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_slice_sum_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 50331648}}
)
@triton.jit
def triton_red_fused__to_copy_mul_slice_sum_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 192*x0 + 786432*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r0_2 + 192*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp2
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ml/cmlyaljfcze6aimyqfmfvy6vaiprjt4k25twa3ex7ivglyeggxs4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_35 : Tensor "f32[32, 1024, 1][1024, 1, 32768]cuda:0" = PlaceHolder[target=sum_35]
#   %sum_50 : Tensor "f32[32, 1024, 1][1024, 1, 32768]cuda:0" = PlaceHolder[target=sum_50]
#   %sum_65 : Tensor "f32[32, 1024, 1][1024, 1, 32768]cuda:0" = PlaceHolder[target=sum_65]
#   %full_default_12 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_4 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_12, %sum_35, 1, 2048, 3072), kwargs = {})
#   %slice_scatter_default_12 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_12, %sum_50, 1, 1024, 2048), kwargs = {})
#   %add_342 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_4, %slice_scatter_default_12), kwargs = {})
#   %slice_scatter_default_19 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_12, %sum_65, 1, 0, 1024), kwargs = {})
#   %add_461 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_342, %slice_scatter_default_19), kwargs = {})
#   return %add_461
triton_poi_fused_add_slice_backward_36 = async_compile.triton('triton_poi_fused_add_slice_backward_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_backward_36(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    tmp6 = tl.load(in_ptr0 + ((-2048) + x0 + 1024*x1), tmp5, other=0.0)
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full([1], 1024, tl.int64)
    tmp10 = tmp0 >= tmp9
    tmp11 = tmp0 < tmp1
    tmp12 = tmp10 & tmp11
    tmp13 = tl.load(in_ptr1 + ((-1024) + x0 + 1024*x1), tmp12, other=0.0)
    tmp14 = tl.where(tmp12, tmp13, tmp7)
    tmp15 = tmp8 + tmp14
    tmp16 = tmp0 < tmp9
    tmp17 = tl.load(in_ptr2 + (x0 + 1024*x1), tmp16, other=0.0)
    tmp18 = tl.where(tmp16, tmp17, tmp7)
    tmp19 = tmp15 + tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/an/canmb62ik5ayph77ecidlv64omd2yfnchm7p2odafcknsdhanoj4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.expand, aten.div, aten.slice_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_31 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_31]
#   %sum_32 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_32]
#   %sum_33 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_33]
#   %sum_46 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_46]
#   %sum_47 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_47]
#   %sum_48 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_48]
#   %sum_61 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_61]
#   %sum_62 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_62]
#   %sum_63 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_63]
#   %add_212 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_31, %sum_32), kwargs = {})
#   %add_213 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_212, %sum_33), kwargs = {})
#   %expand_270 : Tensor "f32[32, 1024, 1][1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%add_213, [32, 1024, 1]), kwargs = {})
#   %div_42 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_270, 1024), kwargs = {})
#   %full_default_12 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_2 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_12, %div_42, 1, 2048, 3072), kwargs = {})
#   %add_322 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_46, %sum_47), kwargs = {})
#   %add_323 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_322, %sum_48), kwargs = {})
#   %expand_271 : Tensor "f32[32, 1024, 1][1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%add_323, [32, 1024, 1]), kwargs = {})
#   %div_67 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_271, 1024), kwargs = {})
#   %slice_scatter_default_10 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_12, %div_67, 1, 1024, 2048), kwargs = {})
#   %add_324 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %slice_scatter_default_10), kwargs = {})
#   %add_438 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_61, %sum_62), kwargs = {})
#   %add_439 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_438, %sum_63), kwargs = {})
#   %expand_272 : Tensor "f32[32, 1024, 1][1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%add_439, [32, 1024, 1]), kwargs = {})
#   %div_92 : Tensor "f32[32, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_272, 1024), kwargs = {})
#   %slice_scatter_default_18 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_12, %div_92, 1, 0, 1024), kwargs = {})
#   %add_440 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_324, %slice_scatter_default_18), kwargs = {})
#   return %add_440
triton_poi_fused_add_div_expand_slice_backward_37 = async_compile.triton('triton_poi_fused_add_div_expand_slice_backward_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_expand_slice_backward_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_expand_slice_backward_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    tmp6 = tl.load(in_ptr0 + (x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = 0.0009765625
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.full([1], 1024, tl.int64)
    tmp18 = tmp0 >= tmp17
    tmp19 = tmp0 < tmp1
    tmp20 = tmp18 & tmp19
    tmp21 = tl.load(in_ptr3 + (x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr4 + (x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.load(in_ptr5 + (x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0009765625
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp20, tmp27, tmp28)
    tmp30 = tl.where(tmp20, tmp29, tmp15)
    tmp31 = tmp16 + tmp30
    tmp32 = tmp0 < tmp17
    tmp33 = tl.load(in_ptr6 + (x1), tmp32, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr7 + (x1), tmp32, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.load(in_ptr8 + (x1), tmp32, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0009765625
    tmp39 = tmp37 * tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp32, tmp39, tmp40)
    tmp42 = tl.where(tmp32, tmp41, tmp15)
    tmp43 = tmp31 + tmp42
    tl.store(out_ptr0 + (x2), tmp43, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/u5/cu5fftflhzahgypkrzipaozcgo3dzapog7372mun57luswjsyjyd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.slice_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_167 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_167]
#   %bmm_169 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_169]
#   %bmm_275 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_275]
#   %bmm_277 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_277]
#   %bmm_383 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_383]
#   %bmm_385 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_385]
#   %add_345 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0" = PlaceHolder[target=add_345]
#   %bmm_491 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_491]
#   %bmm_493 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_493]
#   %convert_element_type_555 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_167, torch.float32), kwargs = {})
#   %convert_element_type_561 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_169, torch.float32), kwargs = {})
#   %add_130 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_555, %convert_element_type_561), kwargs = {})
#   %full_default_5 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 4096], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_1 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_5, %add_130, 2, 3072, 4096), kwargs = {})
#   %convert_element_type_860 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_275, torch.float32), kwargs = {})
#   %convert_element_type_866 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_277, torch.float32), kwargs = {})
#   %add_230 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_860, %convert_element_type_866), kwargs = {})
#   %slice_scatter_default_7 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_5, %add_230, 2, 2048, 3072), kwargs = {})
#   %add_231 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_1, %slice_scatter_default_7), kwargs = {})
#   %convert_element_type_1165 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_383, torch.float32), kwargs = {})
#   %convert_element_type_1171 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_385, torch.float32), kwargs = {})
#   %add_341 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1165, %convert_element_type_1171), kwargs = {})
#   %slice_scatter_default_15 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_5, %add_341, 2, 1024, 2048), kwargs = {})
#   %add_345 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_231, %slice_scatter_default_15), kwargs = {})
#   %convert_element_type_1469 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_491, torch.float32), kwargs = {})
#   %convert_element_type_1474 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_493, torch.float32), kwargs = {})
#   %add_460 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1469, %convert_element_type_1474), kwargs = {})
#   %slice_scatter_default_22 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_5, %add_460, 2, 0, 1024), kwargs = {})
#   %add_464 : Tensor "f32[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_345, %slice_scatter_default_22), kwargs = {})
#   return %add_345,%add_464
triton_poi_fused__to_copy_add_slice_backward_38 = async_compile.triton('triton_poi_fused__to_copy_add_slice_backward_38', '''
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/pr/cprvmdoqv624i75455i5lftmkspxf7mpgasdfkmonfpi2v47r7nz.py
# Topologically Sorted Source Nodes: [silu_1, lr1i, dgate, sigma, mul_4, sub, mul_5, add], Original ATen: [aten.fill, aten._to_copy, aten.silu, aten.slice, aten.mul, aten.transpose, aten.sigmoid, aten.rsub, aten.add, aten.neg, aten.sigmoid_backward, aten.sub]
# Source node to ATen node mapping:
#   add => add
#   dgate => mul_6
#   lr1i => slice_4
#   mul_4 => mul_7
#   mul_5 => mul_8
#   sigma => sigmoid_3
#   silu_1 => convert_element_type_19, convert_element_type_20, mul_2, sigmoid_1
#   sub => sub
# Graph fragment:
#   %bmm_480 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_480]
#   %bmm_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %bmm_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %bmm_478 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_478]
#   %bmm_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_481 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_481]
#   %primals_8 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %add_446 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=add_446]
#   %full_default_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 1024], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_1440 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_481, torch.float32), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_2 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %slice_4 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_8, 1, 0, 1024), kwargs = {})
#   %mul_470 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1440, %slice_4), kwargs = {})
#   %convert_element_type_1441 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_470, torch.bfloat16), kwargs = {})
#   %permute_440 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1441, [0, 2, 1]), kwargs = {})
#   %mul_6 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_5, %bmm_4), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_3,), kwargs = {})
#   %mul_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %sigmoid_3), kwargs = {})
#   %mul_471 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_480, %mul_7), kwargs = {})
#   %sub : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_3), kwargs = {})
#   %mul_8 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, %sub), kwargs = {})
#   %add : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1), kwargs = {})
#   %mul_472 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_480, %add), kwargs = {})
#   %mul_473 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_471, %bmm_3), kwargs = {})
#   %mul_474 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_471, %sub), kwargs = {})
#   %neg_20 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_473,), kwargs = {})
#   %mul_475 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_472, %mul_6), kwargs = {})
#   %mul_476 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_472, %sigmoid_3), kwargs = {})
#   %add_442 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_20, %mul_475), kwargs = {})
#   %convert_element_type_1442 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_442, torch.float32), kwargs = {})
#   %convert_element_type_1443 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_3, torch.float32), kwargs = {})
#   %sub_12 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_1443), kwargs = {})
#   %mul_477 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1443, %sub_12), kwargs = {})
#   %mul_478 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1442, %mul_477), kwargs = {})
#   %convert_element_type_1444 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_478, torch.bfloat16), kwargs = {})
#   %add_443 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_474, %convert_element_type_1444), kwargs = {})
#   %mul_479 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, %bmm_5), kwargs = {})
#   %mul_480 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, %bmm_4), kwargs = {})
#   %mul_481 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_478, %bmm_5), kwargs = {})
#   %mul_482 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_478, %convert_element_type_20), kwargs = {})
#   %add_444 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_480, %mul_482), kwargs = {})
#   %sub_13 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_4, %sigmoid_3), kwargs = {})
#   %mul_483 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, %sub_13), kwargs = {})
#   %add_445 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_483, 1), kwargs = {})
#   %mul_484 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_3, %add_445), kwargs = {})
#   %mul_485 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %mul_484), kwargs = {})
#   %add_446 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_443, %mul_485), kwargs = {})
#   %mul_486 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_440, %convert_element_type_20), kwargs = {})
#   %mul_487 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_440, %bmm_4), kwargs = {})
#   %add_449 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_479, %mul_486), kwargs = {})
#   %mul_490 : Tensor "bf16[32, 192, 1024][196608, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_487, %mul_484), kwargs = {})
#   %add_451 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_446, %mul_490), kwargs = {})
#   return %add_444,%add_446,%add_449,%add_451
triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_39 = async_compile.triton('triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_39', '''
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/yn/cynlfi7qf4qi4k2hfecmjzmuum6zkwp4z2f6lmnafawwug52amwj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.masked_fill, aten.add, aten._to_copy, aten.transpose, aten.div, aten.eq, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_353 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_353]
#   %bmm_484 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_484]
#   %bmm_490 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_490]
#   %sum_24 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_24]
#   %sum_39 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_39]
#   %sum_54 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_54]
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_4]
#   %primals_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_235 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_24, %sum_39), kwargs = {})
#   %add_351 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_235, %sum_54), kwargs = {})
#   %convert_element_type_1449 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_484, torch.float32), kwargs = {})
#   %permute_444 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1449, [0, 2, 1]), kwargs = {})
#   %add_448 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_353, %permute_444), kwargs = {})
#   %convert_element_type_1464 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_490, torch.float32), kwargs = {})
#   %add_456 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_448, %convert_element_type_1464), kwargs = {})
#   %div_94 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %pow_4), kwargs = {})
#   %eq_19 : Tensor "b8[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_4, 0), kwargs = {})
#   %where_19 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_19, %full_default_6, %div_94), kwargs = {})
#   %mul_497 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_351, %where_19), kwargs = {})
#   %add_468 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_456, %mul_497), kwargs = {})
#   return %add_468
triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_40 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2457600, 'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + 192*x2 + 36864*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x2 + 192*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (y3), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y3), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y3), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2 + 192*y3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = 0.0
    tmp14 = tmp12 == tmp13
    tmp16 = (tmp15 / tmp12)
    tmp17 = tl.where(tmp14, tmp13, tmp16)
    tmp18 = tmp11 * tmp17
    tmp19 = tmp6 + tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/n7/cn7ywbpawdkwzqpi6uw3jzd6x6jrlxypchxtrgfcmpv4qf7metle.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_266 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_266]
#   %bmm_267 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_267]
#   %bmm_374 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_374]
#   %bmm_375 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_375]
#   %bmm_482 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_482]
#   %bmm_483 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_483]
#   %add_220 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_266, %bmm_267), kwargs = {})
#   %full_default_21 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 4096], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_8 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_21, %add_220, 2, 2048, 3072), kwargs = {})
#   %add_331 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_374, %bmm_375), kwargs = {})
#   %slice_scatter_default_16 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_21, %add_331, 2, 1024, 2048), kwargs = {})
#   %add_346 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_8, %slice_scatter_default_16), kwargs = {})
#   %add_447 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_482, %bmm_483), kwargs = {})
#   %slice_scatter_default_23 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_21, %add_447, 2, 0, 1024), kwargs = {})
#   %add_465 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, %slice_scatter_default_23), kwargs = {})
#   return %add_465
triton_poi_fused_add_slice_backward_41 = async_compile.triton('triton_poi_fused_add_slice_backward_41', '''
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/gw/cgwb5dllk2zgksdlaprjmfxsakzvhptahlvljhxr3buy3m4g4chv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.masked_fill, aten.add, aten.div, aten.eq, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_1172 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_1172]
#   %bmm_486 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_486]
#   %bmm_494 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_494]
#   %sum_22 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_22]
#   %sum_37 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_37]
#   %sum_52 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_52]
#   %pow_6 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_6]
#   %primals_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_232 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_22, %sum_37), kwargs = {})
#   %add_348 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %sum_52), kwargs = {})
#   %add_452 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1172, %bmm_486), kwargs = {})
#   %add_459 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_452, %bmm_494), kwargs = {})
#   %div_93 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_3, %pow_6), kwargs = {})
#   %eq_18 : Tensor "b8[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_6, 0), kwargs = {})
#   %where_18 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_18, %full_default_6, %div_93), kwargs = {})
#   %mul_496 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_348, %where_18), kwargs = {})
#   %convert_element_type_1475 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_496, torch.bfloat16), kwargs = {})
#   %add_467 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_459, %convert_element_type_1475), kwargs = {})
#   return %add_467
triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_42 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 98304, 'x': 14155776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_out_ptr0 + (x1 + 192*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1 + 192*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x1 + 192*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1 + 192*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = 0.0
    tmp12 = tmp10 == tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp14 / tmp10)
    tmp16 = tl.where(tmp12, tmp11, tmp15)
    tmp17 = tmp9 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp4 + tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 192*y0), tmp19, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/jf/cjf2rymzbzb3gfpmrbchguz4wgt47u5kfubzlu5to7svh6sy6zpl.py
# Topologically Sorted Source Nodes: [lr2i_2, lr0i_2, lr2i_1, lr0i_1, lr2i, lr0i], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.add, aten.transpose, aten.slice_backward]
# Source node to ATen node mapping:
#   lr0i => slice_6
#   lr0i_1 => slice_16
#   lr0i_2 => slice_27
#   lr2i => slice_5
#   lr2i_1 => slice_15
#   lr2i_2 => slice_26
# Graph fragment:
#   %bmm_261 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_261]
#   %primals_9 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %bmm_263 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_263]
#   %primals_10 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %bmm_269 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_269]
#   %bmm_271 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_271]
#   %bmm_369 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_369]
#   %bmm_371 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_371]
#   %bmm_377 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_377]
#   %bmm_379 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_379]
#   %add_347 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=add_347]
#   %bmm_477 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_477]
#   %bmm_479 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0" = PlaceHolder[target=bmm_479]
#   %bmm_485 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_485]
#   %bmm_487 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_487]
#   %convert_element_type_818 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_261, torch.float32), kwargs = {})
#   %slice_26 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_9, 1, 2048, 3072), kwargs = {})
#   %mul_269 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_818, %slice_26), kwargs = {})
#   %convert_element_type_823 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_263, torch.float32), kwargs = {})
#   %slice_27 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_10, 1, 2048, 3072), kwargs = {})
#   %mul_271 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_823, %slice_27), kwargs = {})
#   %add_214 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %mul_271), kwargs = {})
#   %convert_element_type_843 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_269, torch.float32), kwargs = {})
#   %permute_188 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_843, [0, 2, 1]), kwargs = {})
#   %add_224 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, %permute_188), kwargs = {})
#   %convert_element_type_849 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_271, torch.float32), kwargs = {})
#   %permute_191 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_849, [0, 2, 1]), kwargs = {})
#   %add_225 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_224, %permute_191), kwargs = {})
#   %full_default_22 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_9 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %add_225, 1, 2048, 3072), kwargs = {})
#   %convert_element_type_1123 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_369, torch.float32), kwargs = {})
#   %slice_15 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_9, 1, 1024, 2048), kwargs = {})
#   %mul_369 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1123, %slice_15), kwargs = {})
#   %convert_element_type_1128 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_371, torch.float32), kwargs = {})
#   %slice_16 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_10, 1, 1024, 2048), kwargs = {})
#   %mul_371 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1128, %slice_16), kwargs = {})
#   %add_325 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_369, %mul_371), kwargs = {})
#   %convert_element_type_1148 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_377, torch.float32), kwargs = {})
#   %permute_318 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1148, [0, 2, 1]), kwargs = {})
#   %add_335 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_325, %permute_318), kwargs = {})
#   %convert_element_type_1154 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_379, torch.float32), kwargs = {})
#   %permute_321 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1154, [0, 2, 1]), kwargs = {})
#   %add_336 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_335, %permute_321), kwargs = {})
#   %slice_scatter_default_17 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %add_336, 1, 1024, 2048), kwargs = {})
#   %add_347 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_9, %slice_scatter_default_17), kwargs = {})
#   %convert_element_type_1430 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_477, torch.float32), kwargs = {})
#   %slice_5 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_9, 1, 0, 1024), kwargs = {})
#   %mul_466 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1430, %slice_5), kwargs = {})
#   %convert_element_type_1435 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_479, torch.float32), kwargs = {})
#   %slice_6 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_10, 1, 0, 1024), kwargs = {})
#   %mul_468 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1435, %slice_6), kwargs = {})
#   %add_441 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_466, %mul_468), kwargs = {})
#   %convert_element_type_1454 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_485, torch.float32), kwargs = {})
#   %permute_448 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1454, [0, 2, 1]), kwargs = {})
#   %add_453 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_441, %permute_448), kwargs = {})
#   %convert_element_type_1459 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_487, torch.float32), kwargs = {})
#   %permute_451 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1459, [0, 2, 1]), kwargs = {})
#   %add_455 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_453, %permute_451), kwargs = {})
#   %slice_scatter_default_24 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %add_455, 1, 0, 1024), kwargs = {})
#   %add_466 : Tensor "f32[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_347, %slice_scatter_default_24), kwargs = {})
#   return %add_347,%add_466
triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_43 = async_compile.triton('triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 301989888, 'x': 503316480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 192
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
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
    tmp6 = tl.load(in_ptr0 + ((-393216) + x2 + 192*y0 + 196608*y1), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.load(in_ptr2 + ((-393216) + x2 + 192*y0 + 196608*y1), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 + tmp13
    tmp15 = tl.load(in_ptr4 + ((-2048) + y0 + 1024*x2 + 196608*y1), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tl.load(in_ptr5 + ((-2048) + y0 + 1024*x2 + 196608*y1), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp29 = tl.load(in_ptr6 + ((-196608) + x2 + 192*y0 + 196608*y1), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr7 + ((-196608) + x2 + 192*y0 + 196608*y1), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.load(in_ptr3 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 + tmp36
    tmp38 = tl.load(in_ptr8 + ((-1024) + y0 + 1024*x2 + 196608*y1), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 + tmp39
    tmp41 = tl.load(in_ptr9 + ((-1024) + y0 + 1024*x2 + 196608*y1), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tl.where(tmp28, tmp45, tmp23)
    tmp47 = tmp24 + tmp46
    tmp48 = tmp0 < tmp25
    tmp49 = tl.load(in_ptr10 + (x2 + 192*y0 + 196608*y1), tmp48 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.load(in_ptr1 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp48 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 * tmp51
    tmp53 = tl.load(in_ptr11 + (x2 + 192*y0 + 196608*y1), tmp48 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.load(in_ptr3 + (tl.broadcast_to(3*y3, [YBLOCK, XBLOCK])), tmp48 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp52 + tmp56
    tmp58 = tl.load(in_ptr12 + (y0 + 1024*x2 + 196608*y1), tmp48 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp57 + tmp59
    tmp61 = tl.load(in_ptr13 + (y0 + 1024*x2 + 196608*y1), tmp48 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp60 + tmp62
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp48, tmp63, tmp64)
    tmp66 = tl.where(tmp48, tmp65, tmp23)
    tmp67 = tmp47 + tmp66
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp67, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_7, primals_8, primals_9, primals_10, pow_2, pow_4, pow_6, bmm, bmm_1, bmm_3, bmm_4, bmm_5, bmm_6, bmm_7, bmm_8, mean, pow_8, pow_10, convert_element_type_129, pow_12, add_37, add_38, add_39, pow_14, pow_16, pow_18, bmm_54, bmm_55, bmm_57, bmm_58, bmm_59, mean_1, add_44, add_45, add_46, pow_20, pow_22, convert_element_type_307, pow_24, add_80, add_81, add_82, pow_26, pow_28, pow_30, bmm_108, bmm_109, bmm_111, bmm_112, bmm_113, mean_2, convert_element_type_391, pow_32, convert_element_type_438, pow_34, convert_element_type_485, pow_36, add_123, add_124, add_125, pow_38, pow_40, pow_42, bmm_162, bmm_163, permute_62, permute_63, permute_65, permute_66, permute_67, permute_69, permute_70, permute_71, permute_72, permute_74, permute_76, permute_77, permute_78, permute_79, permute_81, permute_83, permute_84, permute_85, permute_86, permute_88, permute_90, permute_91, permute_92, permute_93, permute_95, permute_97, permute_98, permute_99, permute_100, permute_102, permute_104, permute_105, permute_106, permute_107, permute_109, permute_111, permute_112, permute_113, permute_114, permute_116, permute_118, permute_119, permute_120, permute_121, permute_123, permute_125, permute_126, permute_127, permute_128, permute_130, permute_132, permute_133, permute_134, permute_135, permute_137, permute_139, permute_140, permute_141, permute_142, permute_144, permute_146, permute_147, permute_148, permute_149, permute_151, permute_153, permute_154, permute_155, permute_156, permute_158, permute_160, permute_161, permute_162, permute_163, permute_165, permute_167, permute_168, permute_169, permute_170, permute_172, permute_174, permute_175, permute_176, permute_177, permute_178, permute_179, permute_182, permute_186, permute_187, permute_189, permute_192, permute_193, permute_196, permute_199, permute_200, permute_201, permute_202, permute_204, permute_206, permute_207, permute_208, permute_209, permute_211, permute_213, permute_214, permute_215, permute_216, permute_218, permute_220, permute_221, permute_222, permute_223, permute_225, permute_227, permute_228, permute_229, permute_230, permute_232, permute_234, permute_235, permute_236, permute_237, permute_239, permute_241, permute_242, permute_243, permute_244, permute_246, permute_248, permute_249, permute_250, permute_251, permute_253, permute_255, permute_256, permute_257, permute_258, permute_260, permute_262, permute_263, permute_264, permute_265, permute_267, permute_269, permute_270, permute_271, permute_272, permute_274, permute_276, permute_277, permute_278, permute_279, permute_281, permute_283, permute_284, permute_285, permute_286, permute_288, permute_290, permute_291, permute_292, permute_293, permute_295, permute_297, permute_298, permute_299, permute_300, permute_302, permute_304, permute_305, permute_306, permute_307, permute_308, permute_309, permute_312, permute_316, permute_317, permute_319, permute_322, permute_323, permute_326, permute_329, permute_330, permute_331, permute_332, permute_334, permute_336, permute_337, permute_338, permute_339, permute_341, permute_343, permute_344, permute_345, permute_346, permute_348, permute_350, permute_351, permute_352, permute_353, permute_355, permute_357, permute_358, permute_359, permute_360, permute_362, permute_364, permute_365, permute_366, permute_367, permute_369, permute_371, permute_372, permute_373, permute_374, permute_376, permute_378, permute_379, permute_380, permute_381, permute_383, permute_385, permute_386, permute_387, permute_388, permute_390, permute_392, permute_393, permute_394, permute_395, permute_397, permute_399, permute_400, permute_401, permute_402, permute_404, permute_406, permute_407, permute_408, permute_409, permute_411, permute_413, permute_414, permute_415, permute_416, permute_418, permute_420, permute_421, permute_422, permute_423, permute_425, permute_427, permute_428, permute_429, permute_430, permute_432, permute_434, permute_435, permute_436, permute_437, permute_438, permute_439, permute_442, permute_447, permute_452, permute_453, permute_456, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(primals_2, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(primals_3, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(primals_7, (32, 4096, 192), (786432, 192, 1))
        assert_size_stride(primals_8, (32, 4096, 1), (12288, 3, 1))
        assert_size_stride(primals_9, (32, 4096, 1), (12288, 3, 1))
        assert_size_stride(primals_10, (32, 4096, 1), (12288, 3, 1))
        assert_size_stride(pow_2, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_4, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_6, (32, 192, 1), (192, 1, 1))
        assert_size_stride(bmm, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_1, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_3, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_4, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_5, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_6, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(bmm_7, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(bmm_8, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(mean, (32, 1, 1), (1, 1, 1))
        assert_size_stride(pow_8, (32, 1, 1), (1, 1, 1))
        assert_size_stride(pow_10, (32, 1, 1), (1, 1, 1))
        assert_size_stride(convert_element_type_129, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_12, (32, 1, 1), (1, 1, 1))
        assert_size_stride(add_37, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_38, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_39, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_14, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_16, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_18, (32, 192, 1), (192, 1, 1))
        assert_size_stride(bmm_54, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_55, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_57, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_58, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_59, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(mean_1, (32, 1, 1), (1, 1, 1))
        assert_size_stride(add_44, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_45, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_46, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_20, (32, 1, 1), (1, 1, 1))
        assert_size_stride(pow_22, (32, 1, 1), (1, 1, 1))
        assert_size_stride(convert_element_type_307, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_24, (32, 1, 1), (1, 1, 1))
        assert_size_stride(add_80, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_81, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_82, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_26, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_28, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_30, (32, 192, 1), (192, 1, 1))
        assert_size_stride(bmm_108, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_109, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_111, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_112, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_113, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(mean_2, (32, 1, 1), (1, 1, 1))
        assert_size_stride(convert_element_type_391, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_32, (32, 1, 1), (1, 1, 1))
        assert_size_stride(convert_element_type_438, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_34, (32, 1, 1), (1, 1, 1))
        assert_size_stride(convert_element_type_485, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_36, (32, 1, 1), (1, 1, 1))
        assert_size_stride(add_123, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_124, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(add_125, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(pow_38, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_40, (32, 192, 1), (192, 1, 1))
        assert_size_stride(pow_42, (32, 192, 1), (192, 1, 1))
        assert_size_stride(bmm_162, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(bmm_163, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(permute_62, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_63, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_65, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_66, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(permute_67, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_69, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_70, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_71, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_72, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_74, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_76, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_77, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_78, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_79, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_81, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_83, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_84, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_85, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_86, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_88, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_90, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_91, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_92, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_93, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_95, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_97, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_98, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_99, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_100, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_102, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_104, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_105, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_106, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_107, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_109, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_111, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_112, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_113, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_114, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_116, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_118, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_119, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_120, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_121, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_123, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_125, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_126, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_127, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_128, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_130, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_132, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_133, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_134, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_135, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_137, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_139, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_140, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_141, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_142, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_144, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_146, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_147, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_148, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_149, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_151, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_153, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_154, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_155, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_156, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_158, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_160, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_161, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_162, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_163, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_165, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_167, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_168, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_169, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_170, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_172, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_174, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_175, (32, 192, 1024), (196608, 1, 192))
        assert_size_stride(permute_176, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_177, (32, 192, 1024), (196608, 1, 192))
        assert_size_stride(permute_178, (32, 1024, 192), (786432, 192, 1))
        assert_size_stride(permute_179, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(permute_182, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_186, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_187, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(permute_189, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_192, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_193, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_196, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(permute_199, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_200, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_201, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_202, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_204, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_206, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_207, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_208, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_209, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_211, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_213, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_214, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_215, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_216, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_218, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_220, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_221, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_222, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_223, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_225, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_227, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_228, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_229, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_230, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_232, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_234, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_235, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_236, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_237, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_239, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_241, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_242, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_243, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_244, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_246, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_248, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_249, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_250, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_251, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_253, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_255, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_256, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_257, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_258, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_260, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_262, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_263, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_264, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_265, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_267, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_269, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_270, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_271, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_272, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_274, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_276, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_277, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_278, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_279, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_281, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_283, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_284, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_285, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_286, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_288, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_290, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_291, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_292, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_293, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_295, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_297, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_298, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_299, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_300, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_302, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_304, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_305, (32, 192, 1024), (196608, 1, 192))
        assert_size_stride(permute_306, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_307, (32, 192, 1024), (196608, 1, 192))
        assert_size_stride(permute_308, (32, 1024, 192), (786432, 192, 1))
        assert_size_stride(permute_309, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(permute_312, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_316, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_317, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(permute_319, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_322, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_323, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_326, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(permute_329, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_330, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_331, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_332, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_334, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_336, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_337, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_338, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_339, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_341, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_343, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_344, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_345, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_346, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_348, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_350, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_351, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_352, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_353, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_355, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_357, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_358, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_359, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_360, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_362, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_364, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_365, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_366, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_367, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_369, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_371, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_372, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_373, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_374, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_376, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_378, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_379, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_380, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_381, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_383, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_385, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_386, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_387, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_388, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_390, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_392, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_393, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_394, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_395, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_397, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_399, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_400, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_401, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_402, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_404, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_406, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_407, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_408, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_409, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_411, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_413, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_414, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_415, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_416, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_418, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_420, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_421, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_422, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_423, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_425, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_427, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_428, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_429, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_430, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_432, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_434, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_435, (32, 192, 1024), (196608, 1, 192))
        assert_size_stride(permute_436, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_437, (32, 192, 1024), (196608, 1, 192))
        assert_size_stride(permute_438, (32, 1024, 192), (786432, 192, 1))
        assert_size_stride(permute_439, (32, 192, 1024), (196608, 1024, 1))
        assert_size_stride(permute_442, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(permute_447, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(permute_452, (32, 192, 192), (36864, 1, 192))
        assert_size_stride(permute_453, (32, 1024, 192), (196608, 1, 1024))
        assert_size_stride(permute_456, (32, 1024, 192), (196608, 192, 1))
        assert_size_stride(tangents_1, (32, 4096, 192), (786432, 192, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((32, 192, 1024), (196608, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_transpose_0.run(tangents_1, buf0, 6291456, stream=stream0)
            buf2 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf0, permute_63, out=buf2)
            del permute_63
            buf1 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.clone, aten.bmm]
            extern_kernels.bmm(permute_62, buf0, out=buf1)
            del buf0
            del permute_62
            buf12 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf14 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf108 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_127, truediv_16], Original ATen: [aten._to_copy, aten.masked_fill, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1.run(buf2, add_123, pow_40, pow_4, buf12, buf14, buf108, 6144, 192, stream=stream0)
            del add_123
            del pow_40
            buf109 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_139, buf108, out=buf109)
            del permute_139
            buf110 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf108, permute_140, out=buf110)
            buf111 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_141, buf110, out=buf111)
            del permute_141
            buf112 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf110, permute_142, out=buf112)
            del permute_142
            buf113 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf113, buf112, buf110, 1179648, stream=stream0)
            buf114 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_140, buf113, out=buf114)
            del permute_140
            buf115 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf113, permute_144, out=buf115)
            del permute_144
            buf116 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf109, buf14, buf115, buf114, buf116, 6144, 192, stream=stream0)
            buf117 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_146, buf116, out=buf117)
            del permute_146
            buf118 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf116, permute_147, out=buf118)
            buf119 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_148, buf118, out=buf119)
            del permute_148
            buf120 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf118, permute_149, out=buf120)
            del permute_149
            buf121 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf121, buf120, buf118, 1179648, stream=stream0)
            buf122 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_147, buf121, out=buf122)
            del permute_147
            buf123 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf121, permute_151, out=buf123)
            del buf121
            del permute_151
            buf3 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf4 = bmm_162; del bmm_162  # reuse
            # Topologically Sorted Source Nodes: [gate_3], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5.run(buf4, buf1, bmm_163, buf3, 6291456, stream=stream0)
            del bmm_163
            buf124 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf125 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf117, buf109, buf14, buf115, buf114, buf123, buf122, buf124, buf125, 1179648, stream=stream0)
            buf126 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_153, buf125, out=buf126)
            del permute_153
            buf127 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf125, permute_154, out=buf127)
            buf128 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_155, buf127, out=buf128)
            del permute_155
            buf129 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf127, permute_156, out=buf129)
            del permute_156
            buf130 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf130, buf129, buf127, 1179648, stream=stream0)
            buf131 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_154, buf130, out=buf131)
            del permute_154
            buf132 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf130, permute_158, out=buf132)
            del permute_158
            buf6 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf4, permute_66, out=buf6)
            buf8 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf3, permute_66, out=buf8)
            del permute_66
            buf5 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_65, buf4, out=buf5)
            del permute_65
            buf7 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_67, buf3, out=buf7)
            del permute_67
            buf9 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf11 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf18 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [add_128, truediv_17], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq, aten.masked_fill]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1.run(buf8, add_125, pow_42, pow_6, buf9, buf11, buf18, 6144, 192, stream=stream0)
            del add_125
            del pow_42
            buf15 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf17 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf63 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [add_126, truediv_15], Original ATen: [aten._to_copy, aten.masked_fill, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_1.run(buf6, add_124, pow_38, pow_2, buf15, buf17, buf63, 6144, 192, stream=stream0)
            del add_124
            del pow_38
            buf19 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_69, buf18, out=buf19)
            del permute_69
            buf20 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf18, permute_70, out=buf20)
            buf21 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_71, buf20, out=buf21)
            del permute_71
            buf22 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf20, permute_72, out=buf22)
            del permute_72
            buf23 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf23, buf22, buf20, 1179648, stream=stream0)
            buf24 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_70, buf23, out=buf24)
            del permute_70
            buf25 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf23, permute_74, out=buf25)
            del permute_74
            buf64 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_104, buf63, out=buf64)
            del permute_104
            buf65 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf63, permute_105, out=buf65)
            buf66 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_106, buf65, out=buf66)
            del permute_106
            buf67 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf65, permute_107, out=buf67)
            del permute_107
            buf68 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf68, buf67, buf65, 1179648, stream=stream0)
            buf69 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_105, buf68, out=buf69)
            del permute_105
            buf70 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf68, permute_109, out=buf70)
            del permute_109
            buf26 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf19, buf11, buf25, buf24, buf26, 6144, 192, stream=stream0)
            buf27 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_76, buf26, out=buf27)
            del permute_76
            buf28 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf26, permute_77, out=buf28)
            buf29 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_78, buf28, out=buf29)
            del permute_78
            buf30 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf28, permute_79, out=buf30)
            del permute_79
            buf31 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf31, buf30, buf28, 1179648, stream=stream0)
            buf32 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_77, buf31, out=buf32)
            del permute_77
            buf33 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf31, permute_81, out=buf33)
            del permute_81
            buf34 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf35 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf27, buf19, buf11, buf25, buf24, buf33, buf32, buf34, buf35, 1179648, stream=stream0)
            buf36 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_83, buf35, out=buf36)
            del permute_83
            buf37 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf35, permute_84, out=buf37)
            buf38 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_85, buf37, out=buf38)
            del permute_85
            buf39 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf37, permute_86, out=buf39)
            del permute_86
            buf40 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf40, buf39, buf37, 1179648, stream=stream0)
            buf41 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_84, buf40, out=buf41)
            del permute_84
            buf42 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf40, permute_88, out=buf42)
            del permute_88
            buf43 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf36, buf34, buf42, buf41, buf43, 6144, 192, stream=stream0)
            buf44 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_90, buf43, out=buf44)
            del permute_90
            buf45 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf43, permute_91, out=buf45)
            buf46 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_92, buf45, out=buf46)
            del permute_92
            buf47 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf45, permute_93, out=buf47)
            del permute_93
            buf48 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf48, buf47, buf45, 1179648, stream=stream0)
            buf49 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_91, buf48, out=buf49)
            del permute_91
            buf50 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf48, permute_95, out=buf50)
            del permute_95
            buf51 = buf34; del buf34  # reuse
            buf52 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf51, buf44, buf36, buf42, buf41, buf50, buf49, buf52, 1179648, stream=stream0)
            buf53 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_97, buf52, out=buf53)
            del permute_97
            buf54 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf52, permute_98, out=buf54)
            buf55 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_99, buf54, out=buf55)
            del permute_99
            buf56 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf54, permute_100, out=buf56)
            del permute_100
            buf57 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf57, buf56, buf54, 1179648, stream=stream0)
            buf58 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_98, buf57, out=buf58)
            del permute_98
            buf59 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf57, permute_102, out=buf59)
            del permute_102
            buf60 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_112, X_57], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12.run(buf53, buf51, buf59, buf58, convert_element_type_485, pow_36, buf60, 160, 7373, stream=stream0)
            buf61 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [add_112, X_57], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf60, buf61, 32, 5, stream=stream0)
            buf62 = buf51; del buf51  # reuse
            buf159 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [add_112], Original ATen: [aten.masked_fill, aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14.run(buf62, buf53, buf59, buf58, pow_36, buf61, convert_element_type_485, buf159, 1179648, stream=stream0)
            del convert_element_type_485
            del pow_36
            buf153 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_15.run(buf62, add_46, buf153, 160, 7373, stream=stream0)
            del add_46
            buf154 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf153, buf154, 32, 5, stream=stream0)
            buf160 = reinterpret_tensor(buf3, (32, 1024, 192), (196608, 192, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_174, buf159, out=buf160)
            del permute_174
            buf161 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf159, permute_175, out=buf161)
            del permute_175
            buf71 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf64, buf17, buf70, buf69, buf71, 6144, 192, stream=stream0)
            buf72 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_111, buf71, out=buf72)
            del permute_111
            buf73 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf71, permute_112, out=buf73)
            buf74 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_113, buf73, out=buf74)
            del permute_113
            buf75 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf73, permute_114, out=buf75)
            del permute_114
            buf76 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf76, buf75, buf73, 1179648, stream=stream0)
            buf77 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_112, buf76, out=buf77)
            del permute_112
            buf78 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf76, permute_116, out=buf78)
            del permute_116
            buf79 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf80 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf72, buf64, buf17, buf70, buf69, buf78, buf77, buf79, buf80, 1179648, stream=stream0)
            buf81 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_118, buf80, out=buf81)
            del permute_118
            buf82 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf80, permute_119, out=buf82)
            buf83 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_120, buf82, out=buf83)
            del permute_120
            buf84 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf82, permute_121, out=buf84)
            del permute_121
            buf85 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf85, buf84, buf82, 1179648, stream=stream0)
            buf86 = buf84; del buf84  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_119, buf85, out=buf86)
            del permute_119
            buf87 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf85, permute_123, out=buf87)
            del permute_123
            buf88 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf81, buf79, buf87, buf86, buf88, 6144, 192, stream=stream0)
            buf89 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_125, buf88, out=buf89)
            del permute_125
            buf90 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf88, permute_126, out=buf90)
            buf91 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_127, buf90, out=buf91)
            del permute_127
            buf92 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf90, permute_128, out=buf92)
            del permute_128
            buf93 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf93, buf92, buf90, 1179648, stream=stream0)
            buf94 = buf92; del buf92  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_126, buf93, out=buf94)
            del permute_126
            buf95 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf93, permute_130, out=buf95)
            del permute_130
            buf96 = buf79; del buf79  # reuse
            buf97 = buf93; del buf93  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf96, buf89, buf81, buf87, buf86, buf95, buf94, buf97, 1179648, stream=stream0)
            buf98 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_132, buf97, out=buf98)
            del permute_132
            buf99 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf97, permute_133, out=buf99)
            buf100 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_134, buf99, out=buf100)
            del permute_134
            buf101 = buf89; del buf89  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf99, permute_135, out=buf101)
            del permute_135
            buf102 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf102, buf101, buf99, 1179648, stream=stream0)
            buf103 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf102, out=buf103)
            del permute_133
            buf104 = buf101; del buf101  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf102, permute_137, out=buf104)
            del permute_137
            buf105 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [add_101, X_50], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12.run(buf98, buf96, buf104, buf103, convert_element_type_438, pow_34, buf105, 160, 7373, stream=stream0)
            buf106 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [add_101, X_50], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf105, buf106, 32, 5, stream=stream0)
            buf107 = buf96; del buf96  # reuse
            buf163 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [add_101], Original ATen: [aten.masked_fill, aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14.run(buf107, buf98, buf104, buf103, pow_34, buf106, convert_element_type_438, buf163, 1179648, stream=stream0)
            del convert_element_type_438
            del pow_34
            buf164 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_176, buf163, out=buf164)
            del permute_176
            buf165 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf163, permute_177, out=buf165)
            del permute_177
            buf157 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_15.run(buf107, add_44, buf157, 160, 7373, stream=stream0)
            buf158 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf157, buf158, 32, 5, stream=stream0)
            buf162 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            buf166 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [ki_2], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_slice_sum_16.run(buf160, primals_7, buf164, buf162, buf166, 32768, 192, stream=stream0)
            buf133 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf126, buf124, buf132, buf131, buf133, 6144, 192, stream=stream0)
            buf134 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_160, buf133, out=buf134)
            del permute_160
            buf135 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf133, permute_161, out=buf135)
            buf136 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_162, buf135, out=buf136)
            del permute_162
            buf137 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf135, permute_163, out=buf137)
            del permute_163
            buf138 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf138, buf137, buf135, 1179648, stream=stream0)
            buf139 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_161, buf138, out=buf139)
            del permute_161
            buf140 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf138, permute_165, out=buf140)
            del permute_165
            buf141 = buf124; del buf124  # reuse
            buf142 = buf138; del buf138  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf141, buf134, buf126, buf132, buf131, buf140, buf139, buf142, 1179648, stream=stream0)
            buf143 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_167, buf142, out=buf143)
            del permute_167
            buf144 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf142, permute_168, out=buf144)
            buf145 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_169, buf144, out=buf145)
            del permute_169
            buf146 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf144, permute_170, out=buf146)
            del permute_170
            buf147 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf147, buf146, buf144, 1179648, stream=stream0)
            buf148 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_168, buf147, out=buf148)
            del permute_168
            buf149 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf147, permute_172, out=buf149)
            del permute_172
            buf150 = buf157; del buf157  # reuse
            # Topologically Sorted Source Nodes: [add_90, X_43], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12.run(buf143, buf141, buf149, buf148, convert_element_type_391, pow_32, buf150, 160, 7373, stream=stream0)
            buf151 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [add_90, X_43], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf150, buf151, 32, 5, stream=stream0)
            buf152 = buf141; del buf141  # reuse
            buf167 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [add_90], Original ATen: [aten.masked_fill, aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_14.run(buf152, buf143, buf149, buf148, pow_32, buf151, convert_element_type_391, buf167, 1179648, stream=stream0)
            del convert_element_type_391
            del pow_32
            buf169 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf167, permute_179, out=buf169)
            del permute_179
            buf155 = buf150; del buf150  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_15.run(buf152, add_45, buf155, 160, 7373, stream=stream0)
            buf156 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf155, buf156, 32, 5, stream=stream0)
            buf168 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_178, buf167, out=buf168)
            buf171 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf172 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf175 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf176 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [silu_7, lr1i_2, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86], Original ATen: [aten.fill, aten._to_copy, aten.silu, aten.slice, aten.mul, aten.transpose, aten.sigmoid, aten.rsub, aten.add, aten.neg, aten.sigmoid_backward, aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_17.run(buf176, buf165, bmm_111, bmm_112, buf161, bmm_113, buf168, primals_8, buf171, buf175, 6291456, stream=stream0)
            del bmm_113
            buf170 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [silu_7, hidden_2, transpose_43], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.transpose, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_silu_sum_transpose_18.run(buf168, bmm_111, bmm_112, buf170, 32768, 192, stream=stream0)
            del bmm_111
            del bmm_112
            buf174 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf171, permute_178, out=buf174)
            del permute_178
            buf173 = reinterpret_tensor(buf168, (32, 192, 1024), (196608, 1024, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf171, out=buf173)
            del permute_182
            buf178 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf175, permute_187, out=buf178)
            buf180 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf176, permute_187, out=buf180)
            del permute_187
            buf177 = buf171; del buf171  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_186, buf175, out=buf177)
            buf179 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_189, buf176, out=buf179)
            buf181 = reinterpret_tensor(buf176, (32, 192, 1024), (196608, 1, 192), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_copy_slice_transpose_zeros_like_19.run(tangents_1, buf181, 6291456, stream=stream0)
            buf183 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf181, permute_193, out=buf183)
            del permute_193
            buf193 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf195 = buf14; del buf14  # reuse
            buf289 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [add_84, truediv_10], Original ATen: [aten.masked_fill, aten._to_copy, aten.transpose, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20.run(buf195, buf174, buf183, add_80, pow_28, pow_4, buf193, buf289, 6144, 192, stream=stream0)
            del add_80
            del pow_28
            buf182 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone, aten.bmm]
            extern_kernels.bmm(permute_192, buf181, out=buf182)
            del permute_192
            buf184 = reinterpret_tensor(buf181, (32, 192, 1024), (196608, 1024, 1), 0); del buf181  # reuse
            buf185 = bmm_108; del bmm_108  # reuse
            # Topologically Sorted Source Nodes: [gate_2], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5.run(buf185, buf182, bmm_109, buf184, 6291456, stream=stream0)
            del bmm_109
            buf290 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_269, buf289, out=buf290)
            del permute_269
            buf291 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf289, permute_270, out=buf291)
            buf292 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_271, buf291, out=buf292)
            del permute_271
            buf293 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf291, permute_272, out=buf293)
            del permute_272
            buf294 = buf292; del buf292  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf294, buf293, buf291, 1179648, stream=stream0)
            buf295 = buf293; del buf293  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_270, buf294, out=buf295)
            del permute_270
            buf296 = buf291; del buf291  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf294, permute_274, out=buf296)
            del permute_274
            buf187 = buf294; del buf294  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf185, permute_196, out=buf187)
            buf189 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf184, permute_196, out=buf189)
            del permute_196
            buf190 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf192 = buf11; del buf11  # reuse
            buf199 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [add_85, truediv_11], Original ATen: [aten.masked_fill, aten._to_copy, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21.run(buf192, buf178, buf189, add_82, pow_30, pow_6, buf190, buf199, 6144, 192, stream=stream0)
            del add_82
            del pow_30
            buf196 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf198 = buf17; del buf17  # reuse
            buf244 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [add_83, truediv_9], Original ATen: [aten.masked_fill, aten._to_copy, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21.run(buf198, buf180, buf187, add_81, pow_26, pow_2, buf196, buf244, 6144, 192, stream=stream0)
            del add_81
            del pow_26
            buf186 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_189, buf185, out=buf186)
            del permute_189
            buf188 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_186, buf184, out=buf188)
            del permute_186
            buf200 = buf187; del buf187  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_199, buf199, out=buf200)
            del permute_199
            buf201 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf199, permute_200, out=buf201)
            buf202 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_201, buf201, out=buf202)
            del permute_201
            buf203 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf201, permute_202, out=buf203)
            del permute_202
            buf204 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf204, buf203, buf201, 1179648, stream=stream0)
            buf205 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_200, buf204, out=buf205)
            del permute_200
            buf206 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf204, permute_204, out=buf206)
            del permute_204
            buf245 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_234, buf244, out=buf245)
            del permute_234
            buf246 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf244, permute_235, out=buf246)
            buf247 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_236, buf246, out=buf247)
            del permute_236
            buf248 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf246, permute_237, out=buf248)
            del permute_237
            buf249 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf249, buf248, buf246, 1179648, stream=stream0)
            buf250 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_235, buf249, out=buf250)
            del permute_235
            buf251 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf249, permute_239, out=buf251)
            del permute_239
            buf207 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf200, buf192, buf206, buf205, buf207, 6144, 192, stream=stream0)
            buf208 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_206, buf207, out=buf208)
            del permute_206
            buf209 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf207, permute_207, out=buf209)
            buf210 = buf207; del buf207  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_208, buf209, out=buf210)
            del permute_208
            buf211 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf209, permute_209, out=buf211)
            del permute_209
            buf212 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf212, buf211, buf209, 1179648, stream=stream0)
            buf213 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_207, buf212, out=buf213)
            del permute_207
            buf214 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf212, permute_211, out=buf214)
            del permute_211
            buf215 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf216 = buf212; del buf212  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf208, buf200, buf192, buf206, buf205, buf214, buf213, buf215, buf216, 1179648, stream=stream0)
            buf217 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_213, buf216, out=buf217)
            del permute_213
            buf218 = buf213; del buf213  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf216, permute_214, out=buf218)
            buf219 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_215, buf218, out=buf219)
            del permute_215
            buf220 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf218, permute_216, out=buf220)
            del permute_216
            buf221 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf221, buf220, buf218, 1179648, stream=stream0)
            buf222 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_214, buf221, out=buf222)
            del permute_214
            buf223 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf221, permute_218, out=buf223)
            del permute_218
            buf224 = buf221; del buf221  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf217, buf215, buf223, buf222, buf224, 6144, 192, stream=stream0)
            buf225 = buf206; del buf206  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_220, buf224, out=buf225)
            del permute_220
            buf226 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf224, permute_221, out=buf226)
            buf227 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_222, buf226, out=buf227)
            del permute_222
            buf228 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf226, permute_223, out=buf228)
            del permute_223
            buf229 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf229, buf228, buf226, 1179648, stream=stream0)
            buf230 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_221, buf229, out=buf230)
            del permute_221
            buf231 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf229, permute_225, out=buf231)
            del permute_225
            buf232 = buf215; del buf215  # reuse
            buf233 = buf229; del buf229  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf232, buf225, buf217, buf223, buf222, buf231, buf230, buf233, 1179648, stream=stream0)
            buf234 = buf231; del buf231  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_227, buf233, out=buf234)
            del permute_227
            buf235 = buf230; del buf230  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf233, permute_228, out=buf235)
            buf236 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_229, buf235, out=buf236)
            del permute_229
            buf237 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf235, permute_230, out=buf237)
            del permute_230
            buf238 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf238, buf237, buf235, 1179648, stream=stream0)
            buf239 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_228, buf238, out=buf239)
            del permute_228
            buf240 = buf235; del buf235  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf238, permute_232, out=buf240)
            del permute_232
            buf241 = buf155; del buf155  # reuse
            # Topologically Sorted Source Nodes: [add_69, X_36], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12.run(buf234, buf232, buf240, buf239, convert_element_type_307, pow_24, buf241, 160, 7373, stream=stream0)
            buf242 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [add_69, X_36], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf241, buf242, 32, 5, stream=stream0)
            buf243 = buf62; del buf62  # reuse
            buf340 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [add_69], Original ATen: [aten.masked_fill, aten.mul, aten._to_copy, aten.add, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_22.run(buf243, mean_2, buf234, buf232, buf240, buf239, pow_24, buf242, convert_element_type_307, buf340, 1179648, stream=stream0)
            del convert_element_type_307
            del pow_24
            buf341 = reinterpret_tensor(buf184, (32, 1024, 192), (196608, 192, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_304, buf340, out=buf341)
            del permute_304
            buf342 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf340, permute_305, out=buf342)
            del permute_305
            buf252 = buf340; del buf340  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf245, buf198, buf251, buf250, buf252, 6144, 192, stream=stream0)
            buf253 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_241, buf252, out=buf253)
            del permute_241
            buf254 = buf239; del buf239  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf252, permute_242, out=buf254)
            buf255 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_243, buf254, out=buf255)
            del permute_243
            buf256 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf254, permute_244, out=buf256)
            del permute_244
            buf257 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf257, buf256, buf254, 1179648, stream=stream0)
            buf258 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_242, buf257, out=buf258)
            del permute_242
            buf259 = buf254; del buf254  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf257, permute_246, out=buf259)
            del permute_246
            buf260 = buf232; del buf232  # reuse
            buf261 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf253, buf245, buf198, buf251, buf250, buf259, buf258, buf260, buf261, 1179648, stream=stream0)
            buf262 = buf259; del buf259  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_248, buf261, out=buf262)
            del permute_248
            buf263 = buf258; del buf258  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf261, permute_249, out=buf263)
            buf264 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_250, buf263, out=buf264)
            del permute_250
            buf265 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf263, permute_251, out=buf265)
            del permute_251
            buf266 = buf264; del buf264  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf266, buf265, buf263, 1179648, stream=stream0)
            buf267 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_249, buf266, out=buf267)
            del permute_249
            buf268 = buf263; del buf263  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf266, permute_253, out=buf268)
            del permute_253
            buf269 = buf266; del buf266  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf262, buf260, buf268, buf267, buf269, 6144, 192, stream=stream0)
            buf270 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_255, buf269, out=buf270)
            del permute_255
            buf271 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf269, permute_256, out=buf271)
            buf272 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_257, buf271, out=buf272)
            del permute_257
            buf273 = buf245; del buf245  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf271, permute_258, out=buf273)
            del permute_258
            buf274 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf274, buf273, buf271, 1179648, stream=stream0)
            buf275 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_256, buf274, out=buf275)
            del permute_256
            buf276 = buf271; del buf271  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf274, permute_260, out=buf276)
            del permute_260
            buf277 = buf260; del buf260  # reuse
            buf278 = buf274; del buf274  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf277, buf270, buf262, buf268, buf267, buf276, buf275, buf278, 1179648, stream=stream0)
            buf279 = buf276; del buf276  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_262, buf278, out=buf279)
            del permute_262
            buf280 = buf275; del buf275  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf278, permute_263, out=buf280)
            buf281 = buf278; del buf278  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_264, buf280, out=buf281)
            del permute_264
            buf282 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf280, permute_265, out=buf282)
            del permute_265
            buf283 = buf281; del buf281  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf283, buf282, buf280, 1179648, stream=stream0)
            buf284 = buf282; del buf282  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_263, buf283, out=buf284)
            del permute_263
            buf285 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf283, permute_267, out=buf285)
            del permute_267
            buf286 = buf241; del buf241  # reuse
            # Topologically Sorted Source Nodes: [X_28, add_58, X_29], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_23.run(buf279, buf277, buf285, buf284, add_44, pow_22, buf286, 160, 7373, stream=stream0)
            buf287 = buf242; del buf242  # reuse
            # Topologically Sorted Source Nodes: [X_28, add_58, X_29], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf286, buf287, 32, 5, stream=stream0)
            buf297 = buf283; del buf283  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf290, buf195, buf296, buf295, buf297, 6144, 192, stream=stream0)
            buf298 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_276, buf297, out=buf298)
            del permute_276
            buf299 = buf267; del buf267  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf297, permute_277, out=buf299)
            buf300 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_278, buf299, out=buf300)
            del permute_278
            buf301 = buf262; del buf262  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf299, permute_279, out=buf301)
            del permute_279
            buf302 = buf300; del buf300  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf302, buf301, buf299, 1179648, stream=stream0)
            buf303 = buf301; del buf301  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_277, buf302, out=buf303)
            del permute_277
            buf304 = buf299; del buf299  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf302, permute_281, out=buf304)
            del permute_281
            buf305 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf306 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf298, buf290, buf195, buf296, buf295, buf304, buf303, buf305, buf306, 1179648, stream=stream0)
            buf307 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_283, buf306, out=buf307)
            del permute_283
            buf308 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf306, permute_284, out=buf308)
            buf309 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_285, buf308, out=buf309)
            del permute_285
            buf310 = buf298; del buf298  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf308, permute_286, out=buf310)
            del permute_286
            buf311 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf311, buf310, buf308, 1179648, stream=stream0)
            buf312 = buf310; del buf310  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_284, buf311, out=buf312)
            del permute_284
            buf313 = buf308; del buf308  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf311, permute_288, out=buf313)
            del permute_288
            buf314 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf307, buf305, buf313, buf312, buf314, 6144, 192, stream=stream0)
            buf315 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_290, buf314, out=buf315)
            del permute_290
            buf316 = buf295; del buf295  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf314, permute_291, out=buf316)
            buf317 = buf314; del buf314  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_292, buf316, out=buf317)
            del permute_292
            buf318 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf316, permute_293, out=buf318)
            del permute_293
            buf319 = buf317; del buf317  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf319, buf318, buf316, 1179648, stream=stream0)
            buf320 = buf318; del buf318  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_291, buf319, out=buf320)
            del permute_291
            buf321 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf319, permute_295, out=buf321)
            del permute_295
            buf322 = buf305; del buf305  # reuse
            buf323 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf322, buf315, buf307, buf313, buf312, buf321, buf320, buf323, 1179648, stream=stream0)
            buf324 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_297, buf323, out=buf324)
            del permute_297
            buf325 = buf320; del buf320  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf323, permute_298, out=buf325)
            buf326 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_299, buf325, out=buf326)
            del permute_299
            buf327 = buf315; del buf315  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf325, permute_300, out=buf327)
            del permute_300
            buf328 = buf326; del buf326  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf328, buf327, buf325, 1179648, stream=stream0)
            buf329 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_298, buf328, out=buf329)
            del permute_298
            buf330 = buf325; del buf325  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf328, permute_302, out=buf330)
            del permute_302
            buf331 = buf286; del buf286  # reuse
            # Topologically Sorted Source Nodes: [X_21, add_47, X_22], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_23.run(buf324, buf322, buf330, buf329, add_45, pow_20, buf331, 160, 7373, stream=stream0)
            buf332 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [X_21, add_47, X_22], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf331, buf332, 32, 5, stream=stream0)
            buf288 = buf107; del buf107  # reuse
            buf344 = buf328; del buf328  # reuse
            buf333 = buf152; del buf152  # reuse
            buf348 = buf313; del buf313  # reuse
            # Topologically Sorted Source Nodes: [X_28, add_58, X_21, add_47], Original ATen: [aten.masked_fill, aten.mul, aten._to_copy, aten.add, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_24.run(buf288, buf333, mean_2, buf279, buf277, buf285, buf284, pow_22, buf287, add_44, buf324, buf322, buf330, buf329, pow_20, buf332, add_45, buf344, buf348, 1179648, stream=stream0)
            del add_44
            del add_45
            del buf277
            del buf322
            del mean_2
            del pow_20
            del pow_22
            buf334 = buf331; del buf331  # reuse
            buf336 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            buf338 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw2_1, dw1_momentum, mul_11, dw1_1, dw0_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_mul_sum_zeros_like_25.run(buf243, bmm_8, mean, buf333, bmm_6, buf288, bmm_7, buf334, buf336, buf338, 160, 7373, stream=stream0)
            del bmm_8
            buf335 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw2_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf334, buf335, 32, 5, stream=stream0)
            del buf334
            buf337 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf336, buf337, 32, 5, stream=stream0)
            buf339 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf338, buf339, 32, 5, stream=stream0)
            buf345 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_306, buf344, out=buf345)
            del permute_306
            buf346 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf344, permute_307, out=buf346)
            del permute_307
            buf350 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf348, permute_309, out=buf350)
            del permute_309
            buf343 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            buf347 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [ki_1], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_slice_sum_26.run(buf341, primals_7, buf345, buf343, buf347, 32768, 192, stream=stream0)
            buf349 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_308, buf348, out=buf349)
            buf352 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf353 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf356 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf357 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [silu_4, lr1i_1, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43], Original ATen: [aten.fill, aten._to_copy, aten.silu, aten.slice, aten.mul, aten.transpose, aten.sigmoid, aten.rsub, aten.add, aten.neg, aten.sigmoid_backward, aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_27.run(buf357, buf346, bmm_57, bmm_58, buf342, bmm_59, buf349, primals_8, buf352, buf356, 6291456, stream=stream0)
            del bmm_59
            buf351 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [silu_4, hidden_1, transpose_24], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.transpose, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_silu_sum_transpose_18.run(buf349, bmm_57, bmm_58, buf351, 32768, 192, stream=stream0)
            del bmm_57
            del bmm_58
            buf355 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf352, permute_308, out=buf355)
            del permute_308
            buf354 = reinterpret_tensor(buf349, (32, 192, 1024), (196608, 1024, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_312, buf352, out=buf354)
            del permute_312
            buf359 = buf344; del buf344  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf356, permute_317, out=buf359)
            buf361 = buf330; del buf330  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf357, permute_317, out=buf361)
            del permute_317
            buf358 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_316, buf356, out=buf358)
            buf360 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_319, buf357, out=buf360)
            buf362 = reinterpret_tensor(buf357, (32, 192, 1024), (196608, 1, 192), 0); del buf357  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_copy_slice_transpose_zeros_like_28.run(tangents_1, buf362, 6291456, stream=stream0)
            buf364 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf362, permute_323, out=buf364)
            del permute_323
            buf376 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf378 = buf195; del buf195  # reuse
            buf472 = buf324; del buf324  # reuse
            # Topologically Sorted Source Nodes: [add_41, truediv_4], Original ATen: [aten.masked_fill, aten._to_copy, aten.transpose, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_transpose_20.run(buf378, buf355, buf364, add_37, pow_16, pow_4, buf376, buf472, 6144, 192, stream=stream0)
            del add_37
            del pow_16
            buf363 = buf346; del buf346  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone, aten.bmm]
            extern_kernels.bmm(permute_322, buf362, out=buf363)
            del permute_322
            buf365 = reinterpret_tensor(buf362, (32, 192, 1024), (196608, 1024, 1), 0); del buf362  # reuse
            buf366 = bmm_54; del bmm_54  # reuse
            # Topologically Sorted Source Nodes: [gate_1], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5.run(buf366, buf363, bmm_55, buf365, 6291456, stream=stream0)
            del bmm_55
            buf546 = reinterpret_tensor(buf363, (32, 192, 1024), (196608, 1, 192), 0); del buf363  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_copy_slice_transpose_zeros_like_29.run(tangents_1, buf546, 6291456, stream=stream0)
            del tangents_1
            buf548 = buf364; del buf364  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf546, permute_453, out=buf548)
            del permute_453
            buf547 = buf342; del buf342  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.slice, aten.zeros_like, aten.copy, aten.clone, aten.bmm]
            extern_kernels.bmm(permute_452, buf546, out=buf547)
            del permute_452
            buf549 = reinterpret_tensor(buf546, (32, 192, 1024), (196608, 1024, 1), 0); del buf546  # reuse
            buf550 = bmm; del bmm  # reuse
            # Topologically Sorted Source Nodes: [gate], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_silu_sub_5.run(buf550, buf547, bmm_1, buf549, 6291456, stream=stream0)
            del bmm_1
            buf473 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_399, buf472, out=buf473)
            del permute_399
            buf474 = buf285; del buf285  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf472, permute_400, out=buf474)
            buf475 = buf472; del buf472  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_401, buf474, out=buf475)
            del permute_401
            buf476 = buf284; del buf284  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf474, permute_402, out=buf476)
            del permute_402
            buf477 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf477, buf476, buf474, 1179648, stream=stream0)
            buf478 = buf476; del buf476  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_400, buf477, out=buf478)
            del permute_400
            buf479 = buf474; del buf474  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf477, permute_404, out=buf479)
            del permute_404
            buf368 = buf477; del buf477  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf366, permute_326, out=buf368)
            buf370 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf365, permute_326, out=buf370)
            del permute_326
            buf373 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf375 = buf192; del buf192  # reuse
            buf382 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [add_42, truediv_5], Original ATen: [aten.masked_fill, aten._to_copy, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21.run(buf375, buf359, buf370, add_39, pow_18, pow_6, buf373, buf382, 6144, 192, stream=stream0)
            del add_39
            del pow_18
            buf379 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf381 = buf198; del buf198  # reuse
            buf427 = buf370; del buf370  # reuse
            # Topologically Sorted Source Nodes: [add_40, truediv_3], Original ATen: [aten.masked_fill, aten._to_copy, aten.add, aten.div, aten.mul, aten.sum, aten.neg, aten.eq]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_eq_masked_fill_mul_neg_sum_21.run(buf381, buf361, buf368, add_38, pow_14, pow_2, buf379, buf427, 6144, 192, stream=stream0)
            del add_38
            del pow_14
            buf367 = buf547; del buf547  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_319, buf366, out=buf367)
            del permute_319
            buf369 = buf366; del buf366  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_316, buf365, out=buf369)
            del permute_316
            buf383 = buf368; del buf368  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_329, buf382, out=buf383)
            del permute_329
            buf428 = buf361; del buf361  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_364, buf427, out=buf428)
            del permute_364
            buf384 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf382, permute_330, out=buf384)
            buf385 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_331, buf384, out=buf385)
            del permute_331
            buf386 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf384, permute_332, out=buf386)
            del permute_332
            buf387 = buf385; del buf385  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf387, buf386, buf384, 1179648, stream=stream0)
            buf388 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_330, buf387, out=buf388)
            del permute_330
            buf389 = buf384; del buf384  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf387, permute_334, out=buf389)
            del permute_334
            buf390 = buf387; del buf387  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf383, buf375, buf389, buf388, buf390, 6144, 192, stream=stream0)
            buf391 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_336, buf390, out=buf391)
            del permute_336
            buf392 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf390, permute_337, out=buf392)
            buf393 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_338, buf392, out=buf393)
            del permute_338
            buf394 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf392, permute_339, out=buf394)
            del permute_339
            buf395 = buf393; del buf393  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf395, buf394, buf392, 1179648, stream=stream0)
            buf396 = buf394; del buf394  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_337, buf395, out=buf396)
            del permute_337
            buf397 = buf392; del buf392  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf395, permute_341, out=buf397)
            del permute_341
            buf398 = buf375; del buf375  # reuse
            buf399 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_30.run(buf398, buf391, buf383, buf389, buf388, buf397, buf396, buf399, 1179648, stream=stream0)
            buf400 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_343, buf399, out=buf400)
            del permute_343
            buf401 = buf396; del buf396  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf399, permute_344, out=buf401)
            buf402 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_345, buf401, out=buf402)
            del permute_345
            buf403 = buf391; del buf391  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf401, permute_346, out=buf403)
            del permute_346
            buf404 = buf402; del buf402  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf404, buf403, buf401, 1179648, stream=stream0)
            buf405 = buf403; del buf403  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_344, buf404, out=buf405)
            del permute_344
            buf406 = buf401; del buf401  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf404, permute_348, out=buf406)
            del permute_348
            buf407 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf400, buf398, buf406, buf405, buf407, 6144, 192, stream=stream0)
            buf408 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_350, buf407, out=buf408)
            del permute_350
            buf409 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf407, permute_351, out=buf409)
            buf410 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_352, buf409, out=buf410)
            del permute_352
            buf411 = buf383; del buf383  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf409, permute_353, out=buf411)
            del permute_353
            buf412 = buf410; del buf410  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf412, buf411, buf409, 1179648, stream=stream0)
            buf413 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_351, buf412, out=buf413)
            del permute_351
            buf414 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf412, permute_355, out=buf414)
            del permute_355
            buf415 = buf398; del buf398  # reuse
            buf416 = buf412; del buf412  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf415, buf408, buf400, buf406, buf405, buf414, buf413, buf416, 1179648, stream=stream0)
            del buf400
            buf417 = buf414; del buf414  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_357, buf416, out=buf417)
            del permute_357
            buf418 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf416, permute_358, out=buf418)
            buf419 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_359, buf418, out=buf419)
            del permute_359
            buf420 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf418, permute_360, out=buf420)
            del permute_360
            buf421 = buf419; del buf419  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf421, buf420, buf418, 1179648, stream=stream0)
            buf422 = buf420; del buf420  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_358, buf421, out=buf422)
            del permute_358
            buf423 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf421, permute_362, out=buf423)
            del permute_362
            buf424 = buf338; del buf338  # reuse
            # Topologically Sorted Source Nodes: [add_26, X_15], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_12.run(buf417, buf415, buf423, buf422, convert_element_type_129, pow_12, buf424, 160, 7373, stream=stream0)
            buf425 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [add_26, X_15], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf424, buf425, 32, 5, stream=stream0)
            buf426 = buf243; del buf243  # reuse
            buf526 = buf421; del buf421  # reuse
            # Topologically Sorted Source Nodes: [add_26], Original ATen: [aten.masked_fill, aten.mul, aten._to_copy, aten.add, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_22.run(buf426, mean_1, buf417, buf415, buf423, buf422, pow_12, buf425, convert_element_type_129, buf526, 1179648, stream=stream0)
            del buf415
            del convert_element_type_129
            del pow_12
            buf517 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum], Original ATen: [aten.zeros_like, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_zeros_like_31.run(buf426, buf517, 160, 7373, stream=stream0)
            buf518 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum], Original ATen: [aten.zeros_like, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf517, buf518, 32, 5, stream=stream0)
            buf527 = reinterpret_tensor(buf365, (32, 1024, 192), (196608, 192, 1), 0); del buf365  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_434, buf526, out=buf527)
            del permute_434
            buf528 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf526, permute_435, out=buf528)
            del permute_435
            buf429 = buf526; del buf526  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf427, permute_365, out=buf429)
            buf430 = buf423; del buf423  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_366, buf429, out=buf430)
            del permute_366
            buf431 = buf422; del buf422  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf429, permute_367, out=buf431)
            del permute_367
            buf432 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf432, buf431, buf429, 1179648, stream=stream0)
            buf433 = buf431; del buf431  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_365, buf432, out=buf433)
            del permute_365
            buf434 = buf429; del buf429  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf432, permute_369, out=buf434)
            del permute_369
            buf435 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf428, buf381, buf434, buf433, buf435, 6144, 192, stream=stream0)
            buf436 = buf417; del buf417  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_371, buf435, out=buf436)
            del permute_371
            buf437 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf435, permute_372, out=buf437)
            buf438 = buf435; del buf435  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_373, buf437, out=buf438)
            del permute_373
            buf439 = buf405; del buf405  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf437, permute_374, out=buf439)
            del permute_374
            buf440 = buf438; del buf438  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf440, buf439, buf437, 1179648, stream=stream0)
            buf441 = buf439; del buf439  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_372, buf440, out=buf441)
            del permute_372
            buf442 = buf437; del buf437  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf440, permute_376, out=buf442)
            del permute_376
            buf443 = buf381; del buf381  # reuse
            buf444 = buf440; del buf440  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_30.run(buf443, buf436, buf428, buf434, buf433, buf442, buf441, buf444, 1179648, stream=stream0)
            buf445 = buf442; del buf442  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_378, buf444, out=buf445)
            del permute_378
            buf446 = buf441; del buf441  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf444, permute_379, out=buf446)
            buf447 = buf444; del buf444  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_380, buf446, out=buf447)
            del permute_380
            buf448 = buf436; del buf436  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf446, permute_381, out=buf448)
            del permute_381
            buf449 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf449, buf448, buf446, 1179648, stream=stream0)
            buf450 = buf448; del buf448  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_379, buf449, out=buf450)
            del permute_379
            buf451 = buf446; del buf446  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf449, permute_383, out=buf451)
            del permute_383
            buf452 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf445, buf443, buf451, buf450, buf452, 6144, 192, stream=stream0)
            buf453 = buf434; del buf434  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_385, buf452, out=buf453)
            del permute_385
            buf454 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf452, permute_386, out=buf454)
            buf455 = buf452; del buf452  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_387, buf454, out=buf455)
            del permute_387
            buf456 = buf428; del buf428  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf454, permute_388, out=buf456)
            del permute_388
            buf457 = buf455; del buf455  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf457, buf456, buf454, 1179648, stream=stream0)
            buf458 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_386, buf457, out=buf458)
            del permute_386
            buf459 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf457, permute_390, out=buf459)
            del permute_390
            buf460 = buf443; del buf443  # reuse
            buf461 = buf457; del buf457  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf460, buf453, buf445, buf451, buf450, buf459, buf458, buf461, 1179648, stream=stream0)
            buf462 = buf459; del buf459  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_392, buf461, out=buf462)
            del permute_392
            buf463 = buf458; del buf458  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf461, permute_393, out=buf463)
            buf464 = buf461; del buf461  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_394, buf463, out=buf464)
            del permute_394
            buf465 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf463, permute_395, out=buf465)
            del permute_395
            buf466 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf466, buf465, buf463, 1179648, stream=stream0)
            buf467 = buf465; del buf465  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_393, buf466, out=buf467)
            del permute_393
            buf468 = buf463; del buf463  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf466, permute_397, out=buf468)
            del permute_397
            buf480 = buf466; del buf466  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_3.run(buf473, buf378, buf479, buf478, buf480, 6144, 192, stream=stream0)
            buf481 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_406, buf480, out=buf481)
            del permute_406
            buf482 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf480, permute_407, out=buf482)
            buf483 = buf480; del buf480  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_408, buf482, out=buf483)
            del permute_408
            buf484 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf482, permute_409, out=buf484)
            del permute_409
            buf485 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf485, buf484, buf482, 1179648, stream=stream0)
            buf486 = buf484; del buf484  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_407, buf485, out=buf486)
            del permute_407
            buf487 = buf482; del buf482  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf485, permute_411, out=buf487)
            del permute_411
            buf488 = buf426; del buf426  # reuse
            buf489 = buf485; del buf485  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_6.run(buf481, buf473, buf378, buf479, buf478, buf487, buf486, buf488, buf489, 1179648, stream=stream0)
            buf490 = buf487; del buf487  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_413, buf489, out=buf490)
            del permute_413
            buf491 = buf486; del buf486  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf489, permute_414, out=buf491)
            buf492 = buf489; del buf489  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_415, buf491, out=buf492)
            del permute_415
            buf493 = buf481; del buf481  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf491, permute_416, out=buf493)
            del permute_416
            buf494 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf494, buf493, buf491, 1179648, stream=stream0)
            buf495 = buf493; del buf493  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_414, buf494, out=buf495)
            del permute_414
            buf496 = buf491; del buf491  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf494, permute_418, out=buf496)
            del permute_418
            buf497 = buf494; del buf494  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_8.run(buf490, buf488, buf496, buf495, buf497, 6144, 192, stream=stream0)
            buf498 = buf479; del buf479  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_420, buf497, out=buf498)
            del permute_420
            buf499 = buf478; del buf478  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf497, permute_421, out=buf499)
            buf500 = buf497; del buf497  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_422, buf499, out=buf500)
            del permute_422
            buf501 = buf473; del buf473  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf499, permute_423, out=buf501)
            del permute_423
            buf502 = buf500; del buf500  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf502, buf501, buf499, 1179648, stream=stream0)
            buf503 = buf501; del buf501  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_421, buf502, out=buf503)
            del permute_421
            buf504 = buf499; del buf499  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf502, permute_425, out=buf504)
            del permute_425
            buf505 = buf488; del buf488  # reuse
            buf506 = buf502; del buf502  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_10.run(buf505, buf498, buf490, buf496, buf495, buf504, buf503, buf506, 1179648, stream=stream0)
            del buf490
            del buf495
            buf507 = buf504; del buf504  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(permute_427, buf506, out=buf507)
            del permute_427
            buf508 = buf503; del buf503  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf506, permute_428, out=buf508)
            buf509 = buf506; del buf506  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_429, buf508, out=buf509)
            del permute_429
            buf510 = buf498; del buf498  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf508, permute_430, out=buf510)
            del permute_430
            buf511 = buf509; del buf509  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf511, buf510, buf508, 1179648, stream=stream0)
            buf512 = buf510; del buf510  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_428, buf511, out=buf512)
            del permute_428
            buf513 = buf508; del buf508  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf511, permute_432, out=buf513)
            del permute_432
            buf469 = buf517; del buf517  # reuse
            buf514 = buf336; del buf336  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw1_momentum, mul_11, dw1_1, dw0_1, X_7, add_15, X_8, X, add_4, X_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_mul_neg_sum_transpose_zeros_like_32.run(buf462, buf460, buf468, buf467, bmm_7, mean, pow_10, buf507, buf505, buf513, buf512, bmm_6, pow_8, buf469, buf514, 160, 7373, stream=stream0)
            buf470 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, X_7, add_15, X_8], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf469, buf470, 32, 5, stream=stream0)
            buf515 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, X, add_4, X_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.transpose, aten.div, aten.neg, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf514, buf515, 32, 5, stream=stream0)
            buf516 = buf507; del buf507  # reuse
            buf519 = buf511; del buf511  # reuse
            buf471 = buf462; del buf462  # reuse
            buf522 = buf496; del buf496  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw1_momentum, mul_11, dw1_1, dw0_1, X_7, add_15, X, add_4], Original ATen: [aten.masked_fill, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.transpose, aten.div, aten.eq]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_zeros_like_33.run(buf516, buf471, buf505, buf513, buf512, pow_8, buf515, bmm_6, mean, buf333, mean_1, buf460, buf468, buf467, pow_10, buf470, bmm_7, buf288, buf519, buf522, 6144, 192, stream=stream0)
            del bmm_6
            del bmm_7
            del buf460
            del buf467
            del buf468
            del buf505
            del buf512
            del buf513
            del mean
            del pow_10
            del pow_8
            buf520 = buf514; del buf514  # reuse
            buf523 = buf469; del buf469  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum, dw1_momentum], Original ATen: [aten.zeros_like, aten.mul, aten._to_copy, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_zeros_like_34.run(buf333, mean_1, buf516, buf288, buf471, buf520, buf523, 160, 7373, stream=stream0)
            del buf288
            del buf333
            del buf471
            del mean_1
            buf521 = buf515; del buf515  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum], Original ATen: [aten.zeros_like, aten.mul, aten._to_copy, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf520, buf521, 32, 5, stream=stream0)
            del buf520
            buf524 = buf470; del buf470  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum], Original ATen: [aten.zeros_like, aten.mul, aten._to_copy, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_mul_neg_sum_transpose_13.run(buf523, buf524, 32, 5, stream=stream0)
            del buf523
            buf530 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_436, buf522, out=buf530)
            del permute_436
            buf531 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf522, permute_437, out=buf531)
            del permute_437
            buf534 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf519, permute_439, out=buf534)
            del permute_439
            buf529 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            buf532 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [ki], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_slice_sum_35.run(buf527, primals_7, buf530, buf529, buf532, 32768, 192, stream=stream0)
            del primals_7
            buf555 = empty_strided_cuda((32, 4096, 1), (4096, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_36.run(buf166, buf347, buf532, buf555, 131072, stream=stream0)
            del buf166
            del buf347
            del buf532
            buf556 = empty_strided_cuda((32, 4096, 1), (4096, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_36.run(buf162, buf343, buf529, buf556, 131072, stream=stream0)
            del buf162
            del buf343
            buf525 = empty_strided_cuda((32, 4096, 1), (4096, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.expand, aten.div, aten.slice_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_expand_slice_backward_37.run(buf154, buf156, buf158, buf335, buf337, buf339, buf518, buf521, buf524, buf525, 131072, stream=stream0)
            del buf154
            del buf156
            del buf158
            del buf335
            del buf337
            del buf339
            del buf518
            del buf521
            del buf524
            buf552 = buf522; del buf522  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf550, permute_456, out=buf552)
            buf554 = buf516; del buf516  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf549, permute_456, out=buf554)
            del permute_456
            buf551 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_1, (32, 192, 192), (36864, 1, 192), 0), buf550, out=buf551)
            buf553 = buf550; del buf550  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_3, (32, 192, 192), (36864, 1, 192), 0), buf549, out=buf553)
            del buf549
            buf371 = empty_strided_cuda((32, 192, 4096), (786432, 4096, 1), torch.float32)
            buf560 = buf371; del buf371  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.slice_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_slice_backward_38.run(buf560, buf5, buf7, buf186, buf188, buf367, buf369, buf551, buf553, 25165824, stream=stream0)
            del buf186
            del buf188
            del buf367
            del buf369
            buf533 = reinterpret_tensor(buf7, (32, 1024, 192), (196608, 192, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_438, buf519, out=buf533)
            buf536 = buf553; del buf553  # reuse
            buf537 = buf551; del buf551  # reuse
            buf540 = buf5; del buf5  # reuse
            buf541 = buf537; del buf537  # reuse
            # Topologically Sorted Source Nodes: [silu_1, lr1i, dgate, sigma, mul_4, sub, mul_5, add], Original ATen: [aten.fill, aten._to_copy, aten.silu, aten.slice, aten.mul, aten.transpose, aten.sigmoid, aten.rsub, aten.add, aten.neg, aten.sigmoid_backward, aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_fill_mul_neg_rsub_sigmoid_sigmoid_backward_silu_slice_sub_transpose_39.run(buf541, buf531, bmm_3, bmm_4, buf528, bmm_5, buf533, primals_8, buf536, buf540, 6291456, stream=stream0)
            del bmm_5
            del buf528
            del buf531
            del primals_8
            buf535 = buf529; del buf529  # reuse
            # Topologically Sorted Source Nodes: [silu_1, hidden, transpose_5], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.transpose, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_silu_sum_transpose_18.run(buf533, bmm_3, bmm_4, buf535, 32768, 192, stream=stream0)
            del bmm_3
            del bmm_4
            buf539 = buf519; del buf519  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf536, permute_438, out=buf539)
            del permute_438
            buf562 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.masked_fill, aten.add, aten._to_copy, aten.transpose, aten.div, aten.eq, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_transpose_40.run(buf562, buf539, buf548, buf12, buf193, buf376, pow_4, primals_2, 6144, 192, stream=stream0)
            del buf12
            del buf193
            del buf376
            del pow_4
            del primals_2
            buf538 = reinterpret_tensor(buf533, (32, 192, 1024), (196608, 1024, 1), 0); del buf533  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_442, buf536, out=buf538)
            del buf536
            del permute_442
            buf559 = empty_strided_cuda((32, 192, 4096), (786432, 4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_41.run(buf169, buf173, buf350, buf354, buf534, buf538, buf559, 25165824, stream=stream0)
            del buf169
            del buf173
            del buf350
            del buf354
            del buf534
            buf557 = empty_strided_cuda((32, 4096, 1), (4096, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_36.run(buf170, buf351, buf535, buf557, 131072, stream=stream0)
            del buf170
            del buf351
            del buf535
            buf543 = buf548; del buf548  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf540, permute_447, out=buf543)
            buf545 = buf539; del buf539  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf541, permute_447, out=buf545)
            del permute_447
            buf561 = buf382; del buf382  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.masked_fill, aten.add, aten.div, aten.eq, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_42.run(buf561, buf543, buf554, buf9, buf190, buf373, pow_6, primals_3, 6144, 192, stream=stream0)
            del buf190
            del buf373
            del buf543
            del buf554
            del buf9
            del pow_6
            buf563 = buf427; del buf427  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.masked_fill, aten.add, aten.div, aten.eq, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_eq_masked_fill_mul_42.run(buf563, buf545, buf552, buf15, buf196, buf379, pow_2, primals_1, 6144, 192, stream=stream0)
            del buf15
            del buf196
            del buf379
            del buf545
            del buf552
            del pow_2
            buf542 = buf538; del buf538  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_3, (32, 192, 192), (36864, 1, 192), 0), buf540, out=buf542)
            del primals_3
            buf544 = buf540; del buf540  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_1, (32, 192, 192), (36864, 1, 192), 0), buf541, out=buf544)
            del buf541
            del primals_1
            buf372 = empty_strided_cuda((32, 4096, 192), (786432, 192, 1), torch.float32)
            buf558 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [lr2i_2, lr0i_2, lr2i_1, lr0i_1, lr2i, lr0i], Original ATen: [aten._to_copy, aten.slice, aten.mul, aten.add, aten.transpose, aten.slice_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_slice_slice_backward_transpose_43.run(buf558, buf160, primals_9, buf164, primals_10, buf177, buf179, buf341, buf345, buf358, buf360, buf527, buf530, buf542, buf544, 131072, 192, stream=stream0)
            del buf160
            del buf164
            del buf177
            del buf179
            del buf341
            del buf345
            del buf358
            del buf360
            del buf527
            del buf530
            del buf542
            del buf544
            del primals_10
            del primals_9
        return (buf563, buf562, buf561, buf525, reinterpret_tensor(buf560, (32, 4096, 192), (786432, 1, 4096), 0), reinterpret_tensor(buf559, (32, 4096, 192), (786432, 1, 4096), 0), buf558, buf557, buf556, buf555, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((32, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    pow_2 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_4 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_6 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    bmm = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_1 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_3 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_4 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_5 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_6 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_7 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_8 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    mean = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_8 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_10 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_129 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    pow_12 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_37 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_38 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_39 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    pow_14 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_16 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_18 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    bmm_54 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_55 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_57 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_58 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_59 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    mean_1 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_44 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    pow_20 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_22 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_307 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    pow_24 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_80 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_81 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_82 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    pow_26 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_28 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_30 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    bmm_108 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_109 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_111 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_112 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_113 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    mean_2 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_391 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    pow_32 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_438 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    pow_34 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_485 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    pow_36 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_123 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_124 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    add_125 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    pow_38 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_40 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_42 = rand_strided((32, 192, 1), (192, 1, 1), device='cuda:0', dtype=torch.float32)
    bmm_162 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_163 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_62 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_63 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_65 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_66 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_67 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_69 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_70 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_71 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_72 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_74 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_76 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_77 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_78 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_79 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_81 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_83 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_84 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_85 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_86 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_88 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_90 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_91 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_92 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_93 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_95 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_97 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_98 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_99 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_100 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_102 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_104 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_105 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_106 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_107 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_109 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_111 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_112 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_113 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_114 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_116 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_118 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_119 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_120 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_121 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_123 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_125 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_126 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_127 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_128 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_130 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_132 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_133 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_134 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_135 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_137 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_139 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_140 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_141 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_142 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_144 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_146 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_147 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_148 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_149 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_151 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_153 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_154 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_155 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_156 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_158 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_160 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_161 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_162 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_163 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_165 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_167 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_168 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_169 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_170 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_172 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_174 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_175 = rand_strided((32, 192, 1024), (196608, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_176 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_177 = rand_strided((32, 192, 1024), (196608, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_178 = rand_strided((32, 1024, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_179 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_182 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_186 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_187 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_189 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_192 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_193 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_196 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_199 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_200 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_201 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_202 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_204 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_206 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_207 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_208 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_209 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_211 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_213 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_214 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_215 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_216 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_218 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_220 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_221 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_222 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_223 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_225 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_227 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_228 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_229 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_230 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_232 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_234 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_235 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_236 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_237 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_239 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_241 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_242 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_243 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_244 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_246 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_248 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_249 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_250 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_251 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_253 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_255 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_256 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_257 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_258 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_260 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_262 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_263 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_264 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_265 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_267 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_269 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_270 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_271 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_272 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_274 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_276 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_277 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_278 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_279 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_281 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_283 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_284 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_285 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_286 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_288 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_290 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_291 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_292 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_293 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_295 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_297 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_298 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_299 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_300 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_302 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_304 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_305 = rand_strided((32, 192, 1024), (196608, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_306 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_307 = rand_strided((32, 192, 1024), (196608, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_308 = rand_strided((32, 1024, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_309 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_312 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_316 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_317 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_319 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_322 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_323 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_326 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_329 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_330 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_331 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_332 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_334 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_336 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_337 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_338 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_339 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_341 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_343 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_344 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_345 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_346 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_348 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_350 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_351 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_352 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_353 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_355 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_357 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_358 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_359 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_360 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_362 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_364 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_365 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_366 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_367 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_369 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_371 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_372 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_373 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_374 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_376 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_378 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_379 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_380 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_381 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_383 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_385 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_386 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_387 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_388 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_390 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_392 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_393 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_394 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_395 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_397 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_399 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_400 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_401 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_402 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_404 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_406 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_407 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_408 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_409 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_411 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_413 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_414 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_415 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_416 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_418 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_420 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_421 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_422 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_423 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_425 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_427 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_428 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_429 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_430 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_432 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_434 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_435 = rand_strided((32, 192, 1024), (196608, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_436 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_437 = rand_strided((32, 192, 1024), (196608, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_438 = rand_strided((32, 1024, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_439 = rand_strided((32, 192, 1024), (196608, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_442 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_447 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_452 = rand_strided((32, 192, 192), (36864, 1, 192), device='cuda:0', dtype=torch.bfloat16)
    permute_453 = rand_strided((32, 1024, 192), (196608, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_456 = rand_strided((32, 1024, 192), (196608, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_7, primals_8, primals_9, primals_10, pow_2, pow_4, pow_6, bmm, bmm_1, bmm_3, bmm_4, bmm_5, bmm_6, bmm_7, bmm_8, mean, pow_8, pow_10, convert_element_type_129, pow_12, add_37, add_38, add_39, pow_14, pow_16, pow_18, bmm_54, bmm_55, bmm_57, bmm_58, bmm_59, mean_1, add_44, add_45, add_46, pow_20, pow_22, convert_element_type_307, pow_24, add_80, add_81, add_82, pow_26, pow_28, pow_30, bmm_108, bmm_109, bmm_111, bmm_112, bmm_113, mean_2, convert_element_type_391, pow_32, convert_element_type_438, pow_34, convert_element_type_485, pow_36, add_123, add_124, add_125, pow_38, pow_40, pow_42, bmm_162, bmm_163, permute_62, permute_63, permute_65, permute_66, permute_67, permute_69, permute_70, permute_71, permute_72, permute_74, permute_76, permute_77, permute_78, permute_79, permute_81, permute_83, permute_84, permute_85, permute_86, permute_88, permute_90, permute_91, permute_92, permute_93, permute_95, permute_97, permute_98, permute_99, permute_100, permute_102, permute_104, permute_105, permute_106, permute_107, permute_109, permute_111, permute_112, permute_113, permute_114, permute_116, permute_118, permute_119, permute_120, permute_121, permute_123, permute_125, permute_126, permute_127, permute_128, permute_130, permute_132, permute_133, permute_134, permute_135, permute_137, permute_139, permute_140, permute_141, permute_142, permute_144, permute_146, permute_147, permute_148, permute_149, permute_151, permute_153, permute_154, permute_155, permute_156, permute_158, permute_160, permute_161, permute_162, permute_163, permute_165, permute_167, permute_168, permute_169, permute_170, permute_172, permute_174, permute_175, permute_176, permute_177, permute_178, permute_179, permute_182, permute_186, permute_187, permute_189, permute_192, permute_193, permute_196, permute_199, permute_200, permute_201, permute_202, permute_204, permute_206, permute_207, permute_208, permute_209, permute_211, permute_213, permute_214, permute_215, permute_216, permute_218, permute_220, permute_221, permute_222, permute_223, permute_225, permute_227, permute_228, permute_229, permute_230, permute_232, permute_234, permute_235, permute_236, permute_237, permute_239, permute_241, permute_242, permute_243, permute_244, permute_246, permute_248, permute_249, permute_250, permute_251, permute_253, permute_255, permute_256, permute_257, permute_258, permute_260, permute_262, permute_263, permute_264, permute_265, permute_267, permute_269, permute_270, permute_271, permute_272, permute_274, permute_276, permute_277, permute_278, permute_279, permute_281, permute_283, permute_284, permute_285, permute_286, permute_288, permute_290, permute_291, permute_292, permute_293, permute_295, permute_297, permute_298, permute_299, permute_300, permute_302, permute_304, permute_305, permute_306, permute_307, permute_308, permute_309, permute_312, permute_316, permute_317, permute_319, permute_322, permute_323, permute_326, permute_329, permute_330, permute_331, permute_332, permute_334, permute_336, permute_337, permute_338, permute_339, permute_341, permute_343, permute_344, permute_345, permute_346, permute_348, permute_350, permute_351, permute_352, permute_353, permute_355, permute_357, permute_358, permute_359, permute_360, permute_362, permute_364, permute_365, permute_366, permute_367, permute_369, permute_371, permute_372, permute_373, permute_374, permute_376, permute_378, permute_379, permute_380, permute_381, permute_383, permute_385, permute_386, permute_387, permute_388, permute_390, permute_392, permute_393, permute_394, permute_395, permute_397, permute_399, permute_400, permute_401, permute_402, permute_404, permute_406, permute_407, permute_408, permute_409, permute_411, permute_413, permute_414, permute_415, permute_416, permute_418, permute_420, permute_421, permute_422, permute_423, permute_425, permute_427, permute_428, permute_429, permute_430, permute_432, permute_434, permute_435, permute_436, permute_437, permute_438, permute_439, permute_442, permute_447, permute_452, permute_453, permute_456, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
