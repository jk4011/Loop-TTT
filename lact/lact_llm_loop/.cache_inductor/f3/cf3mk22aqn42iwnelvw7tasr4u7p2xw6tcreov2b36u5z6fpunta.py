# AOT ID: ['1_inference']
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/eh/cehqppvgihtn2trsc5yjh7esnovmraurppwpx27wep3vrisjgkjh.py
# Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type
#   norm => convert_element_type_1, pow_1, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   return %buf0
triton_red_fused__to_copy_linalg_vector_norm_0 = async_compile.triton('triton_red_fused__to_copy_linalg_vector_norm_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_linalg_vector_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 4718720}}
)
@triton.jit
def triton_red_fused__to_copy_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/l2/cl2bumxh66fbjouioicd5htlbvgwhr6eypyq2nod4ftrb4fm2njt.py
# Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type
#   norm => convert_element_type_1, pow_1, sum_1
# Graph fragment:
#   %buf0 : Tensor "f32[32, 1, 1, 5][5, 160, 160, 1]cuda:0" = PlaceHolder[target=buf0]
#   %convert_element_type : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   return %sum_1
triton_per_fused__to_copy_linalg_vector_norm_1 = async_compile.triton('triton_per_fused__to_copy_linalg_vector_norm_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_linalg_vector_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 400}}
)
@triton.jit
def triton_per_fused__to_copy_linalg_vector_norm_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7x/c7xadsxcu5epomh3f4mfyit4r52fb33onrmx4zwdubo7nqsbncv3.py
# Topologically Sorted Source Nodes: [X, norm, add, X_1, A, transpose, matmul_2], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A => convert_element_type_2, convert_element_type_3
#   X => convert_element_type
#   X_1 => div
#   add => add
#   matmul_2 => convert_element_type_8
#   norm => pow_2
#   transpose => permute
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_1]
#   %convert_element_type : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %pow_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %permute : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1]), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.bfloat16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %expand,%expand_1,%expand_5
triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_2 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = 1e-07
    tmp6 = tmp4 + tmp5
    tmp7 = (tmp2 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp8, None)
    tl.store(out_ptr2 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/km/ckmiis6yvvtx5ryovziyjgavmmmbsyj44xj2xupzckyigcbzjxpq.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_1
# Graph fragment:
#   %expand_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %mul_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, 2.927), kwargs = {})
#   return %expand_2
triton_poi_fused_mul_3 = async_compile.triton('triton_poi_fused_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.927
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/pa/cparmdslt6dgdb7awcpngess7yjqzx2fdowqjbth345a524tq5yd.py
# Topologically Sorted Source Nodes: [mul, B], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B => add_1
#   mul => mul
# Graph fragment:
#   %expand_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %bmm_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %mul : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, -6.8946), kwargs = {})
#   %add_1 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %bmm_1), kwargs = {})
#   return %expand_4
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -6.8946
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/kd/ckd62c6ifnc437t5elenbxxrqa3sd2nqpji2exkbxtp2np53wf5q.py
# Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, A_1, transpose_1, matmul_5], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   A_1 => convert_element_type_11, convert_element_type_12
#   X => convert_element_type
#   X_1 => div
#   X_2 => add_2
#   add => add
#   matmul_5 => convert_element_type_17
#   mul_2 => mul_2
#   norm => pow_2
#   transpose_1 => permute_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_1]
#   %bmm_2 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %convert_element_type : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %pow_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add), kwargs = {})
#   %mul_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %bmm_2), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %permute_1 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_2, [0, 2, 1]), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   return %expand_6,%expand_7,%expand_11
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_5 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = 1e-07
    tmp6 = tmp4 + tmp5
    tmp7 = (tmp2 / tmp6)
    tmp8 = 4.0848
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp13, None)
    tl.store(out_ptr2 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ek/cek44sxjfetdpxxwnaadcv6p7km6pkk5lrshkfmkfpbe4qhhf6dn.py
# Topologically Sorted Source Nodes: [mul_4], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_4 => mul_4
# Graph fragment:
#   %expand_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %mul_4 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 2.6377), kwargs = {})
#   return %expand_8
triton_poi_fused_mul_6 = async_compile.triton('triton_poi_fused_mul_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.6377
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/5g/c5gaeqecqrt2x5bjasxbl4qgvybw4t5wbql32ggtniqq5amhc74f.py
# Topologically Sorted Source Nodes: [mul_3, B_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_1 => add_3
#   mul_3 => mul_3
# Graph fragment:
#   %expand_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %bmm_4 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %mul_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, -6.3029), kwargs = {})
#   %add_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %bmm_4), kwargs = {})
#   return %expand_10
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -6.3029
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ff/cffja3qgb7ev6h35dndcftfc33dlmdflwar6yfkffs6v23qzlc57.py
# Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_5, X_3, A_2, transpose_2, matmul_8], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   A_2 => convert_element_type_20, convert_element_type_21
#   X => convert_element_type
#   X_1 => div
#   X_2 => add_2
#   X_3 => add_4
#   add => add
#   matmul_8 => convert_element_type_26
#   mul_2 => mul_2
#   mul_5 => mul_5
#   norm => pow_2
#   transpose_2 => permute_2
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_1]
#   %bmm_2 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %convert_element_type : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %pow_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add), kwargs = {})
#   %mul_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %bmm_2), kwargs = {})
#   %mul_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 3.9505), kwargs = {})
#   %add_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %bmm_5), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %permute_2 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [0, 2, 1]), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   return %expand_12,%expand_13,%expand_17
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_8 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = 1e-07
    tmp6 = tmp4 + tmp5
    tmp7 = (tmp2 / tmp6)
    tmp8 = 4.0848
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = 3.9505
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp18, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/2z/c2zgexdglr23rkci4jdrjqbu6ardjpdqesfpmh5bqlgw3aqhy5lo.py
# Topologically Sorted Source Nodes: [mul_7], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_7 => mul_7
# Graph fragment:
#   %expand_15 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %mul_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_6, 2.3037), kwargs = {})
#   return %expand_14
triton_poi_fused_mul_9 = async_compile.triton('triton_poi_fused_mul_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.3037
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/vl/cvlywidc5psqv3qib2dmc4qvoihjc3d6liwkqpyymkhj6vl2r73f.py
# Topologically Sorted Source Nodes: [mul_6, B_2], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_2 => add_5
#   mul_6 => mul_6
# Graph fragment:
#   %expand_15 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %mul_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_6, -5.5913), kwargs = {})
#   %add_5 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %bmm_7), kwargs = {})
#   return %expand_16
triton_poi_fused_add_mul_10 = async_compile.triton('triton_poi_fused_add_mul_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -5.5913
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/yl/cylzsyi4wbqlc66nprompxqujethhuzfiasqduu77lzdmradvxoa.py
# Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_5, X_3, mul_8, X_4, A_3, transpose_3, matmul_11], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   A_3 => convert_element_type_29, convert_element_type_30
#   X => convert_element_type
#   X_1 => div
#   X_2 => add_2
#   X_3 => add_4
#   X_4 => add_6
#   add => add
#   matmul_11 => convert_element_type_35
#   mul_2 => mul_2
#   mul_5 => mul_5
#   mul_8 => mul_8
#   norm => pow_2
#   transpose_3 => permute_3
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_1]
#   %bmm_2 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_6]
#   %convert_element_type : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %pow_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add), kwargs = {})
#   %mul_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %bmm_2), kwargs = {})
#   %mul_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 3.9505), kwargs = {})
#   %add_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %bmm_5), kwargs = {})
#   %mul_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 3.7418), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %bmm_8), kwargs = {})
#   %convert_element_type_30 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   %permute_3 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_6, [0, 2, 1]), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   return %add_6,%expand_18,%expand_19,%expand_23
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_11 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 35389440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp20 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = 1e-07
    tmp6 = tmp4 + tmp5
    tmp7 = (tmp2 / tmp6)
    tmp8 = 4.0848
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = 3.9505
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = 3.7418
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp23, None)
    tl.store(out_ptr2 + (x2), tmp23, None)
    tl.store(out_ptr3 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/gu/cguomz2vsqwipa5n3muoszqf5tcyvxgw36vrohe464utk4xzsyss.py
# Topologically Sorted Source Nodes: [mul_10], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_10 => mul_10
# Graph fragment:
#   %expand_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %mul_10 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 1.2046), kwargs = {})
#   return %expand_20
triton_poi_fused_mul_12 = async_compile.triton('triton_poi_fused_mul_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 1.2046
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/w4/cw4uuffihvrkcf5lgqybung5xixvv4qed4hahi5v3tgbibppfy6o.py
# Topologically Sorted Source Nodes: [mul_9, B_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_3 => add_7
#   mul_9 => mul_9
# Graph fragment:
#   %expand_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %bmm_10 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %mul_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, -3.1427), kwargs = {})
#   %add_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %bmm_10), kwargs = {})
#   return %expand_22
triton_poi_fused_add_mul_13 = async_compile.triton('triton_poi_fused_add_mul_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -3.1427
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/t3/ct3qkh6g6nsq6hfrwhjyqn3yj4pu46pv2ukjk4mrphwovqcaw6wa.py
# Topologically Sorted Source Nodes: [mul_11, X_5, A_4, transpose_4, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_4 => convert_element_type_38, convert_element_type_39
#   X_5 => add_8
#   matmul_14 => convert_element_type_44
#   mul_11 => mul_11
#   transpose_4 => permute_4
# Graph fragment:
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_6]
#   %bmm_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %mul_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 2.8769), kwargs = {})
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %bmm_11), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
#   %permute_4 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_8, [0, 2, 1]), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_44 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
#   return %expand_24,%expand_25,%expand_29
triton_poi_fused__to_copy_add_mul_transpose_14 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = 2.8769
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr2 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7z/c7zjybk3qfg43jk4dp6bfxozsh6sl22ufitdm65lallosljd376e.py
# Topologically Sorted Source Nodes: [mul_13], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_13 => mul_13
# Graph fragment:
#   %expand_27 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %mul_13 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, 1.2012), kwargs = {})
#   return %expand_26
triton_poi_fused_mul_15 = async_compile.triton('triton_poi_fused_mul_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 1.2012
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7s/c7sy264at5bjm3hxwqvsylrnoxxheqrfgkkyfih2xlg4sqx7qfdw.py
# Topologically Sorted Source Nodes: [mul_12, B_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_4 => add_9
#   mul_12 => mul_12
# Graph fragment:
#   %expand_27 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %bmm_13 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %mul_12 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, -3.0525), kwargs = {})
#   %add_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %bmm_13), kwargs = {})
#   return %expand_28
triton_poi_fused_add_mul_16 = async_compile.triton('triton_poi_fused_add_mul_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -3.0525
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/yd/cyd5yo25zyzf4opsc4apjzgjhdqmxjvnzhq2au527gtg5ygkfh4j.py
# Topologically Sorted Source Nodes: [mul_11, X_5, mul_14, X_6], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_5 => add_8
#   X_6 => add_10
#   mul_11 => mul_11
#   mul_14 => mul_14
# Graph fragment:
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_6]
#   %bmm_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %mul_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 2.8769), kwargs = {})
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %bmm_11), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 2.8366), kwargs = {})
#   %add_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %bmm_14), kwargs = {})
#   return %add_10
triton_poi_fused_add_mul_17 = async_compile.triton('triton_poi_fused_add_mul_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = 2.8769
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = 2.8366
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tl.store(in_out_ptr0 + (x0), tmp10, None)
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
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (32, 192, 192), (36864, 192, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_linalg_vector_norm_0.run(arg0_1, buf0, 160, 7373, stream=stream0)
            buf1 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf0, buf1, 32, 5, stream=stream0)
            del buf0
            buf2 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf3 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf8 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, A, transpose, matmul_2], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_2.run(arg0_1, buf1, buf2, buf3, buf8, 1179648, stream=stream0)
            buf4 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, A, transpose], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf2, buf3, out=buf4)
            buf5 = reinterpret_tensor(buf3, (32, 192, 192), (36864, 192, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_3.run(buf4, buf5, 1179648, stream=stream0)
            buf6 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [mul_1, matmul_1], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf5, buf4, out=buf6)
            buf7 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [mul, B], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf7, buf6, 1179648, stream=stream0)
            buf9 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul, B, matmul_2], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.bmm]
            extern_kernels.bmm(buf7, buf8, out=buf9)
            buf10 = buf8; del buf8  # reuse
            buf11 = reinterpret_tensor(buf7, (32, 192, 192), (36864, 1, 192), 0); del buf7  # reuse
            buf16 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, A_1, transpose_1, matmul_5], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_5.run(arg0_1, buf1, buf9, buf10, buf11, buf16, 1179648, stream=stream0)
            buf12 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, A_1, transpose_1], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf10, buf11, out=buf12)
            buf13 = reinterpret_tensor(buf11, (32, 192, 192), (36864, 192, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [mul_4], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_6.run(buf12, buf13, 1179648, stream=stream0)
            buf14 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [mul_4, matmul_4], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf13, buf12, out=buf14)
            buf15 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [mul_3, B_1], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf15, buf14, 1179648, stream=stream0)
            buf17 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_3, B_1, matmul_5], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.bmm]
            extern_kernels.bmm(buf15, buf16, out=buf17)
            buf18 = buf16; del buf16  # reuse
            buf19 = reinterpret_tensor(buf15, (32, 192, 192), (36864, 1, 192), 0); del buf15  # reuse
            buf24 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_5, X_3, A_2, transpose_2, matmul_8], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_8.run(arg0_1, buf1, buf9, buf17, buf18, buf19, buf24, 1179648, stream=stream0)
            buf20 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_5, X_3, A_2, transpose_2], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf18, buf19, out=buf20)
            buf21 = reinterpret_tensor(buf19, (32, 192, 192), (36864, 192, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [mul_7], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_9.run(buf20, buf21, 1179648, stream=stream0)
            buf22 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [mul_7, matmul_7], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf21, buf20, out=buf22)
            buf23 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [mul_6, B_2], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_10.run(buf23, buf22, 1179648, stream=stream0)
            buf25 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_5, X_3, mul_6, B_2, matmul_8], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.bmm]
            extern_kernels.bmm(buf23, buf24, out=buf25)
            buf26 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf27 = buf24; del buf24  # reuse
            buf28 = reinterpret_tensor(buf23, (32, 192, 192), (36864, 1, 192), 0); del buf23  # reuse
            buf33 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [X, norm, add, X_1, mul_2, X_2, mul_5, X_3, mul_8, X_4, A_3, transpose_3, matmul_11], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_11.run(arg0_1, buf1, buf9, buf17, buf25, buf26, buf27, buf28, buf33, 1179648, stream=stream0)
            del arg0_1
            del buf1
            del buf17
            buf29 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [A_3, transpose_3], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf27, buf28, out=buf29)
            buf30 = reinterpret_tensor(buf28, (32, 192, 192), (36864, 192, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [mul_10], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_12.run(buf29, buf30, 1179648, stream=stream0)
            buf31 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [mul_10, matmul_10], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf30, buf29, out=buf31)
            buf32 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [mul_9, B_3], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_13.run(buf32, buf31, 1179648, stream=stream0)
            buf34 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [mul_9, B_3, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf32, buf33, out=buf34)
            buf35 = buf33; del buf33  # reuse
            buf36 = reinterpret_tensor(buf32, (32, 192, 192), (36864, 1, 192), 0); del buf32  # reuse
            buf41 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [mul_11, X_5, A_4, transpose_4, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_14.run(buf26, buf34, buf35, buf36, buf41, 1179648, stream=stream0)
            buf37 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [mul_11, X_5, A_4, transpose_4], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf35, buf36, out=buf37)
            buf38 = reinterpret_tensor(buf36, (32, 192, 192), (36864, 192, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [mul_13], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_15.run(buf37, buf38, 1179648, stream=stream0)
            buf39 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [mul_13, matmul_13], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf38, buf37, out=buf39)
            del buf38
            buf40 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [mul_12, B_4], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_16.run(buf40, buf39, 1179648, stream=stream0)
            buf42 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [mul_11, X_5, mul_12, B_4, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf40, buf41, out=buf42)
            del buf40
            del buf41
            buf43 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [mul_11, X_5, mul_14, X_6], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf43, buf34, buf42, 1179648, stream=stream0)
            del buf34
            del buf42
        return (buf43, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
