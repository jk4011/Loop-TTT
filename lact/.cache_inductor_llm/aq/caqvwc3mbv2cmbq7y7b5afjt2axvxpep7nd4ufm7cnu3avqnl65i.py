# AOT ID: ['1_forward']
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/r3/cr3dte3fgimn5b3jiguypgynen3tfwqelo7r7wznap3zixri3bgp.py
# Topologically Sorted Source Nodes: [ki, lr2i, lr0i, transpose_2, gate_before_act, mul_8, type_as_1, mul_9, type_as_2], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   gate_before_act => convert_element_type_13
#   ki => slice_1
#   lr0i => slice_6
#   lr2i => slice_5
#   mul_8 => mul_11
#   mul_9 => mul_12
#   transpose_2 => permute_3
#   type_as_1 => convert_element_type_29
#   type_as_2 => convert_element_type_32
# Graph fragment:
#   %primals_7 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_10 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %primals_9 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %slice_1 : Tensor "f32[8, 1024, 192][786432, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_7, 1, 0, 1024), kwargs = {})
#   %slice_5 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_9, 1, 0, 1024), kwargs = {})
#   %slice_6 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_10, 1, 0, 1024), kwargs = {})
#   %permute_3 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_1, [0, 2, 1]), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_3, torch.bfloat16), kwargs = {})
#   %mul_11 : Tensor "f32[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, %slice_6), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %mul_12 : Tensor "f32[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, %slice_5), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   return %convert_element_type_13,%convert_element_type_29,%convert_element_type_32
triton_poi_fused__to_copy_mul_slice_transpose_0 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_transpose_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_transpose_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_transpose_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x4 = xindex
    x3 = ((xindex // 192) % 1024)
    tmp0 = tl.load(in_ptr0 + (x0 + 786432*x1), None)
    tmp2 = tl.load(in_ptr1 + (3*x3 + 12288*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (3*x3 + 12288*x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp1, None)
    tl.store(out_ptr1 + (x4), tmp4, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ar/caraqsif6b7fx76hmhlgfno3pzz55zvcrsfqn47fwco5djz4dkxi.py
# Topologically Sorted Source Nodes: [w1_norm, bmm_2, transpose_4, dhidden], Original ATen: [aten.linalg_vector_norm, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   bmm_2 => convert_element_type_10
#   dhidden => convert_element_type_21
#   transpose_4 => permute_5
#   w1_norm => pow_3, pow_4, sum_2
# Graph fragment:
#   %primals_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %sum_2 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_2]
#   %pow_3 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 2), kwargs = {})
#   %sum_2 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [2], True), kwargs = {})
#   %pow_4 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   %permute_5 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_2, [0, 2, 1]), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_5, torch.bfloat16), kwargs = {})
#   return %sum_2,%convert_element_type_10,%convert_element_type_21,%pow_4
triton_per_fused__to_copy_linalg_vector_norm_transpose_1 = async_compile.triton('triton_per_fused__to_copy_linalg_vector_norm_transpose_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_linalg_vector_norm_transpose_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288, 'r0_': 3538944}}
)
@triton.jit
def triton_per_fused__to_copy_linalg_vector_norm_transpose_1(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = tmp0.to(tl.float32)
    tmp7 = libdevice.sqrt(tmp5)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp6, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp6, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/po/cpo74tywej7zehyvaxpqzmz3t3kvzodcao76yjquklq2b3qccoom.py
# Topologically Sorted Source Nodes: [lr1i, silu_1, hidden, dhidden_before_mul, dgate, sigma, mul_4, sub, mul_5, add, dx, transpose_5, mul_7, type_as], Original ATen: [aten.slice, aten.silu, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   dgate => mul_6
#   dhidden_before_mul => mul_5
#   dx => mul_9
#   hidden => mul_3
#   lr1i => slice_4
#   mul_4 => mul_7
#   mul_5 => mul_8
#   mul_7 => mul_10
#   sigma => sigmoid_3
#   silu_1 => convert_element_type_19, convert_element_type_20, mul_2, sigmoid_1
#   sub => sub
#   transpose_5 => permute_6
#   type_as => convert_element_type_26
# Graph fragment:
#   %bmm_5 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_3 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %bmm_4 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %primals_8 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %slice_4 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_8, 1, 0, 1024), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %bmm_4), kwargs = {})
#   %mul_5 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_5, %convert_element_type_20), kwargs = {})
#   %mul_6 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_5, %bmm_4), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_3,), kwargs = {})
#   %mul_7 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %sigmoid_3), kwargs = {})
#   %sub : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_3), kwargs = {})
#   %mul_8 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, %sub), kwargs = {})
#   %add : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1), kwargs = {})
#   %mul_9 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add), kwargs = {})
#   %permute_6 : Tensor "bf16[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_3, [0, 2, 1]), kwargs = {})
#   %mul_10 : Tensor "f32[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, %slice_4), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   return %mul_5,%mul_9,%convert_element_type_26
triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34603008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 1024)
    x3 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (3*x1 + 12288*x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.0
    tmp12 = tmp11 - tmp9
    tmp13 = tmp1 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tmp16 = tmp5 * tmp7
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/3t/c3tjhq2akqfsyedzenmrwb4o3caimkk2h2bofqzh7gimsqwopjsv.py
# Topologically Sorted Source Nodes: [m_i, m_i_1], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i => slice_10
#   m_i_1 => mean
# Graph fragment:
#   %primals_4 : Tensor "f32[8, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %buf25 : Tensor "f32[8, 1, 1][1, 8, 8]cuda:0" = PlaceHolder[target=buf25]
#   %slice_10 : Tensor "f32[8, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_4, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_10, [1], True), kwargs = {})
#   return %buf25,%mean
triton_per_fused_mean_slice_3 = async_compile.triton('triton_per_fused_mean_slice_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64, 'r0_': 32768}}
)
@triton.jit
def triton_per_fused_mean_slice_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/io/ciotycoamtkpdd5zy6d6q37ujyhcgv27tblq6k64c5kcmqaf6xvg.py
# Topologically Sorted Source Nodes: [dw1_momentum, dw0_momentum, mul_10, dw0_1, mul_11, dw1_1, X, norm_3, X_7, norm_4], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type_35
#   X_7 => convert_element_type_82
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   mul_10 => mul_13
#   mul_11 => mul_14
#   norm_3 => convert_element_type_36, pow_7, sum_4
#   norm_4 => convert_element_type_83, pow_9, sum_5
# Graph fragment:
#   %bmm_6 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %bmm_7 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %full_default : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %mul_14 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_36 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_35, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_36, 2), kwargs = {})
#   %sum_4 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_83 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_82, torch.float32), kwargs = {})
#   %pow_9 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_83, 2), kwargs = {})
#   %sum_5 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2], True), kwargs = {})
#   return %buf27,%buf66
triton_red_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_4 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 640, 'r0_': 1179680}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 40
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
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp5 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 0.0
        tmp7 = tmp6 * tmp5
        tmp8 = tmp4 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp17 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 + tmp7
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/or/coro5mhwbtk7gxtzaloxwzecw6chtrg7g3k4qaomqxp6hmgcfswr.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, X_7, norm_4], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_7 => convert_element_type_82
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   mul_10 => mul_13
#   norm_4 => convert_element_type_83, pow_10, pow_9, sum_5
# Graph fragment:
#   %buf66 : Tensor "f32[8, 1, 1, 5][5, 40, 40, 1]cuda:0" = PlaceHolder[target=buf66]
#   %sum_5 : Tensor "f32[8, 1, 1][1, 8, 8]cuda:0" = PlaceHolder[target=sum_5]
#   %full_default_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_83 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_82, torch.float32), kwargs = {})
#   %pow_9 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_83, 2), kwargs = {})
#   %sum_5 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2], True), kwargs = {})
#   %pow_10 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   return %sum_5,%pow_10
triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5 = async_compile.triton('triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64, 'r0_': 100}}
)
@triton.jit
def triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    tmp5 = libdevice.sqrt(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/57/c57h5xm2uiv3vd5fi2tjzjiui5nrhwhcnl5syss7nratkguc5z7c.py
# Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, X, add_4, X_1, transpose_6, A], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A => convert_element_type_37, convert_element_type_38
#   X => convert_element_type_35
#   X_1 => div
#   add_4 => add_4
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   mul_11 => mul_14
#   transpose_6 => permute_7
# Graph fragment:
#   %bmm_6 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %pow_8 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %full_default : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_14 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %add_4 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_35, %add_4), kwargs = {})
#   %permute_7 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1]), kwargs = {})
#   %convert_element_type_37 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.bfloat16), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %expand,%expand_1
triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_6 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2949120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 0.0
    tmp4 = tmp3 * tmp2
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 1e-07
    tmp10 = tmp8 + tmp9
    tmp11 = (tmp7 / tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/tu/ctuxwiqzwzzjjmb4c6hwesp7cuupcr4cz6cdh6tsxn43iq3gcan4.py
# Topologically Sorted Source Nodes: [mul_14], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_14 => mul_17
# Graph fragment:
#   %expand_3 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %mul_17 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 2.927), kwargs = {})
#   return %expand_2
triton_poi_fused_mul_7 = async_compile.triton('triton_poi_fused_mul_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1769472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.927
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/xm/cxmufm2xsirb2tnhza7sdoj2g5xcudda5bevib3jeydaz6kpyjjm.py
# Topologically Sorted Source Nodes: [mul_13, B], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B => add_5
#   mul_13 => mul_16
# Graph fragment:
#   %expand_3 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %bmm_10 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %mul_16 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, -6.8946), kwargs = {})
#   %add_5 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %bmm_10), kwargs = {})
#   return %expand_4
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -6.8946
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/oa/coatdmivohbjsyuaauwtlhxkvooqw6lgp7fgq6idjtf7suswebiq.py
# Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, X, add_4, X_1, mul_15, X_2, transpose_7, A_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_1 => convert_element_type_46, convert_element_type_47
#   X => convert_element_type_35
#   X_1 => div
#   X_2 => add_6
#   add_4 => add_4
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   mul_11 => mul_14
#   mul_15 => mul_18
#   transpose_7 => permute_8
# Graph fragment:
#   %bmm_6 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %pow_8 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %bmm_11 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %full_default : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_14 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %add_4 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_35, %add_4), kwargs = {})
#   %mul_18 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_6 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_11), kwargs = {})
#   %permute_8 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_6, [0, 2, 1]), kwargs = {})
#   %convert_element_type_46 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_8, torch.bfloat16), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   return %expand_6,%expand_7
triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_9 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 0.0
    tmp4 = tmp3 * tmp2
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 1e-07
    tmp10 = tmp8 + tmp9
    tmp11 = (tmp7 / tmp10)
    tmp12 = 4.0848
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/jr/cjrnz4spstg5r42lg2m7t4h7vwz2vbgeevbuexjhrihxhpv5hlpa.py
# Topologically Sorted Source Nodes: [mul_17], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_17 => mul_20
# Graph fragment:
#   %expand_9 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %mul_20 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, 2.6377), kwargs = {})
#   return %expand_8
triton_poi_fused_mul_10 = async_compile.triton('triton_poi_fused_mul_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1769472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.6377
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/nw/cnwuco2iwueg2ka4ae2crc7t7jrvesvf4a2dmmsczkzvgd3qyr5h.py
# Topologically Sorted Source Nodes: [mul_16, B_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_1 => add_7
#   mul_16 => mul_19
# Graph fragment:
#   %expand_9 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %bmm_13 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %mul_19 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, -6.3029), kwargs = {})
#   %add_7 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %bmm_13), kwargs = {})
#   return %expand_10
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -6.3029
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/mf/cmftppycnj3nwergtv57ulhlre7qbinaq3aswod4tlbrptcbphmq.py
# Topologically Sorted Source Nodes: [dw1_momentum, dw0_momentum, mul_10, dw0_1, mul_11, dw1_1, X, add_4, X_1, mul_15, X_2, mul_18, X_3, transpose_8, A_2, X_7, add_15, X_8, transpose_11, A_5], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_2 => convert_element_type_55, convert_element_type_56
#   A_5 => convert_element_type_84, convert_element_type_85
#   X => convert_element_type_35
#   X_1 => div
#   X_2 => add_6
#   X_3 => add_8
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
#   mul_15 => mul_18
#   mul_18 => mul_21
#   transpose_11 => permute_12
#   transpose_8 => permute_9
# Graph fragment:
#   %bmm_6 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %pow_8 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %bmm_11 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %add_8 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_8]
#   %bmm_7 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %pow_10 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_10]
#   %full_default : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %mul_14 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %add_4 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_35, %add_4), kwargs = {})
#   %mul_18 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_6 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_11), kwargs = {})
#   %mul_21 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 3.9505), kwargs = {})
#   %add_8 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %bmm_14), kwargs = {})
#   %permute_9 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_8, [0, 2, 1]), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_9, torch.bfloat16), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %add_15 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_10, 1e-07), kwargs = {})
#   %div_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_82, %add_15), kwargs = {})
#   %permute_12 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_1, [0, 2, 1]), kwargs = {})
#   %convert_element_type_84 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_12, torch.bfloat16), kwargs = {})
#   %convert_element_type_85 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   return %add_8,%expand_12,%expand_13,%expand_30,%expand_31
triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_12 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp28 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 0.0
    tmp4 = tmp3 * tmp2
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 1e-07
    tmp10 = tmp8 + tmp9
    tmp11 = (tmp7 / tmp10)
    tmp12 = 4.0848
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = 3.9505
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 + tmp4
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp28 + tmp9
    tmp30 = (tmp27 / tmp29)
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp22, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
    tl.store(out_ptr3 + (x2), tmp31, None)
    tl.store(out_ptr4 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/nf/cnfsjjfm2dry72d5rczvgusu47ivihrjqc3yazib6cclfhhjhus3.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, dw2_1, X_7, add_15, X_8, mul_30, X_9, mul_33, X_10, transpose_13, A_7, X_14], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_7 => convert_element_type_102, convert_element_type_103
#   X_10 => add_19
#   X_14 => convert_element_type_129
#   X_7 => convert_element_type_82
#   X_8 => div_1
#   X_9 => add_17
#   add_15 => add_15
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw2_1 => add_3
#   mul_10 => mul_13
#   mul_30 => mul_33
#   mul_33 => mul_36
#   transpose_13 => permute_14
# Graph fragment:
#   %bmm_7 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %pow_10 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_10]
#   %bmm_26 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_26]
#   %bmm_29 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_29]
#   %add_19 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_19]
#   %bmm_8 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %full_default_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %add_3 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_13), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %add_15 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_10, 1e-07), kwargs = {})
#   %div_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_82, %add_15), kwargs = {})
#   %mul_33 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, 4.0848), kwargs = {})
#   %add_17 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %bmm_26), kwargs = {})
#   %mul_36 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, 3.9505), kwargs = {})
#   %add_19 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %bmm_29), kwargs = {})
#   %permute_14 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_19, [0, 2, 1]), kwargs = {})
#   %convert_element_type_102 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_14, torch.bfloat16), kwargs = {})
#   %convert_element_type_103 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.bfloat16), kwargs = {})
#   %convert_element_type_129 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   return %add_19,%expand_42,%expand_43,%convert_element_type_129
triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_13 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8257536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 0.0
    tmp4 = tmp3 * tmp2
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 1e-07
    tmp10 = tmp8 + tmp9
    tmp11 = (tmp7 / tmp10)
    tmp12 = 4.0848
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = 3.9505
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 + tmp4
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp22, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
    tl.store(out_ptr3 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ql/cqlxc3b7bbjjwsuipumvtqewcnixpyq6go2rgkpxtfehhzasdmnh.py
# Topologically Sorted Source Nodes: [mul_35], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_35 => mul_38
# Graph fragment:
#   %expand_45 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_45]
#   %mul_38 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_30, 2.3037), kwargs = {})
#   return %expand_44
triton_poi_fused_mul_14 = async_compile.triton('triton_poi_fused_mul_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1769472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.3037
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/qw/cqwi7xac27ysmogkwfeucacl4jgzbcuzylnhw6wvhwjcnqdyxjns.py
# Topologically Sorted Source Nodes: [mul_34, B_7], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_7 => add_20
#   mul_34 => mul_37
# Graph fragment:
#   %expand_45 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_45]
#   %bmm_31 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_31]
#   %mul_37 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_30, -5.5913), kwargs = {})
#   %add_20 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %bmm_31), kwargs = {})
#   return %expand_46
triton_poi_fused_add_mul_15 = async_compile.triton('triton_poi_fused_add_mul_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -5.5913
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ci/ccinyhjpaky7ie7pzdajumm6j2sz64lyeyvkermq6visicl7oqze.py
# Topologically Sorted Source Nodes: [mul_36, X_11, transpose_14, A_8], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_8 => convert_element_type_111, convert_element_type_112
#   X_11 => add_21
#   mul_36 => mul_39
#   transpose_14 => permute_15
# Graph fragment:
#   %add_19 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_19]
#   %bmm_32 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_32]
#   %mul_39 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, 3.7418), kwargs = {})
#   %add_21 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %bmm_32), kwargs = {})
#   %permute_15 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_21, [0, 2, 1]), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_15, torch.bfloat16), kwargs = {})
#   %convert_element_type_112 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.bfloat16), kwargs = {})
#   return %expand_48,%expand_49
triton_poi_fused__to_copy_add_mul_transpose_16 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4128768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = 3.7418
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/5u/c5uicwej6akygl546pg5u34tqrkhbswzicyy5yq4f4cb7pc3j3c4.py
# Topologically Sorted Source Nodes: [mul_38], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_38 => mul_41
# Graph fragment:
#   %expand_51 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_51]
#   %mul_41 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_33, 1.2046), kwargs = {})
#   return %expand_50
triton_poi_fused_mul_17 = async_compile.triton('triton_poi_fused_mul_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1769472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 1.2046
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ez/ceziuuvldpq7yaraq24dv6wflogipemklenbfs6s65ztgsynmocl.py
# Topologically Sorted Source Nodes: [mul_37, B_8], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_8 => add_22
#   mul_37 => mul_40
# Graph fragment:
#   %expand_51 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_51]
#   %bmm_34 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_34]
#   %mul_40 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_33, -3.1427), kwargs = {})
#   %add_22 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, %bmm_34), kwargs = {})
#   return %expand_52
triton_poi_fused_add_mul_18 = async_compile.triton('triton_poi_fused_add_mul_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -3.1427
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ay/cay7ebf2fet36tylvj6fruyub56q3ec5fpu2t6posq5wozn6f3ok.py
# Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, transpose_15, A_9], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_9 => convert_element_type_120, convert_element_type_121
#   X_11 => add_21
#   X_12 => add_23
#   mul_36 => mul_39
#   mul_39 => mul_42
#   transpose_15 => permute_16
# Graph fragment:
#   %add_19 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_19]
#   %bmm_32 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_32]
#   %bmm_35 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_35]
#   %mul_39 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, 3.7418), kwargs = {})
#   %add_21 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %bmm_32), kwargs = {})
#   %mul_42 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 2.8769), kwargs = {})
#   %add_23 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %bmm_35), kwargs = {})
#   %permute_16 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_23, [0, 2, 1]), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_16, torch.bfloat16), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_23, torch.bfloat16), kwargs = {})
#   return %expand_54,%expand_55
triton_poi_fused__to_copy_add_mul_transpose_19 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp1 = 3.7418
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = 2.8769
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr1 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ox/coxxrpzlzgj3etplvgkuuymcqwcxjggkb32ejaincpc5uppiyvjb.py
# Topologically Sorted Source Nodes: [mul_41], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_41 => mul_44
# Graph fragment:
#   %expand_57 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_57]
#   %mul_44 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_36, 1.2012), kwargs = {})
#   return %expand_56
triton_poi_fused_mul_20 = async_compile.triton('triton_poi_fused_mul_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1769472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 1.2012
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/j5/cj5kxav3gdmansoadfaud6jdfpqqn2qgfcek3ef5ockdxsxnbklw.py
# Topologically Sorted Source Nodes: [mul_40, B_9], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_9 => add_24
#   mul_40 => mul_43
# Graph fragment:
#   %expand_57 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_57]
#   %bmm_37 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_37]
#   %mul_43 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_36, -3.0525), kwargs = {})
#   %add_24 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %bmm_37), kwargs = {})
#   return %expand_58
triton_poi_fused_add_mul_21 = async_compile.triton('triton_poi_fused_add_mul_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = -3.0525
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/z2/cz2txmfxx2i6appgz33bqnt6iqp4u346zprnn2id4z6iwnge73n2.py
# Topologically Sorted Source Nodes: [w0_norm, mul_36, X_11, mul_39, X_12, mul_42, X_13, w0_main, norm_6, add_40, truediv_3, w0, bmm_10], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_11 => add_21
#   X_12 => add_23
#   X_13 => add_25
#   add_40 => add_40
#   bmm_10 => convert_element_type_181
#   mul_36 => mul_39
#   mul_39 => mul_42
#   mul_42 => mul_45
#   norm_6 => pow_13, pow_14, sum_7
#   truediv_3 => div_3
#   w0 => mul_61
#   w0_main => add_38
#   w0_norm => convert_element_type, pow_1, pow_2, sum_1
# Graph fragment:
#   %primals_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %sum_1 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_1]
#   %add_19 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_19]
#   %bmm_32 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_32]
#   %bmm_35 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_35]
#   %bmm_38 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_38]
#   %add_38 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_38]
#   %sum_7 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_7]
#   %pow_14 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_14]
#   %pow_2 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_2]
#   %convert_element_type : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %sum_1 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2], True), kwargs = {})
#   %pow_2 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mul_39 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, 3.7418), kwargs = {})
#   %add_21 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %bmm_32), kwargs = {})
#   %mul_42 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 2.8769), kwargs = {})
#   %add_23 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %bmm_35), kwargs = {})
#   %mul_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, 2.8366), kwargs = {})
#   %add_25 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %bmm_38), kwargs = {})
#   %add_38 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %add_25), kwargs = {})
#   %pow_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_38, 2), kwargs = {})
#   %sum_7 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [2], True), kwargs = {})
#   %pow_14 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %add_40 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_14, 1e-05), kwargs = {})
#   %div_3 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_38, %add_40), kwargs = {})
#   %mul_61 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %pow_2), kwargs = {})
#   %convert_element_type_181 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_61, torch.bfloat16), kwargs = {})
#   return %sum_1,%pow_2,%add_38,%sum_7,%pow_14,%convert_element_type_181
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_22 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 7077888}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_22(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp8 = tl.load(in_out_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = libdevice.sqrt(tmp6)
    tmp9 = 3.7418
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = 2.8769
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = 2.8366
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 + tmp22
    tmp24 = tmp1 + tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = libdevice.sqrt(tmp29)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = (tmp24 / tmp32)
    tmp34 = tmp33 * tmp7
    tmp35 = tmp34.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
    tl.store(in_out_ptr1 + (r0_1 + 192*x0), tmp24, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp30, xmask)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/3y/c3yknknafnqut7u3zykhjrcyvk5fhqyxcbopzlryjc2okpa2boyq.py
# Topologically Sorted Source Nodes: [norm_5], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm_5 => convert_element_type_130, pow_11, sum_6
# Graph fragment:
#   %convert_element_type_129 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_129]
#   %convert_element_type_130 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_129, torch.float32), kwargs = {})
#   %pow_11 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_130, 2), kwargs = {})
#   %sum_6 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2], True), kwargs = {})
#   return %buf106
triton_red_fused_linalg_vector_norm_23 = async_compile.triton('triton_red_fused_linalg_vector_norm_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 320, 'r0_': 589840}}
)
@triton.jit
def triton_red_fused_linalg_vector_norm_23(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 40
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
        tmp3 = tl.load(in_ptr0 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/wv/cwv5ov5ilhe4o2a73njchpipbptuyukkfnwq3vyknpdyycdqubqb.py
# Topologically Sorted Source Nodes: [add_26, X_15, transpose_16, A_10], Original ATen: [aten.add, aten.div, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_10 => convert_element_type_131, convert_element_type_132
#   X_15 => div_2
#   add_26 => add_26
#   transpose_16 => permute_17
# Graph fragment:
#   %convert_element_type_129 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_129]
#   %pow_12 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_12]
#   %add_26 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, 1e-07), kwargs = {})
#   %div_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_129, %add_26), kwargs = {})
#   %permute_17 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_2, [0, 2, 1]), kwargs = {})
#   %convert_element_type_131 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_17, torch.bfloat16), kwargs = {})
#   %convert_element_type_132 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.bfloat16), kwargs = {})
#   return %expand_60,%expand_61
triton_poi_fused__to_copy_add_div_transpose_24 = async_compile.triton('triton_poi_fused__to_copy_add_div_transpose_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_transpose_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2949120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_transpose_24(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1e-07
    tmp4 = tmp2 + tmp3
    tmp5 = (tmp1 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/xe/cxegcnpu3z3hr4h5s27hdvu4jn4kflleaeqkqgcatbdlv247wx3z.py
# Topologically Sorted Source Nodes: [add_26, X_15, mul_45, X_16, transpose_17, A_11], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_11 => convert_element_type_140, convert_element_type_141
#   X_15 => div_2
#   X_16 => add_28
#   add_26 => add_26
#   mul_45 => mul_48
#   transpose_17 => permute_18
# Graph fragment:
#   %convert_element_type_129 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_129]
#   %pow_12 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_12]
#   %bmm_41 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_41]
#   %add_26 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, 1e-07), kwargs = {})
#   %div_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_129, %add_26), kwargs = {})
#   %mul_48 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 4.0848), kwargs = {})
#   %add_28 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %bmm_41), kwargs = {})
#   %permute_18 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_28, [0, 2, 1]), kwargs = {})
#   %convert_element_type_140 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_18, torch.bfloat16), kwargs = {})
#   %convert_element_type_141 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_28, torch.bfloat16), kwargs = {})
#   return %expand_66,%expand_67
triton_poi_fused__to_copy_add_div_mul_transpose_25 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1e-07
    tmp4 = tmp2 + tmp3
    tmp5 = (tmp1 / tmp4)
    tmp6 = 4.0848
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/yq/cyq6a5t4gdusbm456cx5oog2ai7iqiezsfrdhxyytbc2ffd4yhqi.py
# Topologically Sorted Source Nodes: [add_26, X_15, mul_45, X_16, mul_48, X_17, transpose_18, A_12], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_12 => convert_element_type_149, convert_element_type_150
#   X_15 => div_2
#   X_16 => add_28
#   X_17 => add_30
#   add_26 => add_26
#   mul_45 => mul_48
#   mul_48 => mul_51
#   transpose_18 => permute_19
# Graph fragment:
#   %convert_element_type_129 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_129]
#   %pow_12 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_12]
#   %bmm_41 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_41]
#   %bmm_44 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_44]
#   %add_26 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, 1e-07), kwargs = {})
#   %div_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_129, %add_26), kwargs = {})
#   %mul_48 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 4.0848), kwargs = {})
#   %add_28 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %bmm_41), kwargs = {})
#   %mul_51 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 3.9505), kwargs = {})
#   %add_30 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %bmm_44), kwargs = {})
#   %permute_19 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_30, [0, 2, 1]), kwargs = {})
#   %convert_element_type_149 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_19, torch.bfloat16), kwargs = {})
#   %convert_element_type_150 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.bfloat16), kwargs = {})
#   return %expand_72,%expand_73
triton_poi_fused__to_copy_add_div_mul_transpose_26 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4128768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1e-07
    tmp4 = tmp2 + tmp3
    tmp5 = (tmp1 / tmp4)
    tmp6 = 4.0848
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = 3.9505
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/vr/cvrrnb5rnrn2nom3majvbvxo5sgb4n3swz3vd5gzmogsow4f6uyf.py
# Topologically Sorted Source Nodes: [add_26, X_15, mul_45, X_16, mul_48, X_17, mul_51, X_18, transpose_19, A_13], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_13 => convert_element_type_158, convert_element_type_159
#   X_15 => div_2
#   X_16 => add_28
#   X_17 => add_30
#   X_18 => add_32
#   add_26 => add_26
#   mul_45 => mul_48
#   mul_48 => mul_51
#   mul_51 => mul_54
#   transpose_19 => permute_20
# Graph fragment:
#   %convert_element_type_129 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_129]
#   %pow_12 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_12]
#   %bmm_41 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_41]
#   %bmm_44 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_44]
#   %bmm_47 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_47]
#   %add_32 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_32]
#   %add_26 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, 1e-07), kwargs = {})
#   %div_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_129, %add_26), kwargs = {})
#   %mul_48 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 4.0848), kwargs = {})
#   %add_28 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %bmm_41), kwargs = {})
#   %mul_51 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 3.9505), kwargs = {})
#   %add_30 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %bmm_44), kwargs = {})
#   %mul_54 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, 3.7418), kwargs = {})
#   %add_32 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %bmm_47), kwargs = {})
#   %permute_20 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %convert_element_type_158 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_20, torch.bfloat16), kwargs = {})
#   %convert_element_type_159 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.bfloat16), kwargs = {})
#   return %add_32,%expand_78,%expand_79
triton_poi_fused__to_copy_add_div_mul_transpose_27 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1e-07
    tmp4 = tmp2 + tmp3
    tmp5 = (tmp1 / tmp4)
    tmp6 = 4.0848
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = 3.9505
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = 3.7418
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/z5/cz5werywyqq7aoxqp57av5ugifls5demio2nkb46w2jopnisbxl2.py
# Topologically Sorted Source Nodes: [mul_54, X_19, transpose_20, A_14], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   A_14 => convert_element_type_167, convert_element_type_168
#   X_19 => add_34
#   mul_54 => mul_57
#   transpose_20 => permute_21
# Graph fragment:
#   %add_32 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_32]
#   %bmm_50 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_50]
#   %mul_57 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 2.8769), kwargs = {})
#   %add_34 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %bmm_50), kwargs = {})
#   %permute_21 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_34, [0, 2, 1]), kwargs = {})
#   %convert_element_type_167 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_21, torch.bfloat16), kwargs = {})
#   %convert_element_type_168 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_34, torch.bfloat16), kwargs = {})
#   return %expand_84,%expand_85
triton_poi_fused__to_copy_add_mul_transpose_28 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4128768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
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
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/c6/cc63xa47i4o4onmrowtp7pxiboif2rkjnlsg42tvy463xt6bfy56.py
# Topologically Sorted Source Nodes: [w2_norm, mul_54, X_19, mul_57, X_20, w2_main, norm_8, add_42, truediv_5, w2, h_1], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_19 => add_34
#   X_20 => add_36
#   add_42 => add_42
#   h_1 => convert_element_type_177
#   mul_54 => mul_57
#   mul_57 => mul_60
#   norm_8 => pow_17, pow_18, sum_9
#   truediv_5 => div_5
#   w2 => mul_63
#   w2_main => add_39
#   w2_norm => convert_element_type_1, pow_5, pow_6, sum_3
# Graph fragment:
#   %primals_3 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %sum_3 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_3]
#   %add_32 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_32]
#   %bmm_50 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_50]
#   %bmm_53 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_53]
#   %add_39 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_39]
#   %sum_9 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_9]
#   %pow_18 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_18]
#   %pow_6 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_6]
#   %convert_element_type_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_3 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [2], True), kwargs = {})
#   %pow_6 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %mul_57 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 2.8769), kwargs = {})
#   %add_34 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %bmm_50), kwargs = {})
#   %mul_60 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 2.8366), kwargs = {})
#   %add_36 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_60, %bmm_53), kwargs = {})
#   %add_39 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %add_36), kwargs = {})
#   %pow_17 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_39, 2), kwargs = {})
#   %sum_9 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_17, [2], True), kwargs = {})
#   %pow_18 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_9, 0.5), kwargs = {})
#   %add_42 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_18, 1e-05), kwargs = {})
#   %div_5 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, %add_42), kwargs = {})
#   %mul_63 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %pow_6), kwargs = {})
#   %convert_element_type_177 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   return %sum_3,%pow_6,%add_39,%sum_9,%pow_18,%convert_element_type_177
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_29 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 6488064}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_29(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp8 = tl.load(in_out_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = libdevice.sqrt(tmp6)
    tmp9 = 2.8769
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = 2.8366
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp1 + tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = (tmp19 / tmp27)
    tmp29 = tmp28 * tmp7
    tmp30 = tmp29.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
    tl.store(in_out_ptr1 + (r0_1 + 192*x0), tmp19, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp30, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/77/c77ehffv4dukwyqaeazwknejv7v2bhtmvvpegzd4zhm6tqgazt2w.py
# Topologically Sorted Source Nodes: [q, qi, h], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
# Source node to ATen node mapping:
#   h => convert_element_type_2
#   q => permute
#   qi => slice_3
# Graph fragment:
#   %primals_5 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %permute : Tensor "f32[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%primals_5, [0, 2, 1]), kwargs = {})
#   %slice_3 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute, 2, 0, 1024), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_slice_transpose_30 = async_compile.triton('triton_poi_fused__to_copy_slice_transpose_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_transpose_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_slice_transpose_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 786432*x1), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ls/clsqra3wdwenkj5ocrentia3gzpgyd4tiqbbstybvsmo7ng7327d.py
# Topologically Sorted Source Nodes: [gate, mul], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   gate => convert_element_type_8, convert_element_type_9, mul, sigmoid
#   mul => mul_1
# Graph fragment:
#   %bmm_1 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm]
#   %convert_element_type_8 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_1, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_8,), kwargs = {})
#   %mul : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %sigmoid), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, %bmm), kwargs = {})
#   return %mul_1
triton_poi_fused_mul_silu_31 = async_compile.triton('triton_poi_fused_mul_silu_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/sh/csh3447a6b3jg23rny4b4pfufqykhs56omhn644u23hcvg4k4ycf.py
# Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, mul_27, X_6, w1_main, norm_7, add_41, truediv_4, w1, bmm_11, transpose_23, dhidden_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   X_4 => add_10
#   X_5 => add_12
#   X_6 => add_14
#   add_41 => add_41
#   bmm_11 => convert_element_type_186
#   dhidden_1 => convert_element_type_199
#   mul_21 => mul_24
#   mul_24 => mul_27
#   mul_27 => mul_30
#   norm_7 => pow_15, pow_16, sum_8
#   transpose_23 => permute_24
#   truediv_4 => div_4
#   w1 => mul_62
#   w1_main => add_37
# Graph fragment:
#   %primals_2 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %add_8 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_8]
#   %bmm_17 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %bmm_20 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %bmm_23 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_23]
#   %add_37 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_37]
#   %sum_8 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_8]
#   %pow_16 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_16]
#   %pow_4 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_4]
#   %mul_24 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.7418), kwargs = {})
#   %add_10 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %bmm_17), kwargs = {})
#   %mul_27 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.8769), kwargs = {})
#   %add_12 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %bmm_20), kwargs = {})
#   %mul_30 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 2.8366), kwargs = {})
#   %add_14 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %bmm_23), kwargs = {})
#   %add_37 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %add_14), kwargs = {})
#   %pow_15 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_37, 2), kwargs = {})
#   %sum_8 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_15, [2], True), kwargs = {})
#   %pow_16 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_8, 0.5), kwargs = {})
#   %add_41 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_16, 1e-05), kwargs = {})
#   %div_4 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_37, %add_41), kwargs = {})
#   %mul_62 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %pow_4), kwargs = {})
#   %convert_element_type_186 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_62, torch.bfloat16), kwargs = {})
#   %permute_24 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_62, [0, 2, 1]), kwargs = {})
#   %convert_element_type_199 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_24, torch.bfloat16), kwargs = {})
#   return %add_37,%sum_8,%pow_16,%convert_element_type_186,%convert_element_type_199
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_32 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 8847360}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_32(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 3.7418
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = 2.8769
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 2.8366
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp0 + tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = (tmp17 / tmp25)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp17, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp29, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp29, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/rn/crnaxp44qma2ikhnnxntt6dsgxlncmfkog7alhwmxkss47gsngd4.py
# Topologically Sorted Source Nodes: [q, qi_1, h_1], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
# Source node to ATen node mapping:
#   h_1 => convert_element_type_176
#   q => permute
#   qi_1 => slice_13
# Graph fragment:
#   %primals_5 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %permute : Tensor "f32[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%primals_5, [0, 2, 1]), kwargs = {})
#   %slice_13 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute, 2, 1024, 2048), kwargs = {})
#   %convert_element_type_176 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_13, torch.bfloat16), kwargs = {})
#   return %convert_element_type_176
triton_poi_fused__to_copy_slice_transpose_33 = async_compile.triton('triton_poi_fused__to_copy_slice_transpose_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_transpose_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_slice_transpose_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (196608 + x0 + 786432*x1), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/lk/clktdwkmcjhews3kcyypvrjo4evkbolknsk4avnkvzvh2etyjoa3.py
# Topologically Sorted Source Nodes: [ki_1, lr2i_1, lr0i_1, transpose_21, gate_before_act_1, mul_69, type_as_4, mul_70, type_as_5], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   gate_before_act_1 => convert_element_type_189
#   ki_1 => slice_11
#   lr0i_1 => slice_16
#   lr2i_1 => slice_15
#   mul_69 => mul_75
#   mul_70 => mul_76
#   transpose_21 => permute_22
#   type_as_4 => convert_element_type_207
#   type_as_5 => convert_element_type_210
# Graph fragment:
#   %primals_7 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_10 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %primals_9 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %slice_11 : Tensor "f32[8, 1024, 192][786432, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_7, 1, 1024, 2048), kwargs = {})
#   %slice_15 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_9, 1, 1024, 2048), kwargs = {})
#   %slice_16 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_10, 1, 1024, 2048), kwargs = {})
#   %permute_22 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_11, [0, 2, 1]), kwargs = {})
#   %convert_element_type_189 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_22, torch.bfloat16), kwargs = {})
#   %mul_75 : Tensor "f32[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_11, %slice_16), kwargs = {})
#   %convert_element_type_207 : Tensor "bf16[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   %mul_76 : Tensor "f32[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_11, %slice_15), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_76, torch.bfloat16), kwargs = {})
#   return %convert_element_type_189,%convert_element_type_207,%convert_element_type_210
triton_poi_fused__to_copy_mul_slice_transpose_34 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_transpose_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_transpose_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_transpose_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x4 = xindex
    x3 = ((xindex // 192) % 1024)
    tmp0 = tl.load(in_ptr0 + (196608 + x0 + 786432*x1), None)
    tmp2 = tl.load(in_ptr1 + (3072 + 3*x3 + 12288*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (3072 + 3*x3 + 12288*x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp1, None)
    tl.store(out_ptr1 + (x4), tmp4, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/cp/ccpwlycrvlbyyec7yoohwkgrnvhz67ykgtf5ozfdrq6yce472l3q.py
# Topologically Sorted Source Nodes: [lr1i_1, silu_4, hidden_1, dhidden_before_mul_1, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43, dx_1, transpose_24, mul_68, type_as_3], Original ATen: [aten.slice, aten.silu, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   add_43 => add_43
#   dgate_1 => mul_70
#   dhidden_before_mul_1 => mul_69
#   dx_1 => mul_73
#   hidden_1 => mul_67
#   lr1i_1 => slice_14
#   mul_65 => mul_71
#   mul_66 => mul_72
#   mul_68 => mul_74
#   sigma_1 => sigmoid_7
#   silu_4 => convert_element_type_197, convert_element_type_198, mul_66, sigmoid_5
#   sub_1 => sub_1
#   transpose_24 => permute_25
#   type_as_3 => convert_element_type_204
# Graph fragment:
#   %bmm_59 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %bmm_57 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %bmm_58 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_58]
#   %primals_8 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %slice_14 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_8, 1, 1024, 2048), kwargs = {})
#   %convert_element_type_197 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_57, torch.float32), kwargs = {})
#   %sigmoid_5 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_197,), kwargs = {})
#   %mul_66 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_197, %sigmoid_5), kwargs = {})
#   %convert_element_type_198 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_66, torch.bfloat16), kwargs = {})
#   %mul_67 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_198, %bmm_58), kwargs = {})
#   %mul_69 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_59, %convert_element_type_198), kwargs = {})
#   %mul_70 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_59, %bmm_58), kwargs = {})
#   %sigmoid_7 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_57,), kwargs = {})
#   %mul_71 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %sigmoid_7), kwargs = {})
#   %sub_1 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_7), kwargs = {})
#   %mul_72 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_57, %sub_1), kwargs = {})
#   %add_43 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, 1), kwargs = {})
#   %mul_73 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_71, %add_43), kwargs = {})
#   %permute_25 : Tensor "bf16[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_67, [0, 2, 1]), kwargs = {})
#   %mul_74 : Tensor "f32[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_25, %slice_14), kwargs = {})
#   %convert_element_type_204 : Tensor "bf16[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_74, torch.bfloat16), kwargs = {})
#   return %mul_69,%mul_73,%convert_element_type_204
triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_35 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34603008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 1024)
    x3 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (3072 + 3*x1 + 12288*x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.0
    tmp12 = tmp11 - tmp9
    tmp13 = tmp1 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tmp16 = tmp5 * tmp7
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/2m/c2mzcgmndrz7jtgw4p52kdirvocqyncyospo6esoknblp4yexbfn.py
# Topologically Sorted Source Nodes: [m_i_2, m_i_3], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i_2 => slice_21
#   m_i_3 => mean_1
# Graph fragment:
#   %primals_4 : Tensor "f32[8, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %buf175 : Tensor "f32[8, 1, 1][1, 8, 8]cuda:0" = PlaceHolder[target=buf175]
#   %slice_21 : Tensor "f32[8, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_4, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_21, [1], True), kwargs = {})
#   return %buf175,%mean_1
triton_per_fused_mean_slice_36 = async_compile.triton('triton_per_fused_mean_slice_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64, 'r0_': 32768}}
)
@triton.jit
def triton_per_fused_mean_slice_36(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + r0_1 + 4096*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/hc/chcbwurxzuaicshyryjloz7mg6e6yvwbl47b4zs4ibs63fv3neal.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, mul_71, dw0_3], Original ATen: [aten.zeros_like, aten.mul, aten.add]
# Source node to ATen node mapping:
#   dw0_1 => add_1
#   dw0_3 => add_44
#   dw0_momentum => full_default_1
#   mul_10 => mul_13
#   mul_71 => mul_77
# Graph fragment:
#   %bmm_61 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_61]
#   %bmm_7 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %mean_1 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_1]
#   %full_default_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %mul_77 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %mean_1), kwargs = {})
#   %add_44 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_61, %mul_77), kwargs = {})
#   return %add_44
triton_poi_fused_add_mul_zeros_like_37 = async_compile.triton('triton_poi_fused_add_mul_zeros_like_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_zeros_like_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_zeros_like_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp3 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp1 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ns/cnsvzxktfxgn7kluvq77ee3spg2p7sxtwx37hccu5irmysokor3r.py
# Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw2_1, mul_73, dw2_3, X_35], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   X_35 => convert_element_type_307
#   dw0_momentum => full_default_1
#   dw2_1 => add_3
#   dw2_3 => add_46
#   mul_10 => mul_13
#   mul_73 => mul_79
# Graph fragment:
#   %bmm_62 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_62]
#   %bmm_8 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %mean : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %mean_1 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_1]
#   %add_46 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_46]
#   %full_default_1 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_3 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_13), kwargs = {})
#   %mul_79 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %mean_1), kwargs = {})
#   %add_46 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_62, %mul_79), kwargs = {})
#   %convert_element_type_307 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_46, torch.bfloat16), kwargs = {})
#   return %add_46,%convert_element_type_307
triton_poi_fused__to_copy_add_mul_zeros_like_38 = async_compile.triton('triton_poi_fused__to_copy_add_mul_zeros_like_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_zeros_like_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_zeros_like_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp3 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp1 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/gw/cgwnjm4vp5sxa7oxqhxno4cc3js4iyu3p5ctq2wfs5kacht2ssy4.py
# Topologically Sorted Source Nodes: [X_21, norm_9], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_21 => convert_element_type_213
#   norm_9 => convert_element_type_214, pow_19, sum_10
# Graph fragment:
#   %add_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %convert_element_type_213 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %convert_element_type_214 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_213, torch.float32), kwargs = {})
#   %pow_19 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_214, 2), kwargs = {})
#   %sum_10 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_19, [1, 2], True), kwargs = {})
#   return %buf180
triton_red_fused__to_copy_linalg_vector_norm_39 = async_compile.triton('triton_red_fused__to_copy_linalg_vector_norm_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_linalg_vector_norm_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 320, 'r0_': 1179680}}
)
@triton.jit
def triton_red_fused__to_copy_linalg_vector_norm_39(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 40
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/nj/cnjex7h4mb5eo6auxqvpl7sur6twx4kx3dcebstmi26d664quxec.py
# Topologically Sorted Source Nodes: [X_21, add_47, X_22, transpose_25, A_15], Original ATen: [aten._to_copy, aten.add, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_15 => convert_element_type_215, convert_element_type_216
#   X_21 => convert_element_type_213
#   X_22 => div_6
#   add_47 => add_47
#   transpose_25 => permute_26
# Graph fragment:
#   %add_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %pow_20 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_20]
#   %convert_element_type_213 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %add_47 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_6 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_213, %add_47), kwargs = {})
#   %permute_26 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_6, [0, 2, 1]), kwargs = {})
#   %convert_element_type_215 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_26, torch.bfloat16), kwargs = {})
#   %convert_element_type_216 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   return %expand_90,%expand_91
triton_poi_fused__to_copy_add_div_transpose_40 = async_compile.triton('triton_poi_fused__to_copy_add_div_transpose_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_transpose_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_transpose_40(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp2 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr1 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/at/catpemoamw2gxnl56twvpf2eo7yokfw3arlg76oq23xm4qeproje.py
# Topologically Sorted Source Nodes: [X_21, add_47, X_22, mul_76, X_23, transpose_26, A_16], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   A_16 => convert_element_type_224, convert_element_type_225
#   X_21 => convert_element_type_213
#   X_22 => div_6
#   X_23 => add_49
#   add_47 => add_47
#   mul_76 => mul_82
#   transpose_26 => permute_27
# Graph fragment:
#   %add_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %pow_20 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_20]
#   %bmm_65 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %convert_element_type_213 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %add_47 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_6 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_213, %add_47), kwargs = {})
#   %mul_82 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_49 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %bmm_65), kwargs = {})
#   %permute_27 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_49, [0, 2, 1]), kwargs = {})
#   %convert_element_type_224 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_27, torch.bfloat16), kwargs = {})
#   %convert_element_type_225 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_49, torch.bfloat16), kwargs = {})
#   return %expand_96,%expand_97
triton_poi_fused__to_copy_add_div_mul_transpose_41 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4128768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp2 / tmp5)
    tmp7 = 4.0848
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/z6/cz6ff3hic27xtjnjupjemga6ycsd53ijhml2nabbzszdhcksynif.py
# Topologically Sorted Source Nodes: [X_21, add_47, X_22, mul_76, X_23, mul_79, X_24, transpose_27, A_17], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   A_17 => convert_element_type_233, convert_element_type_234
#   X_21 => convert_element_type_213
#   X_22 => div_6
#   X_23 => add_49
#   X_24 => add_51
#   add_47 => add_47
#   mul_76 => mul_82
#   mul_79 => mul_85
#   transpose_27 => permute_28
# Graph fragment:
#   %add_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %pow_20 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_20]
#   %bmm_65 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %bmm_68 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_68]
#   %convert_element_type_213 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %add_47 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_6 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_213, %add_47), kwargs = {})
#   %mul_82 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_49 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %bmm_65), kwargs = {})
#   %mul_85 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_49, 3.9505), kwargs = {})
#   %add_51 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %bmm_68), kwargs = {})
#   %permute_28 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_51, [0, 2, 1]), kwargs = {})
#   %convert_element_type_233 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_28, torch.bfloat16), kwargs = {})
#   %convert_element_type_234 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.bfloat16), kwargs = {})
#   return %expand_102,%expand_103
triton_poi_fused__to_copy_add_div_mul_transpose_42 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp2 / tmp5)
    tmp7 = 4.0848
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 3.9505
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/ze/cze4juanllwnyfhdknlmbl7epltvawvhdosmfjajilfwzx4culbt.py
# Topologically Sorted Source Nodes: [X_21, add_47, X_22, mul_76, X_23, mul_79, X_24, mul_82, X_25, transpose_28, A_18], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   A_18 => convert_element_type_242, convert_element_type_243
#   X_21 => convert_element_type_213
#   X_22 => div_6
#   X_23 => add_49
#   X_24 => add_51
#   X_25 => add_53
#   add_47 => add_47
#   mul_76 => mul_82
#   mul_79 => mul_85
#   mul_82 => mul_88
#   transpose_28 => permute_29
# Graph fragment:
#   %add_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %pow_20 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=pow_20]
#   %bmm_65 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %bmm_68 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_68]
#   %bmm_71 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_71]
#   %add_53 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_53]
#   %convert_element_type_213 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %add_47 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_6 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_213, %add_47), kwargs = {})
#   %mul_82 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_49 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %bmm_65), kwargs = {})
#   %mul_85 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_49, 3.9505), kwargs = {})
#   %add_51 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %bmm_68), kwargs = {})
#   %mul_88 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 3.7418), kwargs = {})
#   %add_53 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %bmm_71), kwargs = {})
#   %permute_29 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_53, [0, 2, 1]), kwargs = {})
#   %convert_element_type_242 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_29, torch.bfloat16), kwargs = {})
#   %convert_element_type_243 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_53, torch.bfloat16), kwargs = {})
#   return %add_53,%expand_108,%expand_109
triton_poi_fused__to_copy_add_div_mul_transpose_43 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_transpose_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_transpose_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7667712}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_transpose_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp2 / tmp5)
    tmp7 = 4.0848
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 3.9505
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = 3.7418
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp22, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/7x/c7xjxnwjjvtnpinm7gixsxsuyb6w36l3lkjwhxw5pmgucuvndvez.py
# Topologically Sorted Source Nodes: [mul_85, X_26, mul_88, X_27, w1_main_1, norm_13, add_84, truediv_10, w1_1, bmm_20, transpose_42, dhidden_2], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   X_26 => add_55
#   X_27 => add_57
#   add_84 => add_84
#   bmm_20 => convert_element_type_364
#   dhidden_2 => convert_element_type_377
#   mul_85 => mul_91
#   mul_88 => mul_94
#   norm_13 => pow_27, pow_28, sum_14
#   transpose_42 => permute_43
#   truediv_10 => div_10
#   w1_1 => mul_126
#   w1_main_1 => add_80
# Graph fragment:
#   %add_37 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_37]
#   %add_53 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_53]
#   %bmm_74 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_74]
#   %bmm_77 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_77]
#   %add_80 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_80]
#   %sum_14 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_14]
#   %pow_28 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_28]
#   %pow_4 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_4]
#   %mul_91 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 2.8769), kwargs = {})
#   %add_55 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %bmm_74), kwargs = {})
#   %mul_94 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 2.8366), kwargs = {})
#   %add_57 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %bmm_77), kwargs = {})
#   %add_80 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %add_57), kwargs = {})
#   %pow_27 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_80, 2), kwargs = {})
#   %sum_14 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_27, [2], True), kwargs = {})
#   %pow_28 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_14, 0.5), kwargs = {})
#   %add_84 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_28, 1e-05), kwargs = {})
#   %div_10 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_80, %add_84), kwargs = {})
#   %mul_126 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %pow_4), kwargs = {})
#   %convert_element_type_364 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_126, torch.bfloat16), kwargs = {})
#   %permute_43 : Tensor "f32[8, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_126, [0, 2, 1]), kwargs = {})
#   %convert_element_type_377 : Tensor "bf16[8, 192, 192][36864, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_43, torch.bfloat16), kwargs = {})
#   return %add_80,%sum_14,%pow_28,%convert_element_type_364,%convert_element_type_377
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 8257536}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 2.8769
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = 2.8366
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp0 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp12 / tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp12, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp24, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/zz/czz46wktf75452sloma6yahblsrwmwdcd57nud34c4kywsyku2me.py
# Topologically Sorted Source Nodes: [mul_100, X_33, mul_103, X_34, w0_main_1, norm_12, add_83, truediv_9, w0_1, bmm_19], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_33 => add_66
#   X_34 => add_68
#   add_83 => add_83
#   bmm_19 => convert_element_type_359
#   mul_100 => mul_106
#   mul_103 => mul_109
#   norm_12 => pow_25, pow_26, sum_13
#   truediv_9 => div_9
#   w0_1 => mul_125
#   w0_main_1 => add_81
# Graph fragment:
#   %add_38 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_38]
#   %add_64 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_64]
#   %bmm_89 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_89]
#   %bmm_92 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_92]
#   %add_81 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_81]
#   %sum_13 : Tensor "f32[8, 192, 1][192, 1, 1536]cuda:0" = PlaceHolder[target=sum_13]
#   %pow_26 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_26]
#   %pow_2 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0" = PlaceHolder[target=pow_2]
#   %mul_106 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, 2.8769), kwargs = {})
#   %add_66 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %bmm_89), kwargs = {})
#   %mul_109 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_66, 2.8366), kwargs = {})
#   %add_68 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %bmm_92), kwargs = {})
#   %add_81 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %add_68), kwargs = {})
#   %pow_25 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_81, 2), kwargs = {})
#   %sum_13 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_25, [2], True), kwargs = {})
#   %pow_26 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_13, 0.5), kwargs = {})
#   %add_83 : Tensor "f32[8, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_26, 1e-05), kwargs = {})
#   %div_9 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_81, %add_83), kwargs = {})
#   %mul_125 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %pow_2), kwargs = {})
#   %convert_element_type_359 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_125, torch.bfloat16), kwargs = {})
#   return %add_81,%sum_13,%pow_26,%convert_element_type_359
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 7077888}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 2.8769
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = 2.8366
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp0 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp12 / tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp12, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/43/c43qaovek2ppp3d7hrona2w6dzko6orduoxo577vkmuw2jlb4wgi.py
# Topologically Sorted Source Nodes: [q, qi_2, h_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
# Source node to ATen node mapping:
#   h_2 => convert_element_type_354
#   q => permute
#   qi_2 => slice_24
# Graph fragment:
#   %primals_5 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %permute : Tensor "f32[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%primals_5, [0, 2, 1]), kwargs = {})
#   %slice_24 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute, 2, 2048, 3072), kwargs = {})
#   %convert_element_type_354 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_24, torch.bfloat16), kwargs = {})
#   return %convert_element_type_354
triton_poi_fused__to_copy_slice_transpose_46 = async_compile.triton('triton_poi_fused__to_copy_slice_transpose_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_transpose_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_slice_transpose_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (393216 + x0 + 786432*x1), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/wv/cwvvybvukjvpfniffelq3qbsa4meulxopktxe24je7lkpsnfc72t.py
# Topologically Sorted Source Nodes: [ki_2, lr2i_2, lr0i_2, transpose_40, gate_before_act_2, mul_130, type_as_7, mul_131, type_as_8], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   gate_before_act_2 => convert_element_type_367
#   ki_2 => slice_22
#   lr0i_2 => slice_27
#   lr2i_2 => slice_26
#   mul_130 => mul_139
#   mul_131 => mul_140
#   transpose_40 => permute_41
#   type_as_7 => convert_element_type_385
#   type_as_8 => convert_element_type_388
# Graph fragment:
#   %primals_7 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_10 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %primals_9 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %slice_22 : Tensor "f32[8, 1024, 192][786432, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_7, 1, 2048, 3072), kwargs = {})
#   %slice_26 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_9, 1, 2048, 3072), kwargs = {})
#   %slice_27 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_10, 1, 2048, 3072), kwargs = {})
#   %permute_41 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_22, [0, 2, 1]), kwargs = {})
#   %convert_element_type_367 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_41, torch.bfloat16), kwargs = {})
#   %mul_139 : Tensor "f32[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_22, %slice_27), kwargs = {})
#   %convert_element_type_385 : Tensor "bf16[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_139, torch.bfloat16), kwargs = {})
#   %mul_140 : Tensor "f32[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_22, %slice_26), kwargs = {})
#   %convert_element_type_388 : Tensor "bf16[8, 1024, 192][196608, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_140, torch.bfloat16), kwargs = {})
#   return %convert_element_type_367,%convert_element_type_385,%convert_element_type_388
triton_poi_fused__to_copy_mul_slice_transpose_47 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_transpose_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_transpose_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_transpose_47(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x4 = xindex
    x3 = ((xindex // 192) % 1024)
    tmp0 = tl.load(in_ptr0 + (393216 + x0 + 786432*x1), None)
    tmp2 = tl.load(in_ptr1 + (6144 + 3*x3 + 12288*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (6144 + 3*x3 + 12288*x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp1, None)
    tl.store(out_ptr1 + (x4), tmp4, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/xk/cxk6qldy7vwaxvwkouiykgmd4ietqxmmwc7ceryvukprymg7mj7q.py
# Topologically Sorted Source Nodes: [lr1i_2, silu_7, hidden_2, dhidden_before_mul_2, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86, dx_2, transpose_43, mul_129, type_as_6], Original ATen: [aten.slice, aten.silu, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   add_86 => add_86
#   dgate_2 => mul_134
#   dhidden_before_mul_2 => mul_133
#   dx_2 => mul_137
#   hidden_2 => mul_131
#   lr1i_2 => slice_25
#   mul_126 => mul_135
#   mul_127 => mul_136
#   mul_129 => mul_138
#   sigma_2 => sigmoid_11
#   silu_7 => convert_element_type_375, convert_element_type_376, mul_130, sigmoid_9
#   sub_2 => sub_2
#   transpose_43 => permute_44
#   type_as_6 => convert_element_type_382
# Graph fragment:
#   %bmm_113 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_113]
#   %bmm_111 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_111]
#   %bmm_112 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %primals_8 : Tensor "f32[8, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %slice_25 : Tensor "f32[8, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_8, 1, 2048, 3072), kwargs = {})
#   %convert_element_type_375 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_111, torch.float32), kwargs = {})
#   %sigmoid_9 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_375,), kwargs = {})
#   %mul_130 : Tensor "f32[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_375, %sigmoid_9), kwargs = {})
#   %convert_element_type_376 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_130, torch.bfloat16), kwargs = {})
#   %mul_131 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_376, %bmm_112), kwargs = {})
#   %mul_133 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_113, %convert_element_type_376), kwargs = {})
#   %mul_134 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_113, %bmm_112), kwargs = {})
#   %sigmoid_11 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_111,), kwargs = {})
#   %mul_135 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %sigmoid_11), kwargs = {})
#   %sub_2 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_11), kwargs = {})
#   %mul_136 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_111, %sub_2), kwargs = {})
#   %add_86 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, 1), kwargs = {})
#   %mul_137 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_135, %add_86), kwargs = {})
#   %permute_44 : Tensor "bf16[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_131, [0, 2, 1]), kwargs = {})
#   %mul_138 : Tensor "f32[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_44, %slice_25), kwargs = {})
#   %convert_element_type_382 : Tensor "bf16[8, 1024, 192][196608, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_138, torch.bfloat16), kwargs = {})
#   return %mul_133,%mul_137,%convert_element_type_382
triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_48 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34603008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 1024)
    x3 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (6144 + 3*x1 + 12288*x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.0
    tmp12 = tmp11 - tmp9
    tmp13 = tmp1 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tmp16 = tmp5 * tmp7
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/co/ccoj5rnp765kf4xkwommw7tcjxsktigjrmdjtzfb7uz6ztb2sv4o.py
# Topologically Sorted Source Nodes: [m_i_4, m_i_5], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i_4 => slice_32
#   m_i_5 => mean_2
# Graph fragment:
#   %primals_4 : Tensor "f32[8, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %buf328 : Tensor "f32[8, 1, 1][1, 8, 8]cuda:0" = PlaceHolder[target=buf328]
#   %slice_32 : Tensor "f32[8, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_4, 1, 2048, 3072), kwargs = {})
#   %mean_2 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_32, [1], True), kwargs = {})
#   return %buf328,%mean_2
triton_per_fused_mean_slice_49 = async_compile.triton('triton_per_fused_mean_slice_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64, 'r0_': 32768}}
)
@triton.jit
def triton_per_fused_mean_slice_49(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + r0_1 + 4096*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/vc/cvczkxbqxnw3nkpz54e474nxkn4yofr5voiwb6tr3ttyowdhfrb2.py
# Topologically Sorted Source Nodes: [mul_133, dw1_5, X_42], Original ATen: [aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   X_42 => convert_element_type_391
#   dw1_5 => add_88
#   mul_133 => mul_142
# Graph fragment:
#   %bmm_114 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_114]
#   %add_45 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_45]
#   %mean_2 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=mean_2]
#   %mul_142 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %mean_2), kwargs = {})
#   %add_88 : Tensor "f32[8, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_114, %mul_142), kwargs = {})
#   %convert_element_type_391 : Tensor "bf16[8, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_88, torch.bfloat16), kwargs = {})
#   return %convert_element_type_391
triton_poi_fused__to_copy_add_mul_50 = async_compile.triton('triton_poi_fused__to_copy_add_mul_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2949120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_50(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/mt/cmtpflxd4akdivhhl73j6d6qyotgb7inql5ze6xy6bgzqehsb5qk.py
# Topologically Sorted Source Nodes: [q, qi_3, h_3], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_532
#   q => permute
#   qi_3 => slice_33
# Graph fragment:
#   %primals_5 : Tensor "f32[8, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %permute : Tensor "f32[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%primals_5, [0, 2, 1]), kwargs = {})
#   %slice_33 : Tensor "f32[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute, 2, 3072, 4096), kwargs = {})
#   %convert_element_type_532 : Tensor "bf16[8, 192, 1024][196608, 1, 192]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_33, torch.bfloat16), kwargs = {})
#   return %convert_element_type_532
triton_poi_fused__to_copy_slice_transpose_51 = async_compile.triton('triton_poi_fused__to_copy_slice_transpose_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_transpose_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_slice_transpose_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (589824 + x0 + 786432*x1), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/xa/cxasmawdkaiac4ncidlmiwyue73dqdrbwh5rcnnrmiripvz22hii.py
# Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   output => full_3, permute_2
#   setitem => copy, slice_7
#   setitem_1 => copy_1, slice_18
#   setitem_2 => copy_2, slice_29
#   setitem_3 => copy_3, slice_35
# Graph fragment:
#   %bmm_164 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_164]
#   %bmm_110 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_110]
#   %bmm_56 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_56]
#   %bmm_2 : Tensor "bf16[8, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %full_3 : Tensor "bf16[8, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 4096, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_2 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_3, [0, 2, 1]), kwargs = {})
#   %slice_7 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 1024), kwargs = {})
#   %copy : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %bmm_2), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_2, %copy, 2, 0, 1024), kwargs = {})
#   %slice_18 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 1024, 2048), kwargs = {})
#   %copy_1 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_18, %bmm_56), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 1024, 2048), kwargs = {})
#   %slice_29 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_1, 2, 2048, 3072), kwargs = {})
#   %copy_2 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_29, %bmm_110), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %copy_2, 2, 2048, 3072), kwargs = {})
#   %slice_35 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 2, 3072, 4096), kwargs = {})
#   %copy_3 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_35, %bmm_164), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %copy_3, 2, 3072, 4096), kwargs = {})
#   return %slice_scatter_default_3
triton_poi_fused_copy_slice_zeros_like_52 = async_compile.triton('triton_poi_fused_copy_slice_zeros_like_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_like_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 75497472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_like_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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
    tmp4 = tl.full([1], 2048, tl.int64)
    tmp5 = tmp0 >= tmp4
    tmp6 = tmp0 < tmp1
    tmp7 = tmp5 & tmp6
    tmp8 = tl.load(in_ptr1 + ((-2048) + x0 + 1024*x1), tmp7, other=0.0).to(tl.float32)
    tmp9 = tl.full([1], 1024, tl.int64)
    tmp10 = tmp0 >= tmp9
    tmp11 = tmp0 < tmp4
    tmp12 = tmp10 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-1024) + x0 + 1024*x1), tmp12, other=0.0).to(tl.float32)
    tmp14 = tmp0 < tmp9
    tmp15 = tl.load(in_ptr3 + (x0 + 1024*x1), tmp14, other=0.0).to(tl.float32)
    tmp16 = 0.0
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp12, tmp13, tmp17)
    tmp19 = tl.where(tmp7, tmp8, tmp18)
    tmp20 = tl.where(tmp2, tmp3, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/lp/clpyaitv4w4d2skz756u2x3vex24ysog23nz476tnv4dxhnr7okx.py
# Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   output => full_3, permute_2
#   setitem => copy, slice_7
#   setitem_1 => copy_1, slice_18
#   setitem_2 => copy_2, slice_29
#   setitem_3 => copy_3, slice_35
# Graph fragment:
#   %slice_scatter_default_3 : Tensor "bf16[8, 192, 4096][786432, 4096, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_3]
#   %full_3 : Tensor "bf16[8, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 4096, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_2 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_3, [0, 2, 1]), kwargs = {})
#   %slice_7 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 1024), kwargs = {})
#   %copy : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %bmm_2), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_2, %copy, 2, 0, 1024), kwargs = {})
#   %slice_18 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 1024, 2048), kwargs = {})
#   %copy_1 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_18, %bmm_56), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 1024, 2048), kwargs = {})
#   %slice_29 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_1, 2, 2048, 3072), kwargs = {})
#   %copy_2 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_29, %bmm_110), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %copy_2, 2, 2048, 3072), kwargs = {})
#   %slice_35 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 2, 3072, 4096), kwargs = {})
#   %copy_3 : Tensor "bf16[8, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_35, %bmm_164), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[8, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %copy_3, 2, 3072, 4096), kwargs = {})
#   %permute_461 : Tensor "bf16[8, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_scatter_default_3, [0, 2, 1]), kwargs = {})
#   return %permute_461
triton_poi_fused_copy_slice_zeros_like_53 = async_compile.triton('triton_poi_fused_copy_slice_zeros_like_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_like_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 12582912, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_like_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4096)
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4096*x2 + 786432*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp0, xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
        args.clear()
        assert_size_stride(primals_1, (8, 192, 192), (36864, 192, 1))
        assert_size_stride(primals_2, (8, 192, 192), (36864, 192, 1))
        assert_size_stride(primals_3, (8, 192, 192), (36864, 192, 1))
        assert_size_stride(primals_4, (8, 4096, 1), (4096, 1, 1))
        assert_size_stride(primals_5, (8, 4096, 192), (786432, 192, 1))
        assert_size_stride(primals_6, (8, 4096, 192), (786432, 192, 1))
        assert_size_stride(primals_7, (8, 4096, 192), (786432, 192, 1))
        assert_size_stride(primals_8, (8, 4096, 1), (12288, 3, 1))
        assert_size_stride(primals_9, (8, 4096, 1), (12288, 3, 1))
        assert_size_stride(primals_10, (8, 4096, 1), (12288, 3, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf12 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            buf21 = empty_strided_cuda((8, 1024, 192), (196608, 192, 1), torch.bfloat16)
            buf23 = empty_strided_cuda((8, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki, lr2i, lr0i, transpose_2, gate_before_act, mul_8, type_as_1, mul_9, type_as_2], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_transpose_0.run(primals_7, primals_10, primals_9, buf12, buf21, buf23, 1572864, stream=stream0)
            buf13 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki, transpose_2, gate_before_act], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(primals_1, buf12, out=buf13)
            buf14 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_before_mul], Original ATen: [aten.bmm]
            extern_kernels.bmm(primals_3, buf12, out=buf14)
            buf2 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf10 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf15 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf3 = reinterpret_tensor(buf2, (8, 192, 1), (192, 1, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, bmm_2, transpose_4, dhidden], Original ATen: [aten.linalg_vector_norm, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_transpose_1.run(buf3, primals_2, buf10, buf15, 1536, 192, stream=stream0)
            buf16 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi, transpose_4, dhidden], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf15, reinterpret_tensor(primals_6, (8, 192, 1024), (786432, 1, 192), 0), out=buf16)
            buf17 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf18 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf19 = empty_strided_cuda((8, 1024, 192), (196608, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [lr1i, silu_1, hidden, dhidden_before_mul, dgate, sigma, mul_4, sub, mul_5, add, dx, transpose_5, mul_7, type_as], Original ATen: [aten.slice, aten.silu, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2.run(buf16, buf13, buf14, primals_8, buf17, buf18, buf19, 1572864, stream=stream0)
            buf22 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw0], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf18, buf21, out=buf22)
            buf24 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw2], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf17, buf23, out=buf24)
            buf25 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf26 = reinterpret_tensor(buf25, (8, 1, 1), (1, 1, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_3.run(buf26, primals_4, 8, 1024, stream=stream0)
            buf20 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi, dw1], Original ATen: [aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_6, (8, 192, 1024), (786432, 1, 192), 0), buf19, out=buf20)
            buf27 = empty_strided_cuda((8, 1, 1, 5), (5, 40, 40, 1), torch.float32)
            buf66 = empty_strided_cuda((8, 1, 1, 5), (5, 40, 40, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, dw0_momentum, mul_10, dw0_1, mul_11, dw1_1, X, norm_3, X_7, norm_4], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_4.run(buf20, buf26, buf22, buf27, buf66, 40, 7373, stream=stream0)
            buf67 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf68 = reinterpret_tensor(buf67, (8, 1, 1), (1, 1, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, X_7, norm_4], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf68, buf66, 8, 5, stream=stream0)
            del buf66
            buf28 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf29 = reinterpret_tensor(buf28, (8, 1, 1), (1, 1, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf29, buf27, 8, 5, stream=stream0)
            buf30 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf31 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, X, add_4, X_1, transpose_6, A], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_6.run(buf20, buf26, buf29, buf30, buf31, 294912, stream=stream0)
            buf32 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf30, buf31, out=buf32)
            buf33 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_14], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf32, buf33, 294912, stream=stream0)
            buf34 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_14, matmul_1], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf33, buf32, out=buf34)
            buf35 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [mul_13, B], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf35, buf32, 294912, stream=stream0)
            buf36 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf35, buf30, out=buf36)
            buf37 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf38 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, X, add_4, X_1, mul_15, X_2, transpose_7, A_1], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_9.run(buf20, buf26, buf29, buf36, buf37, buf38, 294912, stream=stream0)
            buf39 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_1], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf37, buf38, out=buf39)
            buf40 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_17], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf39, buf40, 294912, stream=stream0)
            buf41 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_17, matmul_4], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf40, buf39, out=buf41)
            buf42 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [mul_16, B_1], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf42, buf39, 294912, stream=stream0)
            buf43 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf42, buf37, out=buf43)
            buf44 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf45 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf46 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf69 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf70 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, dw0_momentum, mul_10, dw0_1, mul_11, dw1_1, X, add_4, X_1, mul_15, X_2, mul_18, X_3, transpose_8, A_2, X_7, add_15, X_8, transpose_11, A_5], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_12.run(buf20, buf26, buf29, buf36, buf43, buf22, buf68, buf44, buf45, buf46, buf69, buf70, 294912, stream=stream0)
            buf71 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [A_5], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf69, buf70, out=buf71)
            buf72 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [mul_29], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf71, buf72, 294912, stream=stream0)
            buf73 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_29, matmul_16], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf72, buf71, out=buf73)
            buf74 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [mul_28, B_5], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf74, buf71, 294912, stream=stream0)
            buf75 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf74, buf69, out=buf75)
            buf76 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf77 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, X_7, add_15, X_8, mul_30, X_9, transpose_12, A_6], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_9.run(buf22, buf26, buf68, buf75, buf76, buf77, 294912, stream=stream0)
            buf78 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_6], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf76, buf77, out=buf78)
            buf79 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_32], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf78, buf79, 294912, stream=stream0)
            buf80 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_32, matmul_19], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf79, buf78, out=buf80)
            buf81 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [mul_31, B_6], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf81, buf78, 294912, stream=stream0)
            buf82 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf81, buf76, out=buf82)
            buf83 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf84 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf85 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf105 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, dw2_1, X_7, add_15, X_8, mul_30, X_9, mul_33, X_10, transpose_13, A_7, X_14], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_zeros_like_13.run(buf22, buf26, buf68, buf75, buf82, buf24, buf83, buf84, buf85, buf105, 294912, stream=stream0)
            buf86 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [transpose_13, A_7], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf84, buf85, out=buf86)
            buf87 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [mul_35], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf86, buf87, 294912, stream=stream0)
            buf88 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_35, matmul_22], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf87, buf86, out=buf88)
            buf89 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [mul_34, B_7], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf89, buf86, 294912, stream=stream0)
            buf90 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf89, buf84, out=buf90)
            buf91 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf92 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_36, X_11, transpose_14, A_8], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_16.run(buf83, buf90, buf91, buf92, 294912, stream=stream0)
            buf93 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_8], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf91, buf92, out=buf93)
            buf94 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_38], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf93, buf94, 294912, stream=stream0)
            buf95 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_38, matmul_25], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf94, buf93, out=buf95)
            buf96 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [mul_37, B_8], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf96, buf93, 294912, stream=stream0)
            buf97 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf96, buf91, out=buf97)
            buf98 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf99 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, transpose_15, A_9], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_19.run(buf83, buf90, buf97, buf98, buf99, 294912, stream=stream0)
            buf100 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_9], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf98, buf99, out=buf100)
            buf101 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_41], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf100, buf101, 294912, stream=stream0)
            buf102 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_41, matmul_28], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf101, buf100, out=buf102)
            buf103 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [mul_40, B_9], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf103, buf100, 294912, stream=stream0)
            buf104 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf103, buf98, out=buf104)
            buf0 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf1 = reinterpret_tensor(buf0, (8, 192, 1), (192, 1, 1), 0); del buf0  # reuse
            buf146 = buf83; del buf83  # reuse
            buf148 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf149 = reinterpret_tensor(buf148, (8, 192, 1), (192, 1, 1), 0); del buf148  # reuse
            buf157 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [w0_norm, mul_36, X_11, mul_39, X_12, mul_42, X_13, w0_main, norm_6, add_40, truediv_3, w0, bmm_10], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_22.run(buf1, buf146, buf149, primals_1, buf90, buf97, buf104, buf157, 1536, 192, stream=stream0)
            buf106 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [norm_5], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_23.run(buf105, buf106, 40, 7373, stream=stream0)
            buf107 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf108 = reinterpret_tensor(buf107, (8, 1, 1), (1, 1, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [norm_5], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf108, buf106, 8, 5, stream=stream0)
            buf109 = buf97; del buf97  # reuse
            buf110 = reinterpret_tensor(buf90, (8, 192, 192), (36864, 1, 192), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [add_26, X_15, transpose_16, A_10], Original ATen: [aten.add, aten.div, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_24.run(buf105, buf108, buf109, buf110, 294912, stream=stream0)
            buf111 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [A_10], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf109, buf110, out=buf111)
            buf112 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_44], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf111, buf112, 294912, stream=stream0)
            buf113 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_44, matmul_31], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf112, buf111, out=buf113)
            buf114 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [mul_43, B_10], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf114, buf111, 294912, stream=stream0)
            buf115 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf114, buf109, out=buf115)
            buf116 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf117 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_26, X_15, mul_45, X_16, transpose_17, A_11], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_25.run(buf105, buf108, buf115, buf116, buf117, 294912, stream=stream0)
            buf118 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_11], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf116, buf117, out=buf118)
            buf119 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_47], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf118, buf119, 294912, stream=stream0)
            buf120 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_47, matmul_34], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf119, buf118, out=buf120)
            buf121 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [mul_46, B_11], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf121, buf118, 294912, stream=stream0)
            buf122 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf121, buf116, out=buf122)
            buf123 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf124 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_26, X_15, mul_45, X_16, mul_48, X_17, transpose_18, A_12], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_26.run(buf105, buf108, buf115, buf122, buf123, buf124, 294912, stream=stream0)
            buf125 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_12], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf123, buf124, out=buf125)
            buf126 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_50], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf125, buf126, 294912, stream=stream0)
            buf127 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_50, matmul_37], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf126, buf125, out=buf127)
            buf128 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [mul_49, B_12], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf128, buf125, 294912, stream=stream0)
            buf129 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf128, buf123, out=buf129)
            buf130 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf131 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf132 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_26, X_15, mul_45, X_16, mul_48, X_17, mul_51, X_18, transpose_19, A_13], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_27.run(buf105, buf108, buf115, buf122, buf129, buf130, buf131, buf132, 294912, stream=stream0)
            buf133 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, A_13], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf131, buf132, out=buf133)
            buf134 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [mul_53], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf133, buf134, 294912, stream=stream0)
            buf135 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [mul_53, matmul_40], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf134, buf133, out=buf135)
            buf136 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [mul_52, B_13], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf136, buf133, 294912, stream=stream0)
            buf137 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf136, buf131, out=buf137)
            buf138 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf139 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_54, X_19, transpose_20, A_14], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf130, buf137, buf138, buf139, 294912, stream=stream0)
            buf140 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_14], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf138, buf139, out=buf140)
            buf141 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_56], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf140, buf141, 294912, stream=stream0)
            buf142 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_56, matmul_43], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf141, buf140, out=buf142)
            buf143 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [mul_55, B_14], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf143, buf140, 294912, stream=stream0)
            buf144 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf143, buf138, out=buf144)
            buf4 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf5 = reinterpret_tensor(buf4, (8, 192, 1), (192, 1, 1), 0); del buf4  # reuse
            buf147 = buf130; del buf130  # reuse
            buf152 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf153 = reinterpret_tensor(buf152, (8, 192, 1), (192, 1, 1), 0); del buf152  # reuse
            buf154 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [w2_norm, mul_54, X_19, mul_57, X_20, w2_main, norm_8, add_42, truediv_5, w2, h_1], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_29.run(buf5, buf147, buf153, primals_3, buf137, buf144, buf154, 1536, 192, stream=stream0)
            buf6 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi, h], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_slice_transpose_30.run(primals_5, buf6, 1572864, stream=stream0)
            buf7 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi, h], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(primals_3, buf6, out=buf7)
            buf8 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
            extern_kernels.bmm(primals_1, buf6, out=buf8)
            buf9 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate, mul], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_31.run(buf8, buf7, buf9, 1572864, stream=stream0)
            buf11 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_2], Original ATen: [aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf10, buf9, out=buf11)
            buf47 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [transpose_8, A_2], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf45, buf46, out=buf47)
            buf48 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [mul_20], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf47, buf48, 294912, stream=stream0)
            buf49 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_20, matmul_7], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf48, buf47, out=buf49)
            buf50 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [mul_19, B_2], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf50, buf47, 294912, stream=stream0)
            buf51 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf50, buf45, out=buf51)
            buf52 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf53 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_21, X_4, transpose_9, A_3], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_16.run(buf44, buf51, buf52, buf53, 294912, stream=stream0)
            buf54 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_3], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf52, buf53, out=buf54)
            buf55 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_23], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf54, buf55, 294912, stream=stream0)
            buf56 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_23, matmul_10], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf55, buf54, out=buf56)
            buf57 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [mul_22, B_3], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf57, buf54, 294912, stream=stream0)
            buf58 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf57, buf52, out=buf58)
            buf59 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf60 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, transpose_10, A_4], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_19.run(buf44, buf51, buf58, buf59, buf60, 294912, stream=stream0)
            buf61 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_4], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf59, buf60, out=buf61)
            buf62 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_26], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf61, buf62, 294912, stream=stream0)
            buf63 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_26, matmul_13], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf62, buf61, out=buf63)
            buf64 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [mul_25, B_4], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf64, buf61, 294912, stream=stream0)
            buf65 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf64, buf59, out=buf65)
            buf145 = buf44; del buf44  # reuse
            buf150 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf151 = reinterpret_tensor(buf150, (8, 192, 1), (192, 1, 1), 0); del buf150  # reuse
            buf160 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf165 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, mul_27, X_6, w1_main, norm_7, add_41, truediv_4, w1, bmm_11, transpose_23, dhidden_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_32.run(buf145, buf151, primals_2, buf51, buf58, buf65, buf3, buf160, buf165, 1536, 192, stream=stream0)
            buf155 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi_1, h_1], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_slice_transpose_33.run(primals_5, buf155, 1572864, stream=stream0)
            buf156 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi_1, h_1], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf154, buf155, out=buf156)
            buf158 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_10], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf157, buf155, out=buf158)
            buf159 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_1, mul_61], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_31.run(buf158, buf156, buf159, 1572864, stream=stream0)
            buf161 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_11], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf160, buf159, out=buf161)
            buf162 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            buf171 = empty_strided_cuda((8, 1024, 192), (196608, 192, 1), torch.bfloat16)
            buf173 = empty_strided_cuda((8, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki_1, lr2i_1, lr0i_1, transpose_21, gate_before_act_1, mul_69, type_as_4, mul_70, type_as_5], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_transpose_34.run(primals_7, primals_10, primals_9, buf162, buf171, buf173, 1572864, stream=stream0)
            buf163 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki_1, transpose_21, gate_before_act_1], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf157, buf162, out=buf163)
            buf164 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_before_mul_1], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf154, buf162, out=buf164)
            buf166 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi_1, dhidden_1], Original ATen: [aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(buf165, reinterpret_tensor(primals_6, (8, 192, 1024), (786432, 1, 192), 196608), out=buf166)
            buf167 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf168 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf169 = empty_strided_cuda((8, 1024, 192), (196608, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [lr1i_1, silu_4, hidden_1, dhidden_before_mul_1, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43, dx_1, transpose_24, mul_68, type_as_3], Original ATen: [aten.slice, aten.silu, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_35.run(buf166, buf163, buf164, primals_8, buf167, buf168, buf169, 1572864, stream=stream0)
            buf170 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [v, vi_1, dw1_2], Original ATen: [aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_6, (8, 192, 1024), (786432, 1, 192), 196608), buf169, out=buf170)
            buf172 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [dw0_2], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf168, buf171, out=buf172)
            buf174 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [dw2_2], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf167, buf173, out=buf174)
            buf175 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf176 = reinterpret_tensor(buf175, (8, 1, 1), (1, 1, 1), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [m_i_2, m_i_3], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_36.run(buf176, primals_4, 8, 1024, stream=stream0)
            buf177 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw0_1, mul_71, dw0_3], Original ATen: [aten.zeros_like, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_zeros_like_37.run(buf172, buf22, buf26, buf176, buf177, 294912, stream=stream0)
            buf178 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, mul_11, dw1_1, mul_72, dw1_3], Original ATen: [aten.zeros_like, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_zeros_like_37.run(buf170, buf20, buf26, buf176, buf178, 294912, stream=stream0)
            buf179 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf258 = buf170; del buf170  # reuse
            # Topologically Sorted Source Nodes: [dw0_momentum, mul_10, dw2_1, mul_73, dw2_3, X_35], Original ATen: [aten.zeros_like, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_zeros_like_38.run(buf174, buf24, buf26, buf176, buf179, buf258, 294912, stream=stream0)
            buf180 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [X_21, norm_9], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_linalg_vector_norm_39.run(buf178, buf180, 40, 7373, stream=stream0)
            buf181 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf182 = reinterpret_tensor(buf181, (8, 1, 1), (1, 1, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [X_21, norm_9], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf182, buf180, 8, 5, stream=stream0)
            buf183 = buf174; del buf174  # reuse
            buf184 = reinterpret_tensor(buf172, (8, 192, 192), (36864, 1, 192), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [X_21, add_47, X_22, transpose_25, A_15], Original ATen: [aten._to_copy, aten.add, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_40.run(buf178, buf182, buf183, buf184, 294912, stream=stream0)
            buf185 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_15], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf183, buf184, out=buf185)
            buf186 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_75], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf185, buf186, 294912, stream=stream0)
            buf187 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_75, matmul_46], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf186, buf185, out=buf187)
            buf188 = buf187; del buf187  # reuse
            # Topologically Sorted Source Nodes: [mul_74, B_15], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf188, buf185, 294912, stream=stream0)
            buf189 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf188, buf183, out=buf189)
            buf190 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf191 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_21, add_47, X_22, mul_76, X_23, transpose_26, A_16], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_41.run(buf178, buf182, buf189, buf190, buf191, 294912, stream=stream0)
            buf192 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_16], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf190, buf191, out=buf192)
            buf193 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_78], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf192, buf193, 294912, stream=stream0)
            buf194 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_78, matmul_49], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf193, buf192, out=buf194)
            buf195 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [mul_77, B_16], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf195, buf192, 294912, stream=stream0)
            buf196 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf195, buf190, out=buf196)
            buf197 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf198 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_21, add_47, X_22, mul_76, X_23, mul_79, X_24, transpose_27, A_17], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_42.run(buf178, buf182, buf189, buf196, buf197, buf198, 294912, stream=stream0)
            buf199 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_17], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf197, buf198, out=buf199)
            buf200 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_81], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf199, buf200, 294912, stream=stream0)
            buf201 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_81, matmul_52], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf200, buf199, out=buf201)
            buf202 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [mul_80, B_17], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf202, buf199, 294912, stream=stream0)
            buf203 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_53], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf202, buf197, out=buf203)
            buf204 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf205 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf206 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_21, add_47, X_22, mul_76, X_23, mul_79, X_24, mul_82, X_25, transpose_28, A_18], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_43.run(buf178, buf182, buf189, buf196, buf203, buf204, buf205, buf206, 294912, stream=stream0)
            buf207 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [transpose_28, A_18], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf205, buf206, out=buf207)
            buf208 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [mul_84], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf207, buf208, 294912, stream=stream0)
            buf209 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [mul_84, matmul_55], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf208, buf207, out=buf209)
            buf210 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [mul_83, B_18], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf210, buf207, 294912, stream=stream0)
            buf211 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_56], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf210, buf205, out=buf211)
            buf212 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf213 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_85, X_26, transpose_29, A_19], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf204, buf211, buf212, buf213, 294912, stream=stream0)
            buf214 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_19], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf212, buf213, out=buf214)
            buf215 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_87], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf214, buf215, 294912, stream=stream0)
            buf216 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_87, matmul_58], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf215, buf214, out=buf216)
            buf217 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [mul_86, B_19], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf217, buf214, 294912, stream=stream0)
            buf218 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_59], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf217, buf212, out=buf218)
            buf219 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [X_28, norm_10], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_linalg_vector_norm_39.run(buf177, buf219, 40, 7373, stream=stream0)
            buf220 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf221 = reinterpret_tensor(buf220, (8, 1, 1), (1, 1, 1), 0); del buf220  # reuse
            # Topologically Sorted Source Nodes: [X_28, norm_10], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf221, buf219, 8, 5, stream=stream0)
            buf222 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf223 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_28, add_58, X_29, transpose_30, A_20], Original ATen: [aten._to_copy, aten.add, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_40.run(buf177, buf221, buf222, buf223, 294912, stream=stream0)
            buf224 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_20], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf222, buf223, out=buf224)
            buf225 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_90], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf224, buf225, 294912, stream=stream0)
            buf226 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_90, matmul_61], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf225, buf224, out=buf226)
            buf227 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [mul_89, B_20], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf227, buf224, 294912, stream=stream0)
            buf228 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_62], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf227, buf222, out=buf228)
            buf229 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf230 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_28, add_58, X_29, mul_91, X_30, transpose_31, A_21], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_41.run(buf177, buf221, buf228, buf229, buf230, 294912, stream=stream0)
            buf231 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_21], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf229, buf230, out=buf231)
            buf232 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_93], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf231, buf232, 294912, stream=stream0)
            buf233 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_93, matmul_64], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf232, buf231, out=buf233)
            buf234 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [mul_92, B_21], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf234, buf231, 294912, stream=stream0)
            buf235 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_65], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf234, buf229, out=buf235)
            buf236 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf237 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_28, add_58, X_29, mul_91, X_30, mul_94, X_31, transpose_32, A_22], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_42.run(buf177, buf221, buf228, buf235, buf236, buf237, 294912, stream=stream0)
            buf238 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_22], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf236, buf237, out=buf238)
            buf239 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_96], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf238, buf239, 294912, stream=stream0)
            buf240 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_96, matmul_67], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf239, buf238, out=buf240)
            buf241 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [mul_95, B_22], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf241, buf238, 294912, stream=stream0)
            buf242 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_68], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf241, buf236, out=buf242)
            buf243 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf244 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf245 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [X_28, add_58, X_29, mul_91, X_30, mul_94, X_31, mul_97, X_32, transpose_33, A_23], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_43.run(buf177, buf221, buf228, buf235, buf242, buf243, buf244, buf245, 294912, stream=stream0)
            buf246 = buf242; del buf242  # reuse
            # Topologically Sorted Source Nodes: [transpose_33, A_23], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf244, buf245, out=buf246)
            buf247 = buf235; del buf235  # reuse
            # Topologically Sorted Source Nodes: [mul_99], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf246, buf247, 294912, stream=stream0)
            buf248 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [mul_99, matmul_70], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf247, buf246, out=buf248)
            buf249 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [mul_98, B_23], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf249, buf246, 294912, stream=stream0)
            buf250 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_71], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf249, buf244, out=buf250)
            buf251 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf252 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_100, X_33, transpose_34, A_24], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf243, buf250, buf251, buf252, 294912, stream=stream0)
            buf253 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_24], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf251, buf252, out=buf253)
            buf254 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_102], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf253, buf254, 294912, stream=stream0)
            buf255 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_102, matmul_73], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf254, buf253, out=buf255)
            buf256 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [mul_101, B_24], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf256, buf253, 294912, stream=stream0)
            buf257 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_74], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf256, buf251, out=buf257)
            buf259 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [norm_11], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_23.run(buf258, buf259, 40, 7373, stream=stream0)
            buf260 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf261 = reinterpret_tensor(buf260, (8, 1, 1), (1, 1, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [norm_11], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf261, buf259, 8, 5, stream=stream0)
            buf262 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf263 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_69, X_36, transpose_35, A_25], Original ATen: [aten.add, aten.div, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_24.run(buf258, buf261, buf262, buf263, 294912, stream=stream0)
            buf264 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_25], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf262, buf263, out=buf264)
            buf265 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_105], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf264, buf265, 294912, stream=stream0)
            buf266 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_105, matmul_76], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf265, buf264, out=buf266)
            buf267 = buf266; del buf266  # reuse
            # Topologically Sorted Source Nodes: [mul_104, B_25], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf267, buf264, 294912, stream=stream0)
            buf268 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_77], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf267, buf262, out=buf268)
            buf269 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf270 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_69, X_36, mul_106, X_37, transpose_36, A_26], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_25.run(buf258, buf261, buf268, buf269, buf270, 294912, stream=stream0)
            buf271 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_26], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf269, buf270, out=buf271)
            buf272 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_108], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf271, buf272, 294912, stream=stream0)
            buf273 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_108, matmul_79], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf272, buf271, out=buf273)
            buf274 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [mul_107, B_26], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf274, buf271, 294912, stream=stream0)
            buf275 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_80], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf274, buf269, out=buf275)
            buf276 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf277 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_69, X_36, mul_106, X_37, mul_109, X_38, transpose_37, A_27], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_26.run(buf258, buf261, buf268, buf275, buf276, buf277, 294912, stream=stream0)
            buf278 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_27], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf276, buf277, out=buf278)
            buf279 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_111], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf278, buf279, 294912, stream=stream0)
            buf280 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_111, matmul_82], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf279, buf278, out=buf280)
            buf281 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [mul_110, B_27], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf281, buf278, 294912, stream=stream0)
            buf282 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_83], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf281, buf276, out=buf282)
            buf283 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf284 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf285 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_69, X_36, mul_106, X_37, mul_109, X_38, mul_112, X_39, transpose_38, A_28], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_27.run(buf258, buf261, buf268, buf275, buf282, buf283, buf284, buf285, 294912, stream=stream0)
            buf286 = buf282; del buf282  # reuse
            # Topologically Sorted Source Nodes: [transpose_38, A_28], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf284, buf285, out=buf286)
            buf287 = buf275; del buf275  # reuse
            # Topologically Sorted Source Nodes: [mul_114], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf286, buf287, 294912, stream=stream0)
            buf288 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [mul_114, matmul_85], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf287, buf286, out=buf288)
            buf289 = buf288; del buf288  # reuse
            # Topologically Sorted Source Nodes: [mul_113, B_28], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf289, buf286, 294912, stream=stream0)
            buf290 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_86], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf289, buf284, out=buf290)
            buf291 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf292 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_115, X_40, transpose_39, A_29], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf283, buf290, buf291, buf292, 294912, stream=stream0)
            buf293 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_29], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf291, buf292, out=buf293)
            buf294 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_117], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf293, buf294, 294912, stream=stream0)
            buf295 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_117, matmul_88], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf294, buf293, out=buf295)
            buf296 = buf295; del buf295  # reuse
            # Topologically Sorted Source Nodes: [mul_116, B_29], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf296, buf293, 294912, stream=stream0)
            buf297 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_89], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf296, buf291, out=buf297)
            buf298 = buf204; del buf204  # reuse
            buf303 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf304 = reinterpret_tensor(buf303, (8, 192, 1), (192, 1, 1), 0); del buf303  # reuse
            buf313 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf318 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_85, X_26, mul_88, X_27, w1_main_1, norm_13, add_84, truediv_10, w1_1, bmm_20, transpose_42, dhidden_2], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_44.run(buf298, buf304, buf145, buf211, buf218, buf3, buf313, buf318, 1536, 192, stream=stream0)
            buf299 = buf243; del buf243  # reuse
            buf301 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf302 = reinterpret_tensor(buf301, (8, 192, 1), (192, 1, 1), 0); del buf301  # reuse
            buf310 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_33, mul_103, X_34, w0_main_1, norm_12, add_83, truediv_9, w0_1, bmm_19], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45.run(buf299, buf302, buf146, buf250, buf257, buf1, buf310, 1536, 192, stream=stream0)
            buf300 = buf283; del buf283  # reuse
            buf305 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf306 = reinterpret_tensor(buf305, (8, 192, 1), (192, 1, 1), 0); del buf305  # reuse
            buf307 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_40, mul_118, X_41, w2_main_1, norm_14, add_85, truediv_11, w2_1, h_2], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45.run(buf300, buf306, buf147, buf290, buf297, buf5, buf307, 1536, 192, stream=stream0)
            buf308 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi_2, h_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_slice_transpose_46.run(primals_5, buf308, 1572864, stream=stream0)
            buf309 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi_2, h_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf307, buf308, out=buf309)
            buf311 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_19], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf310, buf308, out=buf311)
            buf312 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_2, mul_122], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_31.run(buf311, buf309, buf312, 1572864, stream=stream0)
            buf314 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_20], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf313, buf312, out=buf314)
            buf315 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            buf324 = empty_strided_cuda((8, 1024, 192), (196608, 192, 1), torch.bfloat16)
            buf326 = empty_strided_cuda((8, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki_2, lr2i_2, lr0i_2, transpose_40, gate_before_act_2, mul_130, type_as_7, mul_131, type_as_8], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_transpose_47.run(primals_7, primals_10, primals_9, buf315, buf324, buf326, 1572864, stream=stream0)
            buf316 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki_2, transpose_40, gate_before_act_2], Original ATen: [aten.slice, aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf310, buf315, out=buf316)
            buf317 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_before_mul_2], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf307, buf315, out=buf317)
            buf319 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi_2, dhidden_2], Original ATen: [aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(buf318, reinterpret_tensor(primals_6, (8, 192, 1024), (786432, 1, 192), 393216), out=buf319)
            buf320 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf321 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            buf322 = empty_strided_cuda((8, 1024, 192), (196608, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [lr1i_2, silu_7, hidden_2, dhidden_before_mul_2, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86, dx_2, transpose_43, mul_129, type_as_6], Original ATen: [aten.slice, aten.silu, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_48.run(buf319, buf316, buf317, primals_8, buf320, buf321, buf322, 1572864, stream=stream0)
            buf323 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [v, vi_2, dw1_4], Original ATen: [aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(primals_6, (8, 192, 1024), (786432, 1, 192), 393216), buf322, out=buf323)
            buf325 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [dw0_4], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf321, buf324, out=buf325)
            buf327 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [dw2_4], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf320, buf326, out=buf327)
            buf328 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf329 = reinterpret_tensor(buf328, (8, 1, 1), (1, 1, 1), 0); del buf328  # reuse
            # Topologically Sorted Source Nodes: [m_i_4, m_i_5], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_49.run(buf329, primals_4, 8, 1024, stream=stream0)
            del primals_4
            buf330 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [mul_133, dw1_5, X_42], Original ATen: [aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_50.run(buf330, buf178, buf329, 294912, stream=stream0)
            buf331 = buf259; del buf259  # reuse
            # Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_23.run(buf330, buf331, 40, 7373, stream=stream0)
            buf332 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf333 = reinterpret_tensor(buf332, (8, 1, 1), (1, 1, 1), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf333, buf331, 8, 5, stream=stream0)
            buf334 = buf211; del buf211  # reuse
            buf335 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_90, X_43, transpose_44, A_30], Original ATen: [aten.add, aten.div, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_24.run(buf330, buf333, buf334, buf335, 294912, stream=stream0)
            buf336 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_30], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf334, buf335, out=buf336)
            buf337 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_136], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf336, buf337, 294912, stream=stream0)
            buf338 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_136, matmul_91], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf337, buf336, out=buf338)
            buf339 = buf338; del buf338  # reuse
            # Topologically Sorted Source Nodes: [mul_135, B_30], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf339, buf336, 294912, stream=stream0)
            buf340 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_92], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf339, buf334, out=buf340)
            buf341 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf342 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_90, X_43, mul_137, X_44, transpose_45, A_31], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_25.run(buf330, buf333, buf340, buf341, buf342, 294912, stream=stream0)
            buf343 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_31], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf341, buf342, out=buf343)
            buf344 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_139], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf343, buf344, 294912, stream=stream0)
            buf345 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_139, matmul_94], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf344, buf343, out=buf345)
            buf346 = buf345; del buf345  # reuse
            # Topologically Sorted Source Nodes: [mul_138, B_31], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf346, buf343, 294912, stream=stream0)
            buf347 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_95], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf346, buf341, out=buf347)
            buf348 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf349 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_90, X_43, mul_137, X_44, mul_140, X_45, transpose_46, A_32], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_26.run(buf330, buf333, buf340, buf347, buf348, buf349, 294912, stream=stream0)
            buf350 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_32], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf348, buf349, out=buf350)
            buf351 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_142], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf350, buf351, 294912, stream=stream0)
            buf352 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_142, matmul_97], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf351, buf350, out=buf352)
            buf353 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [mul_141, B_32], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf353, buf350, 294912, stream=stream0)
            buf354 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_98], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf353, buf348, out=buf354)
            buf355 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf356 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf357 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_90, X_43, mul_137, X_44, mul_140, X_45, mul_143, X_46, transpose_47, A_33], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_27.run(buf330, buf333, buf340, buf347, buf354, buf355, buf356, buf357, 294912, stream=stream0)
            buf358 = buf354; del buf354  # reuse
            # Topologically Sorted Source Nodes: [transpose_47, A_33], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf356, buf357, out=buf358)
            buf359 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [mul_145], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf358, buf359, 294912, stream=stream0)
            buf360 = buf340; del buf340  # reuse
            # Topologically Sorted Source Nodes: [mul_145, matmul_100], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf359, buf358, out=buf360)
            buf361 = buf360; del buf360  # reuse
            # Topologically Sorted Source Nodes: [mul_144, B_33], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf361, buf358, 294912, stream=stream0)
            buf362 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_101], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf361, buf356, out=buf362)
            buf363 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf364 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_146, X_47, transpose_48, A_34], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf355, buf362, buf363, buf364, 294912, stream=stream0)
            buf365 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_34], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf363, buf364, out=buf365)
            buf366 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_148], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf365, buf366, 294912, stream=stream0)
            buf367 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_148, matmul_103], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf366, buf365, out=buf367)
            buf368 = buf367; del buf367  # reuse
            # Topologically Sorted Source Nodes: [mul_147, B_34], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf368, buf365, 294912, stream=stream0)
            buf369 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_104], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf368, buf363, out=buf369)
            buf370 = buf325; del buf325  # reuse
            # Topologically Sorted Source Nodes: [mul_132, dw0_5, X_49], Original ATen: [aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_50.run(buf370, buf177, buf329, 294912, stream=stream0)
            buf371 = buf331; del buf331  # reuse
            # Topologically Sorted Source Nodes: [norm_16], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_23.run(buf370, buf371, 40, 7373, stream=stream0)
            buf372 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf373 = reinterpret_tensor(buf372, (8, 1, 1), (1, 1, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [norm_16], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf373, buf371, 8, 5, stream=stream0)
            buf374 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf375 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_101, X_50, transpose_49, A_35], Original ATen: [aten.add, aten.div, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_24.run(buf370, buf373, buf374, buf375, 294912, stream=stream0)
            buf376 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_35], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf374, buf375, out=buf376)
            buf377 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_151], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf376, buf377, 294912, stream=stream0)
            buf378 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_151, matmul_106], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf377, buf376, out=buf378)
            buf379 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [mul_150, B_35], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf379, buf376, 294912, stream=stream0)
            buf380 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_107], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf379, buf374, out=buf380)
            buf381 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf382 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_101, X_50, mul_152, X_51, transpose_50, A_36], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_25.run(buf370, buf373, buf380, buf381, buf382, 294912, stream=stream0)
            buf383 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_36], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf381, buf382, out=buf383)
            buf384 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_154], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf383, buf384, 294912, stream=stream0)
            buf385 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_154, matmul_109], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf384, buf383, out=buf385)
            buf386 = buf385; del buf385  # reuse
            # Topologically Sorted Source Nodes: [mul_153, B_36], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf386, buf383, 294912, stream=stream0)
            buf387 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_110], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf386, buf381, out=buf387)
            buf388 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf389 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_101, X_50, mul_152, X_51, mul_155, X_52, transpose_51, A_37], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_26.run(buf370, buf373, buf380, buf387, buf388, buf389, 294912, stream=stream0)
            buf390 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_37], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf388, buf389, out=buf390)
            buf391 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_157], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf390, buf391, 294912, stream=stream0)
            buf392 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_157, matmul_112], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf391, buf390, out=buf392)
            buf393 = buf392; del buf392  # reuse
            # Topologically Sorted Source Nodes: [mul_156, B_37], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf393, buf390, 294912, stream=stream0)
            buf394 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_113], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf393, buf388, out=buf394)
            buf395 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf396 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf397 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_101, X_50, mul_152, X_51, mul_155, X_52, mul_158, X_53, transpose_52, A_38], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_27.run(buf370, buf373, buf380, buf387, buf394, buf395, buf396, buf397, 294912, stream=stream0)
            buf398 = buf394; del buf394  # reuse
            # Topologically Sorted Source Nodes: [transpose_52, A_38], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf396, buf397, out=buf398)
            buf399 = buf387; del buf387  # reuse
            # Topologically Sorted Source Nodes: [mul_160], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf398, buf399, 294912, stream=stream0)
            buf400 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [mul_160, matmul_115], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf399, buf398, out=buf400)
            buf401 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [mul_159, B_38], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf401, buf398, 294912, stream=stream0)
            buf402 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_116], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf401, buf396, out=buf402)
            buf403 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf404 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_161, X_54, transpose_53, A_39], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf395, buf402, buf403, buf404, 294912, stream=stream0)
            buf405 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_39], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf403, buf404, out=buf405)
            buf406 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_163], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf405, buf406, 294912, stream=stream0)
            buf407 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_163, matmul_118], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf406, buf405, out=buf407)
            buf408 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [mul_162, B_39], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf408, buf405, 294912, stream=stream0)
            buf409 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_119], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf408, buf403, out=buf409)
            buf410 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [mul_134, dw2_5, X_56], Original ATen: [aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_50.run(buf410, buf179, buf329, 294912, stream=stream0)
            buf411 = buf371; del buf371  # reuse
            # Topologically Sorted Source Nodes: [norm_17], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_23.run(buf410, buf411, 40, 7373, stream=stream0)
            buf412 = empty_strided_cuda((8, 1, 1), (1, 8, 8), torch.float32)
            buf413 = reinterpret_tensor(buf412, (8, 1, 1), (1, 1, 1), 0); del buf412  # reuse
            # Topologically Sorted Source Nodes: [norm_17], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_zeros_like_5.run(buf413, buf411, 8, 5, stream=stream0)
            del buf411
            buf414 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf415 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_112, X_57, transpose_54, A_40], Original ATen: [aten.add, aten.div, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_transpose_24.run(buf410, buf413, buf414, buf415, 294912, stream=stream0)
            buf416 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_40], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf414, buf415, out=buf416)
            buf417 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_166], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf416, buf417, 294912, stream=stream0)
            buf418 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_166, matmul_121], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf417, buf416, out=buf418)
            buf419 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [mul_165, B_40], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf419, buf416, 294912, stream=stream0)
            buf420 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_122], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf419, buf414, out=buf420)
            buf421 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf422 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_112, X_57, mul_167, X_58, transpose_55, A_41], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_25.run(buf410, buf413, buf420, buf421, buf422, 294912, stream=stream0)
            buf423 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_41], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf421, buf422, out=buf423)
            buf424 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_169], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf423, buf424, 294912, stream=stream0)
            buf425 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_169, matmul_124], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf424, buf423, out=buf425)
            buf426 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [mul_168, B_41], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf426, buf423, 294912, stream=stream0)
            buf427 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_125], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf426, buf421, out=buf427)
            buf428 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf429 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_112, X_57, mul_167, X_58, mul_170, X_59, transpose_56, A_42], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_26.run(buf410, buf413, buf420, buf427, buf428, buf429, 294912, stream=stream0)
            buf430 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_42], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf428, buf429, out=buf430)
            buf431 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_172], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf430, buf431, 294912, stream=stream0)
            buf432 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_172, matmul_127], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf431, buf430, out=buf432)
            buf433 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [mul_171, B_42], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf433, buf430, 294912, stream=stream0)
            buf434 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_128], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf433, buf428, out=buf434)
            buf435 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.float32)
            buf436 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf437 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [add_112, X_57, mul_167, X_58, mul_170, X_59, mul_173, X_60, transpose_57, A_43], Original ATen: [aten.add, aten.div, aten.mul, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mul_transpose_27.run(buf410, buf413, buf420, buf427, buf434, buf435, buf436, buf437, 294912, stream=stream0)
            buf438 = buf434; del buf434  # reuse
            # Topologically Sorted Source Nodes: [transpose_57, A_43], Original ATen: [aten.transpose, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf436, buf437, out=buf438)
            buf439 = buf427; del buf427  # reuse
            # Topologically Sorted Source Nodes: [mul_175], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf438, buf439, 294912, stream=stream0)
            buf440 = buf420; del buf420  # reuse
            # Topologically Sorted Source Nodes: [mul_175, matmul_130], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf439, buf438, out=buf440)
            buf441 = buf440; del buf440  # reuse
            # Topologically Sorted Source Nodes: [mul_174, B_43], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf441, buf438, 294912, stream=stream0)
            buf442 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_131], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf441, buf436, out=buf442)
            buf443 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf444 = empty_strided_cuda((8, 192, 192), (36864, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_176, X_61, transpose_58, A_44], Original ATen: [aten.mul, aten.add, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_28.run(buf435, buf442, buf443, buf444, 294912, stream=stream0)
            buf445 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_44], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf443, buf444, out=buf445)
            buf446 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_178], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf445, buf446, 294912, stream=stream0)
            buf447 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_178, matmul_133], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf446, buf445, out=buf447)
            buf448 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [mul_177, B_44], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf448, buf445, 294912, stream=stream0)
            buf449 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_134], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf448, buf443, out=buf449)
            buf450 = buf355; del buf355  # reuse
            buf455 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf456 = reinterpret_tensor(buf455, (8, 192, 1), (192, 1, 1), 0); del buf455  # reuse
            buf465 = empty_strided_cuda((8, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_146, X_47, mul_149, X_48, w1_main_2, norm_19, add_127, truediv_16, w1_2, bmm_29], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45.run(buf450, buf456, buf298, buf362, buf369, buf3, buf465, 1536, 192, stream=stream0)
            del buf362
            buf451 = buf395; del buf395  # reuse
            buf453 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf454 = reinterpret_tensor(buf453, (8, 192, 1), (192, 1, 1), 0); del buf453  # reuse
            buf462 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [mul_161, X_54, mul_164, X_55, w0_main_2, norm_18, add_126, truediv_15, w0_2, bmm_28], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45.run(buf451, buf454, buf299, buf402, buf409, buf1, buf462, 1536, 192, stream=stream0)
            del buf402
            buf452 = buf435; del buf435  # reuse
            buf457 = empty_strided_cuda((8, 192, 1), (192, 1, 1536), torch.float32)
            buf458 = reinterpret_tensor(buf457, (8, 192, 1), (192, 1, 1), 0); del buf457  # reuse
            buf459 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [mul_176, X_61, mul_179, X_62, w2_main_2, norm_20, add_128, truediv_17, w2_2, h_3], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_45.run(buf452, buf458, buf300, buf442, buf449, buf5, buf459, 1536, 192, stream=stream0)
            del buf442
            del buf449
            buf460 = empty_strided_cuda((8, 192, 1024), (196608, 1, 192), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi_3, h_3], Original ATen: [aten.transpose, aten.slice, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_slice_transpose_51.run(primals_5, buf460, 1572864, stream=stream0)
            del primals_5
            buf461 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi_3, h_3], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf459, buf460, out=buf461)
            buf463 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_28], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf462, buf460, out=buf463)
            buf464 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_3, mul_183], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_31.run(buf463, buf461, buf464, 1572864, stream=stream0)
            buf466 = empty_strided_cuda((8, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_29], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf465, buf464, out=buf466)
            buf467 = empty_strided_cuda((8, 192, 4096), (786432, 4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_zeros_like_52.run(buf466, buf314, buf161, buf11, buf467, 6291456, stream=stream0)
            del buf11
            del buf161
            del buf314
            del buf466
            buf468 = empty_strided_cuda((8, 4096, 192), (786432, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_zeros_like_53.run(buf467, buf468, 32768, 192, stream=stream0)
            del buf467
        return (buf468, primals_1, primals_2, primals_3, primals_7, primals_8, primals_9, primals_10, buf1, buf3, buf5, buf7, buf8, buf13, buf14, buf16, buf20, buf22, buf24, buf26, buf29, buf68, buf105, buf108, buf145, buf146, buf147, buf149, buf151, buf153, buf156, buf158, buf163, buf164, buf166, buf176, buf177, buf178, buf179, buf182, buf221, buf258, buf261, buf298, buf299, buf300, buf302, buf304, buf306, buf309, buf311, buf316, buf317, buf319, buf329, buf330, buf333, buf370, buf373, buf410, buf413, buf450, buf451, buf452, buf454, buf456, buf458, buf461, buf463, reinterpret_tensor(buf465, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf464, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf462, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf460, (8, 1024, 192), (196608, 192, 1), 0), reinterpret_tensor(buf459, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf448, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf443, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf446, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf445, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf444, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf441, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf436, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf439, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf438, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf437, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf433, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf428, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf431, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf430, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf429, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf426, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf421, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf424, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf423, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf422, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf419, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf414, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf417, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf416, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf415, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf408, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf403, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf406, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf405, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf404, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf401, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf396, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf399, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf398, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf397, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf393, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf388, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf391, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf390, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf389, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf386, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf381, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf384, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf383, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf382, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf379, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf374, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf377, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf376, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf375, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf368, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf363, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf366, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf365, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf364, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf361, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf356, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf359, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf358, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf357, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf353, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf348, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf351, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf350, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf349, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf346, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf341, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf344, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf343, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf342, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf339, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf334, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf337, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf336, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf335, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf320, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf326, (8, 192, 1024), (196608, 1, 192), 0), reinterpret_tensor(buf321, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf324, (8, 192, 1024), (196608, 1, 192), 0), reinterpret_tensor(primals_6, (8, 1024, 192), (786432, 192, 1), 393216), reinterpret_tensor(buf322, (8, 192, 1024), (196608, 1024, 1), 0), reinterpret_tensor(buf318, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf307, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf315, (8, 1024, 192), (196608, 192, 1), 0), reinterpret_tensor(buf310, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf313, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf312, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf308, (8, 1024, 192), (196608, 192, 1), 0), reinterpret_tensor(buf296, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf291, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf294, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf293, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf292, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf289, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf284, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf287, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf286, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf285, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf281, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf276, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf279, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf278, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf277, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf274, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf269, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf272, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf271, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf270, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf267, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf262, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf265, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf264, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf263, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf256, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf251, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf254, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf253, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf252, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf249, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf244, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf247, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf246, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf245, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf241, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf236, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf239, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf238, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf237, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf234, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf229, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf232, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf231, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf230, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf227, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf222, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf225, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf224, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf223, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf217, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf212, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf215, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf214, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf213, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf210, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf205, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf208, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf207, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf206, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf202, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf197, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf200, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf199, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf198, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf195, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf190, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf193, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf192, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf191, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf188, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf183, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf186, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf185, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf184, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf167, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf173, (8, 192, 1024), (196608, 1, 192), 0), reinterpret_tensor(buf168, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf171, (8, 192, 1024), (196608, 1, 192), 0), reinterpret_tensor(primals_6, (8, 1024, 192), (786432, 192, 1), 196608), reinterpret_tensor(buf169, (8, 192, 1024), (196608, 1024, 1), 0), reinterpret_tensor(buf165, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf154, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf162, (8, 1024, 192), (196608, 192, 1), 0), reinterpret_tensor(buf157, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf160, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf159, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf155, (8, 1024, 192), (196608, 192, 1), 0), reinterpret_tensor(buf143, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf138, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf141, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf140, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf139, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf136, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf131, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf134, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf133, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf132, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf128, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf123, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf126, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf125, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf124, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf121, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf116, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf119, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf118, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf117, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf114, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf109, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf112, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf111, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf110, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf103, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf98, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf101, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf100, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf99, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf96, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf91, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf94, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf93, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf92, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf89, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf84, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf87, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf86, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf85, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf81, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf76, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf79, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf78, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf77, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf74, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf69, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf72, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf71, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf70, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf64, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf59, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf62, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf61, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf60, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf57, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf52, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf55, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf54, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf53, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf50, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf45, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf48, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf47, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf46, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf42, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf37, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf40, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf39, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf38, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf35, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf30, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf33, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf32, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf31, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf17, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf23, (8, 192, 1024), (196608, 1, 192), 0), reinterpret_tensor(buf18, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf21, (8, 192, 1024), (196608, 1, 192), 0), reinterpret_tensor(primals_6, (8, 1024, 192), (786432, 192, 1), 0), reinterpret_tensor(buf19, (8, 192, 1024), (196608, 1024, 1), 0), reinterpret_tensor(buf15, (8, 192, 192), (36864, 192, 1), 0), reinterpret_tensor(buf12, (8, 1024, 192), (196608, 192, 1), 0), reinterpret_tensor(buf10, (8, 192, 192), (36864, 1, 192), 0), reinterpret_tensor(buf9, (8, 1024, 192), (196608, 1, 1024), 0), reinterpret_tensor(buf6, (8, 1024, 192), (196608, 192, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((8, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((8, 4096, 1), (4096, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((8, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
