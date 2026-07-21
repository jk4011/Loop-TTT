# AOT ID: ['2_inference']
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/k6/ck6hon2mjjoewrsotmwtkbmq2bwqt7qxcec56ppbhnuy2hmocitw.py
# Topologically Sorted Source Nodes: [bmm_1, gate_before_act], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   bmm_1 => convert_element_type_3
#   gate_before_act => convert_element_type_11
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_3,%convert_element_type_11
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14155776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
    tl.store(out_ptr1 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/mc/cmcd7ig2wo5mdiz7j7kujscitwmjkkmawyumnwojbroajythc7sm.py
# Topologically Sorted Source Nodes: [w1_norm, transpose_4, dhidden], Original ATen: [aten.linalg_vector_norm, aten.transpose, aten._to_copy]
# Source node to ATen node mapping:
#   dhidden => convert_element_type_19
#   transpose_4 => permute_5
#   w1_norm => pow_3, sum_2
# Graph fragment:
#   %arg1_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %pow_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [2], True), kwargs = {})
#   %permute_5 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 2, 1]), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_5, torch.bfloat16), kwargs = {})
#   return %sum_2,%convert_element_type_19
triton_per_fused__to_copy_linalg_vector_norm_transpose_1 = async_compile.triton('triton_per_fused__to_copy_linalg_vector_norm_transpose_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_linalg_vector_norm_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 9437184}}
)
@triton.jit
def triton_per_fused__to_copy_linalg_vector_norm_transpose_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp6, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/to/ctobb6mwcdsp2ydvshe6zt4dyspvl4cxxrlrpagb3waw5l5v4ndg.py
# Topologically Sorted Source Nodes: [silu_1, hidden, transpose_5, lr1i, mul_7, type_as, dgate, sigma, mul_4, sub, mul_5, add, dx], Original ATen: [aten.silu, aten.mul, aten.transpose, aten.slice, aten._to_copy, aten.sigmoid, aten.rsub, aten.add]
# Source node to ATen node mapping:
#   add => add
#   dgate => mul_6
#   dx => mul_9
#   hidden => mul_3
#   lr1i => slice_4
#   mul_4 => mul_7
#   mul_5 => mul_8
#   mul_7 => mul_10
#   sigma => sigmoid_3
#   silu_1 => convert_element_type_17, convert_element_type_18, mul_2, sigmoid_1
#   sub => sub
#   transpose_5 => permute_6
#   type_as => convert_element_type_24
# Graph fragment:
#   %bmm_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %bmm_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %arg7_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %bmm_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %convert_element_type_17 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_17,), kwargs = {})
#   %mul_2 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %sigmoid_1), kwargs = {})
#   %convert_element_type_18 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %bmm_4), kwargs = {})
#   %permute_6 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_3, [0, 2, 1]), kwargs = {})
#   %slice_4 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg7_1, 1, 0, 1024), kwargs = {})
#   %mul_10 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, %slice_4), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_5, %bmm_4), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_3,), kwargs = {})
#   %mul_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %sigmoid_3), kwargs = {})
#   %sub : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_3), kwargs = {})
#   %mul_8 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, %sub), kwargs = {})
#   %add : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1), kwargs = {})
#   %mul_9 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add), kwargs = {})
#   return %convert_element_type_24,%mul_9
triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 113246208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1024)
    x2 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (3*x0 + 12288*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11 * tmp5
    tmp13 = tl.sigmoid(tmp0)
    tmp14 = tmp12 * tmp13
    tmp15 = 1.0
    tmp16 = tmp15 - tmp13
    tmp17 = tmp0 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/mo/cmof3luhrubafgkhwgolyv5lzsbenjkyn53heop2jxd65spc6sot.py
# Topologically Sorted Source Nodes: [m_i, m_i_1], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i => slice_9
#   m_i_1 => mean
# Graph fragment:
#   %arg3_1 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   return %buf13
triton_per_fused_mean_slice_3 = async_compile.triton('triton_per_fused_mean_slice_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_mean_slice_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/nn/cnngjoadfrur65ngoxywizq6p6kjhphajx4xwvpq5keh5khcxlbd.py
# Topologically Sorted Source Nodes: [ki, lr0i, mul_8, type_as_1, lr2i, mul_9, type_as_2], Original ATen: [aten.slice, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   ki => slice_1
#   lr0i => slice_6
#   lr2i => slice_5
#   mul_8 => mul_11
#   mul_9 => mul_12
#   type_as_1 => convert_element_type_27
#   type_as_2 => convert_element_type_30
# Graph fragment:
#   %arg6_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg8_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %slice_1 : Tensor "bf16[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 0, 1024), kwargs = {})
#   %slice_6 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg9_1, 1, 0, 1024), kwargs = {})
#   %mul_11 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, %slice_6), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %slice_5 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg8_1, 1, 0, 1024), kwargs = {})
#   %mul_12 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, %slice_5), kwargs = {})
#   %convert_element_type_30 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   return %convert_element_type_27,%convert_element_type_30
triton_poi_fused__to_copy_mul_slice_4 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62914560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 196608
    x3 = (xindex % 196608)
    x1 = ((xindex // 192) % 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + 786432*x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (3*x1 + 12288*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (3*x1 + 12288*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp4, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/nq/cnq3ounvko2ylhdxwzhuz5cjtj4xizx36puz4f6tl6xhgn546gti.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, dw0_momentum, mul_10, dw0_1, X_7, norm_4], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type_33
#   X_7 => convert_element_type_80
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   mul_10 => mul_13
#   mul_11 => mul_14
#   norm_3 => convert_element_type_34, pow_7, sum_4
#   norm_4 => convert_element_type_81, pow_9, sum_5
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_34 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_33, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_34, 2), kwargs = {})
#   %sum_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %convert_element_type_80 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_81 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_80, torch.float32), kwargs = {})
#   %pow_9 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_81, 2), kwargs = {})
#   %sum_5 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2], True), kwargs = {})
#   return %buf14,%buf65
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 4718720}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp6 = 1024.0
        tmp7 = (tmp5 / tmp6)
        tmp8 = 0.0
        tmp9 = tmp8 * tmp7
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr2 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 + tmp9
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/j6/cj62whk7scckfzvsbbtwfpphbzov65qxbs5yalr6c23tkexicvdc.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type_33
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   mul_11 => mul_14
#   norm_3 => convert_element_type_34, pow_7, sum_4
# Graph fragment:
#   %buf14 : Tensor "f32[32, 1, 1, 5][5, 160, 160, 1]cuda:0" = PlaceHolder[target=buf14]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_34 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_33, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_34, 2), kwargs = {})
#   %sum_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   return %sum_4
triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6 = async_compile.triton('triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 400}}
)
@triton.jit
def triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/tz/ctzuspsqklhkld47mddwjn6g5uq4xf5ky4itqzhmvspqmdz6xf4m.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, A, transpose_6, matmul_2], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A => convert_element_type_35, convert_element_type_36
#   X => convert_element_type_33
#   X_1 => div
#   add_4 => add_4
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   matmul_2 => convert_element_type_41
#   mul_11 => mul_14
#   norm_3 => pow_8
#   transpose_6 => permute_7
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %sum_4 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_4]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_33, %add_4), kwargs = {})
#   %convert_element_type_36 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %permute_7 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1]), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.bfloat16), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %expand,%expand_1,%expand_5
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_7 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16515072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1024.0
    tmp4 = (tmp2 / tmp3)
    tmp5 = 0.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp1 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = 1e-07
    tmp13 = tmp11 + tmp12
    tmp14 = (tmp9 / tmp13)
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp15, None)
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/sk/csk2ghwx3mzaj5mwsvi2tk6xr3d3ratlrjnib625nsopuia7pyvl.py
# Topologically Sorted Source Nodes: [mul_14], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_14 => mul_17
# Graph fragment:
#   %expand_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %mul_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 2.927), kwargs = {})
#   return %expand_2
triton_poi_fused_mul_8 = async_compile.triton('triton_poi_fused_mul_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ic/ciclnmopjrmnzhar7tutpjwgm3z5hqk5rdv2l5cn3f4srnzjgcd6.py
# Topologically Sorted Source Nodes: [mul_13, B], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B => add_5
#   mul_13 => mul_16
# Graph fragment:
#   %expand_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %bmm_10 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %mul_16 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, -6.8946), kwargs = {})
#   %add_5 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %bmm_10), kwargs = {})
#   return %expand_4
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/vn/cvnngkc4vyfewuc7mws4l6r7jbpcciad242useabusdizv6eimad.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_15, X_2, A_1, transpose_7, matmul_5], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_1 => convert_element_type_44, convert_element_type_45
#   X => convert_element_type_33
#   X_1 => div
#   X_2 => add_6
#   add_4 => add_4
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   matmul_5 => convert_element_type_50
#   mul_11 => mul_14
#   mul_15 => mul_18
#   norm_3 => pow_8
#   transpose_7 => permute_8
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %sum_4 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_4]
#   %bmm_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_33, %add_4), kwargs = {})
#   %mul_18 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_11), kwargs = {})
#   %convert_element_type_45 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   %permute_8 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_6, [0, 2, 1]), kwargs = {})
#   %convert_element_type_44 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_8, torch.bfloat16), kwargs = {})
#   %convert_element_type_50 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   return %expand_6,%expand_7,%expand_11
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1024.0
    tmp4 = (tmp2 / tmp3)
    tmp5 = 0.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp1 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = 1e-07
    tmp13 = tmp11 + tmp12
    tmp14 = (tmp9 / tmp13)
    tmp15 = 4.0848
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp20, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ht/cht5y5rzjrwuq7w76cocdweftkzasa6rl7l5tyrirouj3o4dv34a.py
# Topologically Sorted Source Nodes: [mul_17], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_17 => mul_20
# Graph fragment:
#   %expand_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %mul_20 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, 2.6377), kwargs = {})
#   return %expand_8
triton_poi_fused_mul_11 = async_compile.triton('triton_poi_fused_mul_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/su/csubanzwyfbqspso75tgz3xdphl2tzczo3btfazejqjmhng62wpt.py
# Topologically Sorted Source Nodes: [mul_16, B_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_1 => add_7
#   mul_16 => mul_19
# Graph fragment:
#   %expand_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %bmm_13 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %mul_19 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, -6.3029), kwargs = {})
#   %add_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %bmm_13), kwargs = {})
#   return %expand_10
triton_poi_fused_add_mul_12 = async_compile.triton('triton_poi_fused_add_mul_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ja/cja2x7s5baxygxwk26b4yodvwsioginxj5mydk3fpo4mhh2h4ppo.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_15, X_2, mul_18, X_3, A_2, transpose_8, matmul_8, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, A_5, transpose_11, matmul_17], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_2 => convert_element_type_53, convert_element_type_54
#   A_5 => convert_element_type_82, convert_element_type_83
#   X => convert_element_type_33
#   X_1 => div
#   X_2 => add_6
#   X_3 => add_8
#   X_7 => convert_element_type_80
#   X_8 => div_1
#   add_15 => add_15
#   add_4 => add_4
#   dw0_1 => add_1
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   matmul_17 => convert_element_type_88
#   matmul_8 => convert_element_type_59
#   mul_10 => mul_13
#   mul_11 => mul_14
#   mul_15 => mul_18
#   mul_18 => mul_21
#   norm_3 => pow_8
#   norm_4 => pow_10
#   transpose_11 => permute_12
#   transpose_8 => permute_9
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %sum_4 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_4]
#   %bmm_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_8]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %sum_5 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_5]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_33, %add_4), kwargs = {})
#   %mul_18 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_11), kwargs = {})
#   %mul_21 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 3.9505), kwargs = {})
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %bmm_14), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_8, [0, 2, 1]), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_9, torch.bfloat16), kwargs = {})
#   %convert_element_type_59 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %convert_element_type_80 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %pow_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   %add_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_10, 1e-07), kwargs = {})
#   %div_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_80, %add_15), kwargs = {})
#   %convert_element_type_83 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   %permute_12 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_1, [0, 2, 1]), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_12, torch.bfloat16), kwargs = {})
#   %convert_element_type_88 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   return %add_8,%expand_12,%expand_13,%expand_17,%expand_30,%expand_31,%expand_35
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_13 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 47185920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp26 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp31 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1024.0
    tmp4 = (tmp2 / tmp3)
    tmp5 = 0.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp1 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = 1e-07
    tmp13 = tmp11 + tmp12
    tmp14 = (tmp9 / tmp13)
    tmp15 = 4.0848
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = 3.9505
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 + tmp6
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp32 = libdevice.sqrt(tmp31)
    tmp33 = tmp32 + tmp12
    tmp34 = (tmp30 / tmp33)
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp24, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
    tl.store(out_ptr2 + (x2), tmp25, None)
    tl.store(out_ptr3 + (x2), tmp25, None)
    tl.store(out_ptr4 + (x2), tmp35, None)
    tl.store(out_ptr5 + (x2), tmp35, None)
    tl.store(out_ptr6 + (x2), tmp35, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7c/c7cz6sh6kuxe2kgysqm3mlmsok3taoncgkyhmpfezydnbm7ikuku.py
# Topologically Sorted Source Nodes: [mul_20], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_20 => mul_23
# Graph fragment:
#   %expand_15 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %mul_23 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_15, 2.3037), kwargs = {})
#   return %expand_14
triton_poi_fused_mul_14 = async_compile.triton('triton_poi_fused_mul_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/lz/clzse7etvvakrt4rgutjzlsmvwlvcv6g65v32dmjoankjyu6axbk.py
# Topologically Sorted Source Nodes: [mul_19, B_2], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_2 => add_9
#   mul_19 => mul_22
# Graph fragment:
#   %expand_15 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %bmm_16 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_16]
#   %mul_22 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_15, -5.5913), kwargs = {})
#   %add_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %bmm_16), kwargs = {})
#   return %expand_16
triton_poi_fused_add_mul_15 = async_compile.triton('triton_poi_fused_add_mul_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/l6/cl6zbpwb63mkvvqybkubrtd2pzacvleuur5k62i6xlmk2alegeop.py
# Topologically Sorted Source Nodes: [mul_21, X_4, A_3, transpose_9, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_3 => convert_element_type_62, convert_element_type_63
#   X_4 => add_10
#   matmul_11 => convert_element_type_68
#   mul_21 => mul_24
#   transpose_9 => permute_10
# Graph fragment:
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_8]
#   %bmm_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %mul_24 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.7418), kwargs = {})
#   %add_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %bmm_17), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   %permute_10 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_10, [0, 2, 1]), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_10, torch.bfloat16), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   return %expand_18,%expand_19,%expand_23
triton_poi_fused__to_copy_add_mul_transpose_16 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
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
    tl.store(out_ptr2 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ow/cowhpi7jr3cakaystxlp6747dnevxmlkayszglsnfdyvv6mnpi55.py
# Topologically Sorted Source Nodes: [mul_23], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_23 => mul_26
# Graph fragment:
#   %expand_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %mul_26 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_18, 1.2046), kwargs = {})
#   return %expand_20
triton_poi_fused_mul_17 = async_compile.triton('triton_poi_fused_mul_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/z4/cz4s5fvwzstaucx72ine3giueaqskiaobofqspudxz3ctqwrbn2f.py
# Topologically Sorted Source Nodes: [mul_22, B_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_3 => add_11
#   mul_22 => mul_25
# Graph fragment:
#   %expand_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %bmm_19 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_19]
#   %mul_25 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_18, -3.1427), kwargs = {})
#   %add_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %bmm_19), kwargs = {})
#   return %expand_22
triton_poi_fused_add_mul_18 = async_compile.triton('triton_poi_fused_add_mul_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/hc/chctfaelaruxz2744cmidpbb44quzcgq247osvo3q3d7cozb2lnp.py
# Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, A_4, transpose_10, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_4 => convert_element_type_71, convert_element_type_72
#   X_4 => add_10
#   X_5 => add_12
#   matmul_14 => convert_element_type_77
#   mul_21 => mul_24
#   mul_24 => mul_27
#   transpose_10 => permute_11
# Graph fragment:
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_8]
#   %bmm_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %bmm_20 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %mul_24 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.7418), kwargs = {})
#   %add_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %bmm_17), kwargs = {})
#   %mul_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.8769), kwargs = {})
#   %add_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %bmm_20), kwargs = {})
#   %convert_element_type_72 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   %permute_11 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_12, [0, 2, 1]), kwargs = {})
#   %convert_element_type_71 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_11, torch.bfloat16), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   return %expand_24,%expand_25,%expand_29
triton_poi_fused__to_copy_add_mul_transpose_19 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
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
    tl.store(out_ptr2 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/jk/cjkwa77sv3gbgacr65yjwkdn7litshvmrhmw5dnfufczth55lbnp.py
# Topologically Sorted Source Nodes: [mul_26], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_26 => mul_29
# Graph fragment:
#   %expand_27 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %mul_29 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_21, 1.2012), kwargs = {})
#   return %expand_26
triton_poi_fused_mul_20 = async_compile.triton('triton_poi_fused_mul_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ag/cagawx56sczlrnip2wnpqyhbs6abj2rkd5ojaaodwc47daimuyge.py
# Topologically Sorted Source Nodes: [mul_25, B_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_4 => add_13
#   mul_25 => mul_28
# Graph fragment:
#   %expand_27 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %bmm_22 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_22]
#   %mul_28 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_21, -3.0525), kwargs = {})
#   %add_13 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %bmm_22), kwargs = {})
#   return %expand_28
triton_poi_fused_add_mul_21 = async_compile.triton('triton_poi_fused_add_mul_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ma/cmaqshx4onvu6rvo4tlw5y3jlfrhipypvg563xopucphl753wlvg.py
# Topologically Sorted Source Nodes: [bmm_2, mul_21, X_4, mul_24, X_5, mul_27, X_6, w1_main, norm_7, add_41, truediv_4, w1_norm, w1, bmm_11, transpose_23, dhidden_1], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   X_4 => add_10
#   X_5 => add_12
#   X_6 => add_14
#   add_41 => add_41
#   bmm_11 => convert_element_type_182
#   bmm_2 => convert_element_type_8
#   dhidden_1 => convert_element_type_193
#   mul_21 => mul_24
#   mul_24 => mul_27
#   mul_27 => mul_30
#   norm_7 => pow_15, pow_16, sum_8
#   transpose_23 => permute_24
#   truediv_4 => div_4
#   w1 => mul_62
#   w1_main => add_37
#   w1_norm => pow_4
# Graph fragment:
#   %arg1_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_8]
#   %bmm_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %bmm_20 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %bmm_23 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_23]
#   %add_37 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_37]
#   %sum_8 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_8]
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_2]
#   %convert_element_type_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   %mul_24 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.7418), kwargs = {})
#   %add_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %bmm_17), kwargs = {})
#   %mul_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.8769), kwargs = {})
#   %add_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %bmm_20), kwargs = {})
#   %mul_30 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 2.8366), kwargs = {})
#   %add_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %bmm_23), kwargs = {})
#   %add_37 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %add_14), kwargs = {})
#   %pow_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_37, 2), kwargs = {})
#   %sum_8 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_15, [2], True), kwargs = {})
#   %pow_16 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_8, 0.5), kwargs = {})
#   %add_41 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_16, 1e-05), kwargs = {})
#   %div_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_37, %add_41), kwargs = {})
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mul_62 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %pow_4), kwargs = {})
#   %convert_element_type_182 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_62, torch.bfloat16), kwargs = {})
#   %permute_24 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_62, [0, 2, 1]), kwargs = {})
#   %convert_element_type_193 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_24, torch.bfloat16), kwargs = {})
#   return %convert_element_type_8,%add_37,%sum_8,%convert_element_type_182,%convert_element_type_193
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_22 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 40108032}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 3.7418
    tmp4 = tmp2 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = 2.8769
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = 2.8366
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp0 + tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = (tmp18 / tmp26)
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tmp27 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 192*x0), tmp1, r0_mask & xmask)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp18, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp31, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/hj/chjtt24pwob37b6tt4ddcizp5stdg6xsjmc6anyfoo74iqzlts4z.py
# Topologically Sorted Source Nodes: [gate, mul], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   gate => convert_element_type_6, convert_element_type_7, mul, sigmoid
#   mul => mul_1
# Graph fragment:
#   %bmm_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm]
#   %convert_element_type_6 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_1, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %bmm), kwargs = {})
#   return %mul_1
triton_poi_fused_mul_silu_23 = async_compile.triton('triton_poi_fused_mul_silu_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/5j/c5jxpovqypaihuhp3iqqwexilq6hb4rnhvc4bhtxwbrkbjvhkmux.py
# Topologically Sorted Source Nodes: [silu_2, dhidden_before_mul], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   dhidden_before_mul => mul_5
#   silu_2 => convert_element_type_22, convert_element_type_23, mul_4, sigmoid_2
# Graph fragment:
#   %bmm_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %convert_element_type_22 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_2 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_22,), kwargs = {})
#   %mul_4 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_22, %sigmoid_2), kwargs = {})
#   %convert_element_type_23 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_5, %convert_element_type_23), kwargs = {})
#   return %mul_5
triton_poi_fused_mul_silu_24 = async_compile.triton('triton_poi_fused_mul_silu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/vs/cvs46bnafd7i2gfmqudbvglfkyz6vonzwv3fjgmvpgdjgd2lbg3g.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_14 => convert_element_type_127
#   dw2_1 => add_3
#   dw2_momentum => full_default_2
#   m_i => slice_9
#   m_i_1 => mean
#   mul_12 => mul_15
#   norm_5 => convert_element_type_128, pow_11, sum_6
# Graph fragment:
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_15), kwargs = {})
#   %convert_element_type_127 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_128 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_127, torch.float32), kwargs = {})
#   %pow_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_128, 2), kwargs = {})
#   %sum_6 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2], True), kwargs = {})
#   return %buf116
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_25 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 2359360}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_25(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp5 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 1024.0
        tmp7 = (tmp5 / tmp6)
        tmp8 = 0.0
        tmp9 = tmp8 * tmp7
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ml/cmlku45yflsvng3u6yix2s3t55u5pdvx2qbjeb24mfvvo53yhb5s.py
# Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, mul_42, X_13, w0_main, norm_6, add_40, truediv_3, w0_norm, w0, bmm_10, gate_before_act_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_11 => add_21
#   X_12 => add_23
#   X_13 => add_25
#   add_40 => add_40
#   bmm_10 => convert_element_type_177
#   gate_before_act_1 => convert_element_type_185
#   mul_36 => mul_39
#   mul_39 => mul_42
#   mul_42 => mul_45
#   norm_6 => pow_13, pow_14, sum_7
#   truediv_3 => div_3
#   w0 => mul_61
#   w0_main => add_38
#   w0_norm => pow_1, pow_2, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %add_19 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_19]
#   %bmm_32 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_32]
#   %bmm_35 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_35]
#   %bmm_38 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_38]
#   %add_38 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_38]
#   %sum_7 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_7]
#   %sum_1 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_1]
#   %mul_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, 3.7418), kwargs = {})
#   %add_21 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %bmm_32), kwargs = {})
#   %mul_42 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 2.8769), kwargs = {})
#   %add_23 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %bmm_35), kwargs = {})
#   %mul_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, 2.8366), kwargs = {})
#   %add_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %bmm_38), kwargs = {})
#   %add_38 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %add_25), kwargs = {})
#   %pow_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_38, 2), kwargs = {})
#   %sum_7 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [2], True), kwargs = {})
#   %pow_14 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %add_40 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_14, 1e-05), kwargs = {})
#   %div_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_38, %add_40), kwargs = {})
#   %pow_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %sum_1 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2], True), kwargs = {})
#   %pow_2 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mul_61 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %pow_2), kwargs = {})
#   %convert_element_type_177 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_61, torch.bfloat16), kwargs = {})
#   %convert_element_type_185 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_61, torch.bfloat16), kwargs = {})
#   return %sum_1,%add_38,%sum_7,%convert_element_type_177,%convert_element_type_185
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_26 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 35389440}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp6 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = 3.7418
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 2.8769
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = 2.8366
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp0 + tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(r0_mask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = (tmp22 / tmp30)
    tmp32 = libdevice.sqrt(tmp5)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp22, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp34, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp34, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7f/c7fintvbdx5fm554vr4f5wtyvnbc3wwkdtt6rx2t7qyeiqdaypzh.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, mul_45, X_16, mul_48, X_17, A_12, transpose_18, matmul_38], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_12 => convert_element_type_147, convert_element_type_148
#   X_14 => convert_element_type_127
#   X_15 => div_2
#   X_16 => add_28
#   X_17 => add_30
#   add_26 => add_26
#   dw2_1 => add_3
#   dw2_momentum => full_default_2
#   m_i => slice_9
#   m_i_1 => mean
#   matmul_38 => convert_element_type_153
#   mul_12 => mul_15
#   mul_45 => mul_48
#   mul_48 => mul_51
#   norm_5 => pow_12
#   transpose_18 => permute_19
# Graph fragment:
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %sum_6 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_6]
#   %bmm_41 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_41]
#   %bmm_44 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_44]
#   %add_30 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_30]
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_15), kwargs = {})
#   %convert_element_type_127 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %pow_12 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %add_26 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, 1e-07), kwargs = {})
#   %div_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_127, %add_26), kwargs = {})
#   %mul_48 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 4.0848), kwargs = {})
#   %add_28 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %bmm_41), kwargs = {})
#   %mul_51 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 3.9505), kwargs = {})
#   %add_30 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %bmm_44), kwargs = {})
#   %convert_element_type_148 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.bfloat16), kwargs = {})
#   %permute_19 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_30, [0, 2, 1]), kwargs = {})
#   %convert_element_type_147 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_19, torch.bfloat16), kwargs = {})
#   %convert_element_type_153 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.bfloat16), kwargs = {})
#   return %add_30,%expand_72,%expand_73,%expand_77
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30670848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1024.0
    tmp4 = (tmp2 / tmp3)
    tmp5 = 0.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp1 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = 1e-07
    tmp13 = tmp11 + tmp12
    tmp14 = (tmp9 / tmp13)
    tmp15 = 4.0848
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = 3.9505
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp24, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
    tl.store(out_ptr2 + (x2), tmp25, None)
    tl.store(out_ptr3 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/3b/c3bxisgewugbrflhjr3jzegqhrexypof3xhf5jxyvbaoigrfrm6u.py
# Topologically Sorted Source Nodes: [m_i_2, m_i_3], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i_2 => slice_19
#   m_i_3 => mean_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %slice_19 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_19, [1], True), kwargs = {})
#   return %buf173
triton_per_fused_mean_slice_28 = async_compile.triton('triton_per_fused_mean_slice_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_mean_slice_28(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/4v/c4vi3w77r3fjvupfk3p45aeglcvcseoqdhftcdhvpqaqebp2yjlu.py
# Topologically Sorted Source Nodes: [silu_5, dhidden_before_mul_1], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   dhidden_before_mul_1 => mul_69
#   silu_5 => convert_element_type_196, convert_element_type_197, mul_68, sigmoid_6
# Graph fragment:
#   %bmm_59 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %bmm_57 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %convert_element_type_196 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_57, torch.float32), kwargs = {})
#   %sigmoid_6 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_196,), kwargs = {})
#   %mul_68 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_196, %sigmoid_6), kwargs = {})
#   %convert_element_type_197 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_68, torch.bfloat16), kwargs = {})
#   %mul_69 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_59, %convert_element_type_197), kwargs = {})
#   return %mul_69
triton_poi_fused_mul_silu_29 = async_compile.triton('triton_poi_fused_mul_silu_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/zp/czp63ll5vdgewx47dgzl2aeanhsf7h7ibojwwsltrh2aly6ekrhv.py
# Topologically Sorted Source Nodes: [ki_1, lr0i_1, mul_69, type_as_4, lr2i_1, mul_70, type_as_5], Original ATen: [aten.slice, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   ki_1 => slice_10
#   lr0i_1 => slice_15
#   lr2i_1 => slice_14
#   mul_69 => mul_75
#   mul_70 => mul_76
#   type_as_4 => convert_element_type_201
#   type_as_5 => convert_element_type_204
# Graph fragment:
#   %arg6_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg8_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %slice_10 : Tensor "bf16[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 1024, 2048), kwargs = {})
#   %slice_15 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg9_1, 1, 1024, 2048), kwargs = {})
#   %mul_75 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_10, %slice_15), kwargs = {})
#   %convert_element_type_201 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   %slice_14 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg8_1, 1, 1024, 2048), kwargs = {})
#   %mul_76 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_10, %slice_14), kwargs = {})
#   %convert_element_type_204 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_76, torch.bfloat16), kwargs = {})
#   return %convert_element_type_201,%convert_element_type_204
triton_poi_fused__to_copy_mul_slice_30 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62914560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 196608
    x3 = (xindex % 196608)
    x1 = ((xindex // 192) % 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (196608 + x3 + 786432*x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (3072 + 3*x1 + 12288*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (3072 + 3*x1 + 12288*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp4, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ji/cjiomcw5gls6p6max2rp5yawchokuoa6cpkb7ydptqapxpti2yn6.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, m_i_2, m_i_3, mul_73, dw2_3, X_35, norm_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_35 => convert_element_type_301
#   dw2_1 => add_3
#   dw2_3 => add_46
#   dw2_momentum => full_default_2
#   m_i => slice_9
#   m_i_1 => mean
#   m_i_2 => slice_19
#   m_i_3 => mean_1
#   mul_12 => mul_15
#   mul_73 => mul_79
#   norm_11 => convert_element_type_302, pow_23, sum_12
# Graph fragment:
#   %bmm_62 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_62]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %buf173 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf173]
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_15), kwargs = {})
#   %slice_19 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_19, [1], True), kwargs = {})
#   %mul_79 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %mean_1), kwargs = {})
#   %add_46 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_62, %mul_79), kwargs = {})
#   %convert_element_type_301 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_46, torch.bfloat16), kwargs = {})
#   %convert_element_type_302 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_301, torch.float32), kwargs = {})
#   %pow_23 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_302, 2), kwargs = {})
#   %sum_12 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_23, [1, 2], True), kwargs = {})
#   return %buf276
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_31 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 4718720}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 1024.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 0.0
        tmp11 = tmp10 * tmp9
        tmp12 = tmp6 + tmp11
        tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = (tmp13 / tmp8)
        tmp15 = tmp12 * tmp14
        tmp16 = tmp4 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(r0_mask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/sg/csg3agrmn2d7jdbmbsjoqhgvjtuunf42akxqslchmpndhgqealmm.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, m_i_2, m_i_3, mul_73, dw2_3, X_35, norm_11, add_69, X_36, A_25, transpose_35, matmul_77], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_25 => convert_element_type_303, convert_element_type_304
#   X_35 => convert_element_type_301
#   X_36 => div_8
#   add_69 => add_69
#   dw2_1 => add_3
#   dw2_3 => add_46
#   dw2_momentum => full_default_2
#   m_i => slice_9
#   m_i_1 => mean
#   m_i_2 => slice_19
#   m_i_3 => mean_1
#   matmul_77 => convert_element_type_309
#   mul_12 => mul_15
#   mul_73 => mul_79
#   norm_11 => pow_24
#   transpose_35 => permute_36
# Graph fragment:
#   %bmm_62 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_62]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %buf173 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf173]
#   %sum_12 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_12]
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_8]
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_15), kwargs = {})
#   %slice_19 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_19, [1], True), kwargs = {})
#   %mul_79 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %mean_1), kwargs = {})
#   %add_46 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_62, %mul_79), kwargs = {})
#   %convert_element_type_301 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_46, torch.bfloat16), kwargs = {})
#   %pow_24 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_12, 0.5), kwargs = {})
#   %add_69 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_24, 1e-07), kwargs = {})
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_301, %add_69), kwargs = {})
#   %convert_element_type_304 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.bfloat16), kwargs = {})
#   %permute_36 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_8, [0, 2, 1]), kwargs = {})
#   %convert_element_type_303 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_36, torch.bfloat16), kwargs = {})
#   %convert_element_type_309 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.bfloat16), kwargs = {})
#   return %div_8,%expand_150,%expand_151,%expand_155
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_32 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28311552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 0.0
    tmp8 = tmp7 * tmp6
    tmp9 = tmp3 + tmp8
    tmp11 = (tmp10 / tmp5)
    tmp12 = tmp9 * tmp11
    tmp13 = tmp1 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = 1e-07
    tmp19 = tmp17 + tmp18
    tmp20 = (tmp15 / tmp19)
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
    tl.store(out_ptr3 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ca/ccacvpshi5gdweyofztyiomyj57f6v2rjnyiwptpua4p7yrnenfd.py
# Topologically Sorted Source Nodes: [mul_106, X_37, A_26, transpose_36, matmul_80], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_26 => convert_element_type_312, convert_element_type_313
#   X_37 => add_71
#   matmul_80 => convert_element_type_318
#   mul_106 => mul_112
#   transpose_36 => permute_37
# Graph fragment:
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_8]
#   %bmm_95 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_95]
#   %mul_112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, 4.0848), kwargs = {})
#   %add_71 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %bmm_95), kwargs = {})
#   %convert_element_type_313 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.bfloat16), kwargs = {})
#   %permute_37 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_71, [0, 2, 1]), kwargs = {})
#   %convert_element_type_312 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_37, torch.bfloat16), kwargs = {})
#   %convert_element_type_318 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.bfloat16), kwargs = {})
#   return %expand_156,%expand_157,%expand_161
triton_poi_fused__to_copy_add_mul_transpose_33 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_33(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = 4.0848
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr2 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/by/cbyudrx4mkaop6s27ermaqvwbbqfuff4rnysssejrxc72wunrl57.py
# Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, A_27, transpose_37, matmul_83], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_27 => convert_element_type_321, convert_element_type_322
#   X_37 => add_71
#   X_38 => add_73
#   matmul_83 => convert_element_type_327
#   mul_106 => mul_112
#   mul_109 => mul_115
#   transpose_37 => permute_38
# Graph fragment:
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_8]
#   %bmm_95 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_95]
#   %bmm_98 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_98]
#   %mul_112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, 4.0848), kwargs = {})
#   %add_71 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %bmm_95), kwargs = {})
#   %mul_115 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, 3.9505), kwargs = {})
#   %add_73 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, %bmm_98), kwargs = {})
#   %convert_element_type_322 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.bfloat16), kwargs = {})
#   %permute_38 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_73, [0, 2, 1]), kwargs = {})
#   %convert_element_type_321 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_38, torch.bfloat16), kwargs = {})
#   %convert_element_type_327 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.bfloat16), kwargs = {})
#   return %expand_162,%expand_163,%expand_167
triton_poi_fused__to_copy_add_mul_transpose_34 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp1 = 4.0848
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = 3.9505
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr1 + (x0), tmp11, None)
    tl.store(out_ptr2 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/o2/co22zefdmos3jypuajb4m4hjlnav2x4fp7b534miwwitgvaj7bzy.py
# Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_112, X_39, A_28, transpose_38, matmul_86], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_28 => convert_element_type_330, convert_element_type_331
#   X_37 => add_71
#   X_38 => add_73
#   X_39 => add_75
#   matmul_86 => convert_element_type_336
#   mul_106 => mul_112
#   mul_109 => mul_115
#   mul_112 => mul_118
#   transpose_38 => permute_39
# Graph fragment:
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_8]
#   %bmm_95 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_95]
#   %bmm_98 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_98]
#   %bmm_101 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_101]
#   %mul_112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, 4.0848), kwargs = {})
#   %add_71 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %bmm_95), kwargs = {})
#   %mul_115 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, 3.9505), kwargs = {})
#   %add_73 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, %bmm_98), kwargs = {})
#   %mul_118 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, 3.7418), kwargs = {})
#   %add_75 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %bmm_101), kwargs = {})
#   %convert_element_type_331 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_75, torch.bfloat16), kwargs = {})
#   %permute_39 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_75, [0, 2, 1]), kwargs = {})
#   %convert_element_type_330 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_39, torch.bfloat16), kwargs = {})
#   %convert_element_type_336 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_75, torch.bfloat16), kwargs = {})
#   return %expand_168,%expand_169,%expand_173
triton_poi_fused__to_copy_add_mul_transpose_35 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25952256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp1 = 4.0848
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = 3.9505
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = 3.7418
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
    tl.store(out_ptr2 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/g4/cg475riyytpa2q4c4g22jgrhrzdlpzg3zzrwwlmghxn63sodvqg2.py
# Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_112, X_39, mul_115, X_40, A_29, transpose_39, matmul_89], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_29 => convert_element_type_339, convert_element_type_340
#   X_37 => add_71
#   X_38 => add_73
#   X_39 => add_75
#   X_40 => add_77
#   matmul_89 => convert_element_type_345
#   mul_106 => mul_112
#   mul_109 => mul_115
#   mul_112 => mul_118
#   mul_115 => mul_121
#   transpose_39 => permute_40
# Graph fragment:
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_8]
#   %bmm_95 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_95]
#   %bmm_98 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_98]
#   %bmm_101 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_101]
#   %bmm_104 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_104]
#   %add_77 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_77]
#   %mul_112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, 4.0848), kwargs = {})
#   %add_71 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %bmm_95), kwargs = {})
#   %mul_115 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, 3.9505), kwargs = {})
#   %add_73 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, %bmm_98), kwargs = {})
#   %mul_118 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, 3.7418), kwargs = {})
#   %add_75 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %bmm_101), kwargs = {})
#   %mul_121 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_75, 2.8769), kwargs = {})
#   %add_77 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %bmm_104), kwargs = {})
#   %convert_element_type_340 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_77, torch.bfloat16), kwargs = {})
#   %permute_40 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_77, [0, 2, 1]), kwargs = {})
#   %convert_element_type_339 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_40, torch.bfloat16), kwargs = {})
#   %convert_element_type_345 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_77, torch.bfloat16), kwargs = {})
#   return %add_77,%expand_174,%expand_175,%expand_179
triton_poi_fused__to_copy_add_mul_transpose_36 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp1 = 4.0848
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = 3.9505
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = 3.7418
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = 2.8769
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
    tl.store(out_ptr0 + (x0), tmp21, None)
    tl.store(out_ptr1 + (x0), tmp21, None)
    tl.store(out_ptr2 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/f2/cf2ub52ory36upf65ax7qrxf2ufynzjgdwl4v2h2zfakiq4br6ld.py
# Topologically Sorted Source Nodes: [mul_51, X_18, mul_54, X_19, mul_57, X_20, w2_main, norm_8, add_42, truediv_5, w2_norm, w2, h_1, hidden_before_mul_1, mul_118, X_41, w2_main_1, norm_14, add_85, truediv_11, w2_1, h_2, hidden_before_mul_2], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_18 => add_32
#   X_19 => add_34
#   X_20 => add_36
#   X_41 => add_79
#   add_42 => add_42
#   add_85 => add_85
#   h_1 => convert_element_type_174
#   h_2 => convert_element_type_348
#   hidden_before_mul_1 => convert_element_type_188
#   hidden_before_mul_2 => convert_element_type_362
#   mul_118 => mul_124
#   mul_51 => mul_54
#   mul_54 => mul_57
#   mul_57 => mul_60
#   norm_14 => pow_29, pow_30, sum_15
#   norm_8 => pow_17, pow_18, sum_9
#   truediv_11 => div_11
#   truediv_5 => div_5
#   w2 => mul_63
#   w2_1 => mul_127
#   w2_main => add_39
#   w2_main_1 => add_82
#   w2_norm => pow_5, pow_6, sum_3
# Graph fragment:
#   %arg2_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_30 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_30]
#   %bmm_47 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_47]
#   %bmm_50 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_50]
#   %bmm_53 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_53]
#   %add_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_39]
#   %sum_9 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_9]
#   %sum_3 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_3]
#   %add_77 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_77]
#   %bmm_107 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_107]
#   %sum_15 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_15]
#   %mul_127 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=mul_127]
#   %mul_54 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, 3.7418), kwargs = {})
#   %add_32 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %bmm_47), kwargs = {})
#   %mul_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 2.8769), kwargs = {})
#   %add_34 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %bmm_50), kwargs = {})
#   %mul_60 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 2.8366), kwargs = {})
#   %add_36 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_60, %bmm_53), kwargs = {})
#   %add_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %add_36), kwargs = {})
#   %pow_17 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_39, 2), kwargs = {})
#   %sum_9 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_17, [2], True), kwargs = {})
#   %pow_18 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_9, 0.5), kwargs = {})
#   %add_42 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_18, 1e-05), kwargs = {})
#   %div_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, %add_42), kwargs = {})
#   %pow_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg2_1, 2), kwargs = {})
#   %sum_3 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [2], True), kwargs = {})
#   %pow_6 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %mul_63 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %pow_6), kwargs = {})
#   %convert_element_type_174 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   %convert_element_type_188 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   %mul_124 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_77, 2.8366), kwargs = {})
#   %add_79 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %bmm_107), kwargs = {})
#   %add_82 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %add_79), kwargs = {})
#   %pow_29 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_82, 2), kwargs = {})
#   %sum_15 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_29, [2], True), kwargs = {})
#   %pow_30 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_15, 0.5), kwargs = {})
#   %add_85 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_30, 1e-05), kwargs = {})
#   %div_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_82, %add_85), kwargs = {})
#   %mul_127 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_11, %pow_6), kwargs = {})
#   %convert_element_type_348 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_127, torch.bfloat16), kwargs = {})
#   %convert_element_type_362 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_127, torch.bfloat16), kwargs = {})
#   return %sum_3,%add_39,%sum_9,%convert_element_type_174,%convert_element_type_188,%sum_15,%mul_127,%convert_element_type_348,%convert_element_type_362
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr6': '*bf16', 'out_ptr7': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 3, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 51904512}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp6 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp37 = tl.load(in_ptr5 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = 3.7418
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 2.8769
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = 2.8366
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp0 + tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(r0_mask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = (tmp22 / tmp30)
    tmp32 = libdevice.sqrt(tmp5)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp35 * tmp17
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 + tmp38
    tmp40 = tmp22 + tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
    tmp44 = tl.where(r0_mask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None].to(tl.float32)
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tmp46 + tmp29
    tmp48 = (tmp40 / tmp47)
    tmp49 = tmp48 * tmp32
    tmp50 = tmp49.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp22, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp34, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp34, r0_mask & xmask)
    tl.store(out_ptr6 + (r0_1 + 192*x0), tmp50, r0_mask & xmask)
    tl.store(out_ptr7 + (r0_1 + 192*x0), tmp50, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/yb/cybdpqebbsv7jom6dkuxn52dwwobpb3ggcx7durd6yofavrmj6lp.py
# Topologically Sorted Source Nodes: [silu_4, hidden_1, transpose_24, lr1i_1, mul_68, type_as_3, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43, dx_1], Original ATen: [aten.silu, aten.mul, aten.transpose, aten.slice, aten._to_copy, aten.sigmoid, aten.rsub, aten.add]
# Source node to ATen node mapping:
#   add_43 => add_43
#   dgate_1 => mul_70
#   dx_1 => mul_73
#   hidden_1 => mul_67
#   lr1i_1 => slice_13
#   mul_65 => mul_71
#   mul_66 => mul_72
#   mul_68 => mul_74
#   sigma_1 => sigmoid_7
#   silu_4 => convert_element_type_191, convert_element_type_192, mul_66, sigmoid_5
#   sub_1 => sub_1
#   transpose_24 => permute_25
#   type_as_3 => convert_element_type_198
# Graph fragment:
#   %bmm_57 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %bmm_58 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_58]
#   %arg7_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %bmm_59 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %convert_element_type_191 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_57, torch.float32), kwargs = {})
#   %sigmoid_5 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_191,), kwargs = {})
#   %mul_66 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_191, %sigmoid_5), kwargs = {})
#   %convert_element_type_192 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_66, torch.bfloat16), kwargs = {})
#   %mul_67 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_192, %bmm_58), kwargs = {})
#   %permute_25 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_67, [0, 2, 1]), kwargs = {})
#   %slice_13 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg7_1, 1, 1024, 2048), kwargs = {})
#   %mul_74 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_25, %slice_13), kwargs = {})
#   %convert_element_type_198 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_74, torch.bfloat16), kwargs = {})
#   %mul_70 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_59, %bmm_58), kwargs = {})
#   %sigmoid_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_57,), kwargs = {})
#   %mul_71 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %sigmoid_7), kwargs = {})
#   %sub_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_7), kwargs = {})
#   %mul_72 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_57, %sub_1), kwargs = {})
#   %add_43 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, 1), kwargs = {})
#   %mul_73 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_71, %add_43), kwargs = {})
#   return %convert_element_type_198,%mul_73
triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_38 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 113246208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1024)
    x2 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (3072 + 3*x0 + 12288*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11 * tmp5
    tmp13 = tl.sigmoid(tmp0)
    tmp14 = tmp12 * tmp13
    tmp15 = 1.0
    tmp16 = tmp15 - tmp13
    tmp17 = tmp0 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/j6/cj6vapuxdfmdznr4r7ruxwio74lcrc6zabix6bsdmkrib63prrjb.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, dw0_momentum, mul_10, dw0_1, m_i_2, m_i_3, mul_72, dw1_3, X_21, norm_9, mul_71, dw0_3, X_28, norm_10], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_21 => convert_element_type_207
#   X_28 => convert_element_type_254
#   dw0_1 => add_1
#   dw0_3 => add_44
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_3 => add_45
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   m_i_2 => slice_19
#   m_i_3 => mean_1
#   mul_10 => mul_13
#   mul_11 => mul_14
#   mul_71 => mul_77
#   mul_72 => mul_78
#   norm_10 => convert_element_type_255, pow_21, sum_11
#   norm_9 => convert_element_type_208, pow_19, sum_10
# Graph fragment:
#   %bmm_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %buf173 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf173]
#   %bmm_61 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_61]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %slice_19 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_19, [1], True), kwargs = {})
#   %mul_78 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %mean_1), kwargs = {})
#   %add_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_60, %mul_78), kwargs = {})
#   %convert_element_type_207 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %convert_element_type_208 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_207, torch.float32), kwargs = {})
#   %pow_19 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_208, 2), kwargs = {})
#   %sum_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_19, [1, 2], True), kwargs = {})
#   %mul_77 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %mean_1), kwargs = {})
#   %add_44 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_61, %mul_77), kwargs = {})
#   %convert_element_type_254 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_44, torch.bfloat16), kwargs = {})
#   %convert_element_type_255 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_254, torch.float32), kwargs = {})
#   %pow_21 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_255, 2), kwargs = {})
#   %sum_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_21, [1, 2], True), kwargs = {})
#   return %buf174,%buf225
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_39 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 9437440}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp5 = tl.load(in_ptr1 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 1024.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 0.0
        tmp11 = tmp10 * tmp9
        tmp12 = tmp6 + tmp11
        tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = (tmp13 / tmp8)
        tmp15 = tmp12 * tmp14
        tmp16 = tmp4 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(r0_mask & xmask, tmp24, _tmp23)
        tmp25 = tl.load(in_ptr4 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tl.load(in_ptr5 + (36864*x1 + (((r0_2 + 7373*x0) % 36864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 + tmp11
        tmp30 = tmp29 * tmp14
        tmp31 = tmp26 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 * tmp33
        tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
        tmp36 = tl.where(tmp2, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask & xmask, tmp39, _tmp38)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tl.store(out_ptr1 + (x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/rg/crgf2rbolokjsbb5m6cpv7ckv7uzekrhtu7rh77ppvm6wc3e6q25.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, dw0_momentum, mul_10, dw0_1, m_i_2, m_i_3, mul_72, dw1_3, X_21, norm_9, add_47, X_22, A_15, transpose_25, matmul_47, mul_71, dw0_3, X_28, norm_10, add_58, X_29, A_20, transpose_30, matmul_62], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_15 => convert_element_type_209, convert_element_type_210
#   A_20 => convert_element_type_256, convert_element_type_257
#   X_21 => convert_element_type_207
#   X_22 => div_6
#   X_28 => convert_element_type_254
#   X_29 => div_7
#   add_47 => add_47
#   add_58 => add_58
#   dw0_1 => add_1
#   dw0_3 => add_44
#   dw0_momentum => full_default_1
#   dw1_1 => add_2
#   dw1_3 => add_45
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   m_i_2 => slice_19
#   m_i_3 => mean_1
#   matmul_47 => convert_element_type_215
#   matmul_62 => convert_element_type_262
#   mul_10 => mul_13
#   mul_11 => mul_14
#   mul_71 => mul_77
#   mul_72 => mul_78
#   norm_10 => pow_22
#   norm_9 => pow_20
#   transpose_25 => permute_26
#   transpose_30 => permute_31
# Graph fragment:
#   %bmm_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %buf173 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf173]
#   %sum_10 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_10]
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_6]
#   %bmm_61 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_61]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %sum_11 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_11]
#   %div_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_7]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_13), kwargs = {})
#   %slice_19 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_19, [1], True), kwargs = {})
#   %mul_78 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %mean_1), kwargs = {})
#   %add_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_60, %mul_78), kwargs = {})
#   %convert_element_type_207 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.bfloat16), kwargs = {})
#   %pow_20 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_10, 0.5), kwargs = {})
#   %add_47 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_207, %add_47), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   %permute_26 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_6, [0, 2, 1]), kwargs = {})
#   %convert_element_type_209 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_26, torch.bfloat16), kwargs = {})
#   %convert_element_type_215 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   %mul_77 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %mean_1), kwargs = {})
#   %add_44 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_61, %mul_77), kwargs = {})
#   %convert_element_type_254 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_44, torch.bfloat16), kwargs = {})
#   %pow_22 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_11, 0.5), kwargs = {})
#   %add_58 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_22, 1e-07), kwargs = {})
#   %div_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_254, %add_58), kwargs = {})
#   %convert_element_type_257 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.bfloat16), kwargs = {})
#   %permute_31 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_7, [0, 2, 1]), kwargs = {})
#   %convert_element_type_256 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_31, torch.bfloat16), kwargs = {})
#   %convert_element_type_262 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.bfloat16), kwargs = {})
#   return %div_6,%expand_90,%expand_91,%expand_95,%div_7,%expand_120,%expand_121,%expand_125
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_40 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'out_ptr7': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 56623104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp24 = tl.load(in_ptr6 + (x2), None).to(tl.float32)
    tmp31 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 0.0
    tmp8 = tmp7 * tmp6
    tmp9 = tmp3 + tmp8
    tmp11 = (tmp10 / tmp5)
    tmp12 = tmp9 * tmp11
    tmp13 = tmp1 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = 1e-07
    tmp19 = tmp17 + tmp18
    tmp20 = (tmp15 / tmp19)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp8
    tmp27 = tmp26 * tmp11
    tmp28 = tmp23 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp32 = libdevice.sqrt(tmp31)
    tmp33 = tmp32 + tmp18
    tmp34 = (tmp30 / tmp33)
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
    tl.store(out_ptr2 + (x2), tmp21, None)
    tl.store(out_ptr3 + (x2), tmp21, None)
    tl.store(out_ptr4 + (x2), tmp34, None)
    tl.store(out_ptr5 + (x2), tmp35, None)
    tl.store(out_ptr6 + (x2), tmp35, None)
    tl.store(out_ptr7 + (x2), tmp35, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/zp/czpi7o3spiyzg7zv6bszah33dayfrdq6nso3dbljobknlwy6qtce.py
# Topologically Sorted Source Nodes: [w1_norm, mul_88, X_27, w1_main_1, norm_13, add_84, truediv_10, w1_1, bmm_20, transpose_42, dhidden_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   X_27 => add_57
#   add_84 => add_84
#   bmm_20 => convert_element_type_356
#   dhidden_2 => convert_element_type_367
#   mul_88 => mul_94
#   norm_13 => pow_27, pow_28, sum_14
#   transpose_42 => permute_43
#   truediv_10 => div_10
#   w1_1 => mul_126
#   w1_main_1 => add_80
#   w1_norm => pow_4
# Graph fragment:
#   %add_37 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_37]
#   %add_55 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_55]
#   %bmm_77 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_77]
#   %sum_14 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_14]
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_2]
#   %mul_126 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=mul_126]
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mul_94 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 2.8366), kwargs = {})
#   %add_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %bmm_77), kwargs = {})
#   %add_80 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %add_57), kwargs = {})
#   %pow_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_80, 2), kwargs = {})
#   %sum_14 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_27, [2], True), kwargs = {})
#   %pow_28 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_14, 0.5), kwargs = {})
#   %add_84 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_28, 1e-05), kwargs = {})
#   %div_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_80, %add_84), kwargs = {})
#   %mul_126 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %pow_4), kwargs = {})
#   %convert_element_type_356 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_126, torch.bfloat16), kwargs = {})
#   %permute_43 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_126, [0, 2, 1]), kwargs = {})
#   %convert_element_type_367 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_43, torch.bfloat16), kwargs = {})
#   return %sum_14,%mul_126,%convert_element_type_356,%convert_element_type_367
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_41 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 21233664}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 2.8366
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = (tmp7 / tmp15)
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp20, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/32/c32abi6gbvrovijhmim6i2k4sx2swmi4b27heydblnsdqze5c2i3.py
# Topologically Sorted Source Nodes: [silu_7, hidden_2, transpose_43, lr1i_2, mul_129, type_as_6, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86, dx_2], Original ATen: [aten.silu, aten.mul, aten.transpose, aten.slice, aten._to_copy, aten.sigmoid, aten.rsub, aten.add]
# Source node to ATen node mapping:
#   add_86 => add_86
#   dgate_2 => mul_134
#   dx_2 => mul_137
#   hidden_2 => mul_131
#   lr1i_2 => slice_23
#   mul_126 => mul_135
#   mul_127 => mul_136
#   mul_129 => mul_138
#   sigma_2 => sigmoid_11
#   silu_7 => convert_element_type_365, convert_element_type_366, mul_130, sigmoid_9
#   sub_2 => sub_2
#   transpose_43 => permute_44
#   type_as_6 => convert_element_type_372
# Graph fragment:
#   %bmm_111 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_111]
#   %bmm_112 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %arg7_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %bmm_113 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_113]
#   %convert_element_type_365 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_111, torch.float32), kwargs = {})
#   %sigmoid_9 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_365,), kwargs = {})
#   %mul_130 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_365, %sigmoid_9), kwargs = {})
#   %convert_element_type_366 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_130, torch.bfloat16), kwargs = {})
#   %mul_131 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_366, %bmm_112), kwargs = {})
#   %permute_44 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_131, [0, 2, 1]), kwargs = {})
#   %slice_23 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg7_1, 1, 2048, 3072), kwargs = {})
#   %mul_138 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_44, %slice_23), kwargs = {})
#   %convert_element_type_372 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_138, torch.bfloat16), kwargs = {})
#   %mul_134 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_113, %bmm_112), kwargs = {})
#   %sigmoid_11 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_111,), kwargs = {})
#   %mul_135 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %sigmoid_11), kwargs = {})
#   %sub_2 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_11), kwargs = {})
#   %mul_136 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_111, %sub_2), kwargs = {})
#   %add_86 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, 1), kwargs = {})
#   %mul_137 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_135, %add_86), kwargs = {})
#   return %convert_element_type_372,%mul_137
triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_42 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 113246208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1024)
    x2 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (6144 + 3*x0 + 12288*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11 * tmp5
    tmp13 = tl.sigmoid(tmp0)
    tmp14 = tmp12 * tmp13
    tmp15 = 1.0
    tmp16 = tmp15 - tmp13
    tmp17 = tmp0 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/lu/clurafxhgo67a6rjbjqylrvxtxorfj4kuo4exwyc4pljf37lr5mb.py
# Topologically Sorted Source Nodes: [m_i_4, m_i_5], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i_4 => slice_29
#   m_i_5 => mean_2
# Graph fragment:
#   %arg3_1 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %slice_29 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 2048, 3072), kwargs = {})
#   %mean_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_29, [1], True), kwargs = {})
#   return %buf333
triton_per_fused_mean_slice_43 = async_compile.triton('triton_per_fused_mean_slice_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_mean_slice_43(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/eq/cequalbvhliua73wv2xt3gefmqwu7o2fy4uaqjlywajwnwwcv6wr.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, m_i_2, m_i_3, mul_72, dw1_3, m_i_4, m_i_5, mul_133, dw1_5, X_42], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   X_42 => convert_element_type_381
#   dw1_1 => add_2
#   dw1_3 => add_45
#   dw1_5 => add_88
#   dw1_momentum => full_default
#   m_i => slice_9
#   m_i_1 => mean
#   m_i_2 => slice_19
#   m_i_3 => mean_1
#   m_i_4 => slice_29
#   m_i_5 => mean_2
#   mul_11 => mul_14
#   mul_133 => mul_142
#   mul_72 => mul_78
# Graph fragment:
#   %bmm_114 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_114]
#   %bmm_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf13 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf13]
#   %buf173 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf173]
#   %buf333 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf333]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_9 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_9, [1], True), kwargs = {})
#   %mul_14 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_14), kwargs = {})
#   %slice_19 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_19, [1], True), kwargs = {})
#   %mul_78 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %mean_1), kwargs = {})
#   %add_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_60, %mul_78), kwargs = {})
#   %slice_29 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 2048, 3072), kwargs = {})
#   %mean_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_29, [1], True), kwargs = {})
#   %mul_142 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %mean_2), kwargs = {})
#   %add_88 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_114, %mul_142), kwargs = {})
#   %convert_element_type_381 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_88, torch.bfloat16), kwargs = {})
#   return %convert_element_type_381
triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44 = async_compile.triton('triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = 1024.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 0.0
    tmp10 = tmp9 * tmp8
    tmp11 = tmp5 + tmp10
    tmp13 = (tmp12 / tmp7)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp3 + tmp14
    tmp17 = (tmp16 / tmp7)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp1 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/u7/cu7vr3k3oett6hbyhhtxupsrb4kk3ogt6rjwxxbp6a7wk2ddhkgc.py
# Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm_15 => convert_element_type_382, pow_31, sum_16
# Graph fragment:
#   %convert_element_type_381 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_381]
#   %convert_element_type_382 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_381, torch.float32), kwargs = {})
#   %pow_31 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_382, 2), kwargs = {})
#   %sum_16 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_31, [1, 2], True), kwargs = {})
#   return %buf335
triton_red_fused_linalg_vector_norm_45 = async_compile.triton('triton_red_fused_linalg_vector_norm_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 2359360}}
)
@triton.jit
def triton_red_fused_linalg_vector_norm_45(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/a4/ca4ond3cgk6jpbuxhgypt4k2ps5u7crjhjiol6rmzvxmdq2vqnza.py
# Topologically Sorted Source Nodes: [norm_15, add_90, X_43, A_30, transpose_44, matmul_92], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_30 => convert_element_type_383, convert_element_type_384
#   X_43 => div_12
#   add_90 => add_90
#   matmul_92 => convert_element_type_389
#   norm_15 => pow_32
#   transpose_44 => permute_45
# Graph fragment:
#   %convert_element_type_381 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_381]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_90 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_381, %add_90), kwargs = {})
#   %convert_element_type_384 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.bfloat16), kwargs = {})
#   %permute_45 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_12, [0, 2, 1]), kwargs = {})
#   %convert_element_type_383 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_45, torch.bfloat16), kwargs = {})
#   %convert_element_type_389 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.bfloat16), kwargs = {})
#   return %expand_180,%expand_181,%expand_185
triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16515072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = libdevice.sqrt(tmp2)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr1 + (x2), tmp7, None)
    tl.store(out_ptr2 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/mg/cmg4utzhhvsxrltyvwfrxw2gpxdlnhgk2mz2hvyqoj45uug6ez5o.py
# Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, A_31, transpose_45, matmul_95], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_31 => convert_element_type_392, convert_element_type_393
#   X_43 => div_12
#   X_44 => add_92
#   add_90 => add_90
#   matmul_95 => convert_element_type_398
#   mul_137 => mul_146
#   norm_15 => pow_32
#   transpose_45 => permute_46
# Graph fragment:
#   %convert_element_type_381 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_381]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %bmm_119 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_90 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_381, %add_90), kwargs = {})
#   %mul_146 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, 4.0848), kwargs = {})
#   %add_92 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %bmm_119), kwargs = {})
#   %convert_element_type_393 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_92, torch.bfloat16), kwargs = {})
#   %permute_46 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_92, [0, 2, 1]), kwargs = {})
#   %convert_element_type_392 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_46, torch.bfloat16), kwargs = {})
#   %convert_element_type_398 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_92, torch.bfloat16), kwargs = {})
#   return %expand_186,%expand_187,%expand_191
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = libdevice.sqrt(tmp2)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = 4.0848
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp12, None)
    tl.store(out_ptr2 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/kx/ckxjt4clz3d27yqjnirnfb3mhuzl67zxl3uozugyzci3w6npk4n2.py
# Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_140, X_45, A_32, transpose_46, matmul_98], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_32 => convert_element_type_401, convert_element_type_402
#   X_43 => div_12
#   X_44 => add_92
#   X_45 => add_94
#   add_90 => add_90
#   matmul_98 => convert_element_type_407
#   mul_137 => mul_146
#   mul_140 => mul_149
#   norm_15 => pow_32
#   transpose_46 => permute_47
# Graph fragment:
#   %convert_element_type_381 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_381]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %bmm_119 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %bmm_122 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_122]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_90 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_381, %add_90), kwargs = {})
#   %mul_146 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, 4.0848), kwargs = {})
#   %add_92 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %bmm_119), kwargs = {})
#   %mul_149 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, 3.9505), kwargs = {})
#   %add_94 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %bmm_122), kwargs = {})
#   %convert_element_type_402 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_94, torch.bfloat16), kwargs = {})
#   %permute_47 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_94, [0, 2, 1]), kwargs = {})
#   %convert_element_type_401 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_47, torch.bfloat16), kwargs = {})
#   %convert_element_type_407 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_94, torch.bfloat16), kwargs = {})
#   return %expand_192,%expand_193,%expand_197
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = libdevice.sqrt(tmp2)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
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
    tl.store(out_ptr2 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/s5/cs5amjln3eunztc6azqjvgmrbnb2jkfan7xvzxqjc5t6icmpsw7m.py
# Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_140, X_45, mul_143, X_46, A_33, transpose_47, matmul_101], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_33 => convert_element_type_410, convert_element_type_411
#   X_43 => div_12
#   X_44 => add_92
#   X_45 => add_94
#   X_46 => add_96
#   add_90 => add_90
#   matmul_101 => convert_element_type_416
#   mul_137 => mul_146
#   mul_140 => mul_149
#   mul_143 => mul_152
#   norm_15 => pow_32
#   transpose_47 => permute_48
# Graph fragment:
#   %convert_element_type_381 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_381]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %bmm_119 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %bmm_122 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_122]
#   %bmm_125 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_125]
#   %add_96 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_96]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_90 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_381, %add_90), kwargs = {})
#   %mul_146 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, 4.0848), kwargs = {})
#   %add_92 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %bmm_119), kwargs = {})
#   %mul_149 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, 3.9505), kwargs = {})
#   %add_94 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %bmm_122), kwargs = {})
#   %mul_152 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, 3.7418), kwargs = {})
#   %add_96 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_152, %bmm_125), kwargs = {})
#   %convert_element_type_411 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_96, torch.bfloat16), kwargs = {})
#   %permute_48 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_96, [0, 2, 1]), kwargs = {})
#   %convert_element_type_410 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_48, torch.bfloat16), kwargs = {})
#   %convert_element_type_416 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_96, torch.bfloat16), kwargs = {})
#   return %add_96,%expand_198,%expand_199,%expand_203
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33030144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 36864
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = libdevice.sqrt(tmp2)
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
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
    tl.store(out_ptr3 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/br/cbr7tglfe4pd5zz73ma4f2ulqxrffknwm6p4eb3zoy5hzryahgy3.py
# Topologically Sorted Source Nodes: [mul_146, X_47, A_34, transpose_48, matmul_104], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_34 => convert_element_type_419, convert_element_type_420
#   X_47 => add_98
#   matmul_104 => convert_element_type_425
#   mul_146 => mul_155
#   transpose_48 => permute_49
# Graph fragment:
#   %add_96 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_96]
#   %bmm_128 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_128]
#   %mul_155 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_96, 2.8769), kwargs = {})
#   %add_98 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %bmm_128), kwargs = {})
#   %convert_element_type_420 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_98, torch.bfloat16), kwargs = {})
#   %permute_49 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_98, [0, 2, 1]), kwargs = {})
#   %convert_element_type_419 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_49, torch.bfloat16), kwargs = {})
#   %convert_element_type_425 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_98, torch.bfloat16), kwargs = {})
#   return %expand_204,%expand_205,%expand_209
triton_poi_fused__to_copy_add_mul_transpose_50 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_50', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_50(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/yn/cynkikrdsitqhqyqfnve7h6im2owwe5g6iy73uhijp2vzak3oita.py
# Topologically Sorted Source Nodes: [w1_norm, mul_88, X_27, w1_main_1, mul_146, X_47, mul_149, X_48, w1_main_2, norm_19, add_127, truediv_16, w1_2, bmm_29], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_27 => add_57
#   X_47 => add_98
#   X_48 => add_100
#   add_127 => add_127
#   bmm_29 => convert_element_type_530
#   mul_146 => mul_155
#   mul_149 => mul_158
#   mul_88 => mul_94
#   norm_19 => pow_39, pow_40, sum_20
#   truediv_16 => div_16
#   w1_2 => mul_190
#   w1_main_1 => add_80
#   w1_main_2 => add_123
#   w1_norm => pow_4
# Graph fragment:
#   %add_37 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_37]
#   %add_55 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_55]
#   %bmm_77 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_77]
#   %add_96 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_96]
#   %bmm_128 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_128]
#   %bmm_131 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_131]
#   %add_123 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_123]
#   %sum_20 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_20]
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_2]
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mul_94 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 2.8366), kwargs = {})
#   %add_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %bmm_77), kwargs = {})
#   %add_80 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %add_57), kwargs = {})
#   %mul_155 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_96, 2.8769), kwargs = {})
#   %add_98 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %bmm_128), kwargs = {})
#   %mul_158 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_98, 2.8366), kwargs = {})
#   %add_100 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %bmm_131), kwargs = {})
#   %add_123 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %add_100), kwargs = {})
#   %pow_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_123, 2), kwargs = {})
#   %sum_20 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_39, [2], True), kwargs = {})
#   %pow_40 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_20, 0.5), kwargs = {})
#   %add_127 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_40, 1e-05), kwargs = {})
#   %div_16 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_123, %add_127), kwargs = {})
#   %mul_190 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %pow_4), kwargs = {})
#   %convert_element_type_530 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_190, torch.bfloat16), kwargs = {})
#   return %add_123,%sum_20,%convert_element_type_530
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 25952256}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 2.8366
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = 2.8769
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp13 * tmp2
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp7 + tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = (tmp18 / tmp26)
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tmp27 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/4p/c4p6jgkogi45rfbw2646apnjnv7kvdac27j5qa64au3u5ixmf3ok.py
# Topologically Sorted Source Nodes: [ki_2, lr0i_2, mul_130, type_as_7, lr2i_2, mul_131, type_as_8], Original ATen: [aten.slice, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   ki_2 => slice_20
#   lr0i_2 => slice_25
#   lr2i_2 => slice_24
#   mul_130 => mul_139
#   mul_131 => mul_140
#   type_as_7 => convert_element_type_375
#   type_as_8 => convert_element_type_378
# Graph fragment:
#   %arg6_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg8_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %slice_20 : Tensor "bf16[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 2048, 3072), kwargs = {})
#   %slice_25 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg9_1, 1, 2048, 3072), kwargs = {})
#   %mul_139 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_20, %slice_25), kwargs = {})
#   %convert_element_type_375 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_139, torch.bfloat16), kwargs = {})
#   %slice_24 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg8_1, 1, 2048, 3072), kwargs = {})
#   %mul_140 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_20, %slice_24), kwargs = {})
#   %convert_element_type_378 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_140, torch.bfloat16), kwargs = {})
#   return %convert_element_type_375,%convert_element_type_378
triton_poi_fused__to_copy_mul_slice_52 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62914560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_52(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 196608
    x3 = (xindex % 196608)
    x1 = ((xindex // 192) % 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (393216 + x3 + 786432*x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (6144 + 3*x1 + 12288*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (6144 + 3*x1 + 12288*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp4, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/o2/co2jzzpaskqzhei3gzvv74uisd4grbyemxj2arva6snec6537pon.py
# Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   output => full_3, permute_2
#   setitem => copy, slice_7
#   setitem_1 => copy_1, slice_17
#   setitem_2 => copy_2, slice_27
#   setitem_3 => copy_3, slice_32
# Graph fragment:
#   %bmm_164 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_164]
#   %bmm_110 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_110]
#   %bmm_56 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_56]
#   %bmm_2 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %full_3 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_3, [0, 2, 1]), kwargs = {})
#   %slice_7 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 1024), kwargs = {})
#   %copy : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %bmm_2), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_2, %copy, 2, 0, 1024), kwargs = {})
#   %slice_17 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 1024, 2048), kwargs = {})
#   %copy_1 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_17, %bmm_56), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 1024, 2048), kwargs = {})
#   %slice_27 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_1, 2, 2048, 3072), kwargs = {})
#   %copy_2 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_27, %bmm_110), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %copy_2, 2, 2048, 3072), kwargs = {})
#   %slice_32 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 2, 3072, 4096), kwargs = {})
#   %copy_3 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_32, %bmm_164), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %copy_3, 2, 3072, 4096), kwargs = {})
#   return %slice_scatter_default_3
triton_poi_fused_copy_slice_zeros_like_53 = async_compile.triton('triton_poi_fused_copy_slice_zeros_like_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_like_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 301989888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_like_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/nm/cnmrv3y7tzz2izlj4ksr3dttv53dkrdrnafreotdlbs5koi7t4kf.py
# Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   output => full_3, permute_2
#   setitem => copy, slice_7
#   setitem_1 => copy_1, slice_17
#   setitem_2 => copy_2, slice_27
#   setitem_3 => copy_3, slice_32
# Graph fragment:
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_3]
#   %full_3 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_3, [0, 2, 1]), kwargs = {})
#   %slice_7 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 1024), kwargs = {})
#   %copy : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_7, %bmm_2), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_2, %copy, 2, 0, 1024), kwargs = {})
#   %slice_17 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 1024, 2048), kwargs = {})
#   %copy_1 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_17, %bmm_56), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 1024, 2048), kwargs = {})
#   %slice_27 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_1, 2, 2048, 3072), kwargs = {})
#   %copy_2 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_27, %bmm_110), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %copy_2, 2, 2048, 3072), kwargs = {})
#   %slice_32 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 2, 3072, 4096), kwargs = {})
#   %copy_3 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_32, %bmm_164), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %copy_3, 2, 3072, 4096), kwargs = {})
#   %permute_61 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_scatter_default_3, [0, 2, 1]), kwargs = {})
#   return %permute_61
triton_poi_fused_copy_slice_zeros_like_54 = async_compile.triton('triton_poi_fused_copy_slice_zeros_like_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_like_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331648, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_like_54(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 192
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4096)
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4096*x2 + 786432*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp0, xmask & ymask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        assert_size_stride(arg0_1, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(arg1_1, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(arg2_1, (32, 192, 192), (36864, 192, 1))
        assert_size_stride(arg3_1, (32, 4096, 1), (4096, 1, 1))
        assert_size_stride(arg4_1, (32, 4096, 192), (786432, 192, 1))
        assert_size_stride(arg5_1, (32, 4096, 192), (786432, 192, 1))
        assert_size_stride(arg6_1, (32, 4096, 192), (786432, 192, 1))
        assert_size_stride(arg7_1, (32, 4096, 1), (12288, 3, 1))
        assert_size_stride(arg8_1, (32, 4096, 1), (12288, 3, 1))
        assert_size_stride(arg9_1, (32, 4096, 1), (12288, 3, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf7 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_1, gate_before_act], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg0_1, buf0, buf7, 1179648, stream=stream0)
            buf1 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_1, q, qi], Original ATen: [aten._to_copy, aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(buf0, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf1)
            buf2 = buf0; del buf0  # reuse
            buf9 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h, hidden_before_mul], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg2_1, buf2, buf9, 1179648, stream=stream0)
            buf3 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi, h], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf2, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf3)
            buf10 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki, hidden_before_mul, transpose_3], Original ATen: [aten.slice, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf9, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf10)
            buf59 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf60 = reinterpret_tensor(buf9, (32, 192, 192), (36864, 1, 192), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, transpose_4, dhidden], Original ATen: [aten.linalg_vector_norm, aten.transpose, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_transpose_1.run(arg1_1, buf59, buf60, 6144, 192, stream=stream0)
            buf61 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi, transpose_4, dhidden], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf60, reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf61)
            buf8 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_before_act, ki, transpose_2], Original ATen: [aten._to_copy, aten.slice, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf7, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf8)
            buf11 = empty_strided_cuda((32, 1024, 192), (196608, 1, 1024), torch.bfloat16)
            buf62 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_1, hidden, transpose_5, lr1i, mul_7, type_as, dgate, sigma, mul_4, sub, mul_5, add, dx], Original ATen: [aten.silu, aten.mul, aten.transpose, aten.slice, aten._to_copy, aten.sigmoid, aten.rsub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_2.run(buf8, buf10, arg7_1, buf61, buf11, buf62, 6291456, stream=stream0)
            buf12 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [v, vi, silu_1, hidden, transpose_5, lr1i, mul_7, type_as, dw1], Original ATen: [aten.transpose, aten.slice, aten.silu, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 0), buf11, out=buf12)
            buf13 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_3.run(arg3_1, buf13, 32, 1024, stream=stream0)
            buf63 = reinterpret_tensor(buf11, (32, 1024, 192), (196608, 192, 1), 0); del buf11  # reuse
            buf114 = reinterpret_tensor(buf10, (32, 1024, 192), (196608, 192, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [ki, lr0i, mul_8, type_as_1, lr2i, mul_9, type_as_2], Original ATen: [aten.slice, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_4.run(arg6_1, arg9_1, arg8_1, buf63, buf114, 6291456, stream=stream0)
            buf64 = reinterpret_tensor(buf60, (32, 192, 192), (36864, 192, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [ki, dgate, sigma, mul_4, sub, mul_5, add, dx, lr0i, mul_8, type_as_1, dw0], Original ATen: [aten.slice, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf62, buf63, out=buf64)
            buf14 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            buf65 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, dw0_momentum, mul_10, dw0_1, X_7, norm_4], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf12, buf13, buf64, buf14, buf65, 160, 7373, stream=stream0)
            buf15 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf14, buf15, 32, 5, stream=stream0)
            buf16 = buf2; del buf2  # reuse
            buf17 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf22 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, A, transpose_6, matmul_2], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_7.run(buf12, buf13, buf15, buf16, buf17, buf22, 1179648, stream=stream0)
            buf18 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, A, transpose_6], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf16, buf17, out=buf18)
            buf19 = reinterpret_tensor(buf17, (32, 192, 192), (36864, 192, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [mul_14], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf18, buf19, 1179648, stream=stream0)
            buf20 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [mul_14, matmul_1], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf19, buf18, out=buf20)
            buf21 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [mul_13, B], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf21, buf20, 1179648, stream=stream0)
            buf23 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_13, B, matmul_2], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf21, buf22, out=buf23)
            buf24 = buf22; del buf22  # reuse
            buf25 = reinterpret_tensor(buf21, (32, 192, 192), (36864, 1, 192), 0); del buf21  # reuse
            buf30 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_15, X_2, A_1, transpose_7, matmul_5], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10.run(buf12, buf13, buf15, buf23, buf24, buf25, buf30, 1179648, stream=stream0)
            buf26 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_15, X_2, A_1, transpose_7], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf24, buf25, out=buf26)
            buf27 = reinterpret_tensor(buf25, (32, 192, 192), (36864, 192, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [mul_17], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf26, buf27, 1179648, stream=stream0)
            buf28 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [mul_17, matmul_4], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf27, buf26, out=buf28)
            buf29 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [mul_16, B_1], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf29, buf28, 1179648, stream=stream0)
            buf31 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_15, X_2, mul_16, B_1, matmul_5], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf29, buf30, out=buf31)
            buf66 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf65, buf66, 32, 5, stream=stream0)
            buf32 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf33 = buf30; del buf30  # reuse
            buf34 = reinterpret_tensor(buf29, (32, 192, 192), (36864, 1, 192), 0); del buf29  # reuse
            buf39 = buf27; del buf27  # reuse
            buf67 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf68 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf73 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, X, norm_3, add_4, X_1, mul_15, X_2, mul_18, X_3, A_2, transpose_8, matmul_8, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, A_5, transpose_11, matmul_17], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_13.run(buf12, buf13, buf15, buf23, buf31, buf64, buf66, buf32, buf33, buf34, buf39, buf67, buf68, buf73, 1179648, stream=stream0)
            buf35 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [A_2, transpose_8], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf33, buf34, out=buf35)
            buf36 = reinterpret_tensor(buf34, (32, 192, 192), (36864, 192, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [mul_20], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf35, buf36, 1179648, stream=stream0)
            buf37 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [mul_20, matmul_7], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf36, buf35, out=buf37)
            buf38 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [mul_19, B_2], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf38, buf37, 1179648, stream=stream0)
            buf40 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [mul_19, B_2, matmul_8], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf38, buf39, out=buf40)
            buf41 = buf39; del buf39  # reuse
            buf42 = reinterpret_tensor(buf38, (32, 192, 192), (36864, 1, 192), 0); del buf38  # reuse
            buf47 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [mul_21, X_4, A_3, transpose_9, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_16.run(buf32, buf40, buf41, buf42, buf47, 1179648, stream=stream0)
            buf43 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [mul_21, X_4, A_3, transpose_9], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf41, buf42, out=buf43)
            buf44 = reinterpret_tensor(buf42, (32, 192, 192), (36864, 192, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [mul_23], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf43, buf44, 1179648, stream=stream0)
            buf45 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [mul_23, matmul_10], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf44, buf43, out=buf45)
            buf46 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [mul_22, B_3], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf46, buf45, 1179648, stream=stream0)
            buf48 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [mul_21, X_4, mul_22, B_3, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf46, buf47, out=buf48)
            buf49 = buf47; del buf47  # reuse
            buf50 = reinterpret_tensor(buf46, (32, 192, 192), (36864, 1, 192), 0); del buf46  # reuse
            buf55 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, A_4, transpose_10, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_19.run(buf32, buf40, buf48, buf49, buf50, buf55, 1179648, stream=stream0)
            buf51 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, A_4, transpose_10], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf49, buf50, out=buf51)
            buf52 = reinterpret_tensor(buf50, (32, 192, 192), (36864, 192, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [mul_26], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf51, buf52, 1179648, stream=stream0)
            buf53 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [mul_26, matmul_13], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf52, buf51, out=buf53)
            buf54 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [mul_25, B_4], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf54, buf53, 1179648, stream=stream0)
            buf56 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [mul_21, X_4, mul_24, X_5, mul_25, B_4, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf54, buf55, out=buf56)
            buf4 = buf55; del buf55  # reuse
            buf57 = buf32; del buf32  # reuse
            buf164 = buf54; del buf54  # reuse
            buf220 = reinterpret_tensor(buf52, (32, 192, 192), (36864, 1, 192), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [bmm_2, mul_21, X_4, mul_24, X_5, mul_27, X_6, w1_main, norm_7, add_41, truediv_4, w1_norm, w1, bmm_11, transpose_23, dhidden_1], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_22.run(buf57, arg1_1, buf40, buf48, buf56, buf59, buf4, buf164, buf220, 6144, 192, stream=stream0)
            del arg1_1
            buf5 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [gate, mul], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_23.run(buf5, buf3, 6291456, stream=stream0)
            buf6 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [bmm_2, gate, mul], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.bmm]
            extern_kernels.bmm(buf4, buf5, out=buf6)
            buf69 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, A_5, transpose_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf67, buf68, out=buf69)
            buf70 = reinterpret_tensor(buf68, (32, 192, 192), (36864, 192, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [mul_29], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf69, buf70, 1179648, stream=stream0)
            buf71 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [mul_29, matmul_16], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf70, buf69, out=buf71)
            buf72 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [mul_28, B_5], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf72, buf71, 1179648, stream=stream0)
            buf74 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, mul_28, B_5, matmul_17], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf72, buf73, out=buf74)
            buf75 = buf73; del buf73  # reuse
            buf76 = reinterpret_tensor(buf72, (32, 192, 192), (36864, 1, 192), 0); del buf72  # reuse
            buf81 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, mul_30, X_9, A_6, transpose_12, matmul_20], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10.run(buf64, buf13, buf66, buf74, buf75, buf76, buf81, 1179648, stream=stream0)
            buf77 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, mul_30, X_9, A_6, transpose_12], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf75, buf76, out=buf77)
            buf78 = reinterpret_tensor(buf76, (32, 192, 192), (36864, 192, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [mul_32], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf77, buf78, 1179648, stream=stream0)
            buf79 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [mul_32, matmul_19], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf78, buf77, out=buf79)
            buf80 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [mul_31, B_6], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf80, buf79, 1179648, stream=stream0)
            buf82 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, mul_30, X_9, mul_31, B_6, matmul_20], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf80, buf81, out=buf82)
            buf113 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [silu_2, dhidden_before_mul], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_24.run(buf113, buf8, 6291456, stream=stream0)
            buf115 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [ki, silu_2, dhidden_before_mul, lr2i, mul_9, type_as_2, dw2], Original ATen: [aten.slice, aten.silu, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf113, buf114, out=buf115)
            buf116 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_25.run(buf115, buf13, buf116, 160, 7373, stream=stream0)
            buf117 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf116, buf117, 32, 5, stream=stream0)
            buf83 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf84 = buf80; del buf80  # reuse
            buf85 = reinterpret_tensor(buf78, (32, 192, 192), (36864, 1, 192), 0); del buf78  # reuse
            buf90 = buf48; del buf48  # reuse
            buf118 = buf40; del buf40  # reuse
            buf119 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf124 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, X_7, norm_4, add_15, X_8, mul_30, X_9, mul_33, X_10, A_7, transpose_13, matmul_23, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, A_10, transpose_16, matmul_32], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_13.run(buf64, buf13, buf66, buf74, buf82, buf115, buf117, buf83, buf84, buf85, buf90, buf118, buf119, buf124, 1179648, stream=stream0)
            buf86 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [A_7, transpose_13], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf84, buf85, out=buf86)
            buf87 = reinterpret_tensor(buf85, (32, 192, 192), (36864, 192, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [mul_35], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf86, buf87, 1179648, stream=stream0)
            buf88 = buf84; del buf84  # reuse
            # Topologically Sorted Source Nodes: [mul_35, matmul_22], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf87, buf86, out=buf88)
            buf89 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [mul_34, B_7], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf89, buf88, 1179648, stream=stream0)
            buf91 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [mul_34, B_7, matmul_23], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf89, buf90, out=buf91)
            buf92 = buf90; del buf90  # reuse
            buf93 = reinterpret_tensor(buf89, (32, 192, 192), (36864, 1, 192), 0); del buf89  # reuse
            buf98 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [mul_36, X_11, A_8, transpose_14, matmul_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_16.run(buf83, buf91, buf92, buf93, buf98, 1179648, stream=stream0)
            buf94 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [mul_36, X_11, A_8, transpose_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf92, buf93, out=buf94)
            buf95 = reinterpret_tensor(buf93, (32, 192, 192), (36864, 192, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [mul_38], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf94, buf95, 1179648, stream=stream0)
            buf96 = buf92; del buf92  # reuse
            # Topologically Sorted Source Nodes: [mul_38, matmul_25], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf95, buf94, out=buf96)
            buf97 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [mul_37, B_8], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf97, buf96, 1179648, stream=stream0)
            buf99 = buf96; del buf96  # reuse
            # Topologically Sorted Source Nodes: [mul_36, X_11, mul_37, B_8, matmul_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf97, buf98, out=buf99)
            buf100 = buf98; del buf98  # reuse
            buf101 = reinterpret_tensor(buf97, (32, 192, 192), (36864, 1, 192), 0); del buf97  # reuse
            buf106 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, A_9, transpose_15, matmul_29], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_19.run(buf83, buf91, buf99, buf100, buf101, buf106, 1179648, stream=stream0)
            buf102 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, A_9, transpose_15], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf100, buf101, out=buf102)
            buf103 = reinterpret_tensor(buf101, (32, 192, 192), (36864, 192, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [mul_41], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf102, buf103, 1179648, stream=stream0)
            buf104 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [mul_41, matmul_28], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf103, buf102, out=buf104)
            buf105 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [mul_40, B_9], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf105, buf104, 1179648, stream=stream0)
            buf107 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, mul_40, B_9, matmul_29], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf105, buf106, out=buf107)
            buf110 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf108 = buf83; del buf83  # reuse
            buf111 = buf106; del buf106  # reuse
            buf167 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [mul_36, X_11, mul_39, X_12, mul_42, X_13, w0_main, norm_6, add_40, truediv_3, w0_norm, w0, bmm_10, gate_before_act_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_26.run(buf108, arg0_1, buf91, buf99, buf107, buf110, buf111, buf167, 6144, 192, stream=stream0)
            del arg0_1
            buf112 = reinterpret_tensor(buf114, (32, 192, 1024), (196608, 1024, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [q, norm_6, add_40, truediv_3, w0_norm, w0, bmm_10, qi_1], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.slice, aten.bmm]
            extern_kernels.bmm(buf111, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf112)
            buf120 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, A_10, transpose_16], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf118, buf119, out=buf120)
            buf121 = reinterpret_tensor(buf119, (32, 192, 192), (36864, 192, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [mul_44], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf120, buf121, 1179648, stream=stream0)
            buf122 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [mul_44, matmul_31], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf121, buf120, out=buf122)
            buf123 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [mul_43, B_10], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf123, buf122, 1179648, stream=stream0)
            buf125 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, mul_43, B_10, matmul_32], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf123, buf124, out=buf125)
            buf126 = buf124; del buf124  # reuse
            buf127 = reinterpret_tensor(buf123, (32, 192, 192), (36864, 1, 192), 0); del buf123  # reuse
            buf132 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, mul_45, X_16, A_11, transpose_17, matmul_35], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_10.run(buf115, buf13, buf117, buf125, buf126, buf127, buf132, 1179648, stream=stream0)
            buf128 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, mul_45, X_16, A_11, transpose_17], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf126, buf127, out=buf128)
            buf129 = reinterpret_tensor(buf127, (32, 192, 192), (36864, 192, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [mul_47], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf128, buf129, 1179648, stream=stream0)
            buf130 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [mul_47, matmul_34], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf129, buf128, out=buf130)
            buf131 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [mul_46, B_11], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf131, buf130, 1179648, stream=stream0)
            buf133 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, mul_45, X_16, mul_46, B_11, matmul_35], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf131, buf132, out=buf133)
            buf134 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf135 = buf132; del buf132  # reuse
            buf136 = reinterpret_tensor(buf131, (32, 192, 192), (36864, 1, 192), 0); del buf131  # reuse
            buf141 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, X_14, norm_5, add_26, X_15, mul_45, X_16, mul_48, X_17, A_12, transpose_18, matmul_38], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27.run(buf115, buf13, buf117, buf125, buf133, buf134, buf135, buf136, buf141, 1179648, stream=stream0)
            buf137 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [A_12, transpose_18], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf135, buf136, out=buf137)
            buf138 = reinterpret_tensor(buf136, (32, 192, 192), (36864, 192, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [mul_50], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf137, buf138, 1179648, stream=stream0)
            buf139 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [mul_50, matmul_37], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf138, buf137, out=buf139)
            buf140 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [mul_49, B_12], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf140, buf139, 1179648, stream=stream0)
            buf142 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [mul_49, B_12, matmul_38], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf140, buf141, out=buf142)
            buf143 = buf141; del buf141  # reuse
            buf144 = reinterpret_tensor(buf140, (32, 192, 192), (36864, 1, 192), 0); del buf140  # reuse
            buf149 = buf138; del buf138  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, A_13, transpose_19, matmul_41], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_16.run(buf134, buf142, buf143, buf144, buf149, 1179648, stream=stream0)
            buf145 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, A_13, transpose_19], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf143, buf144, out=buf145)
            buf146 = reinterpret_tensor(buf144, (32, 192, 192), (36864, 192, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [mul_53], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf145, buf146, 1179648, stream=stream0)
            buf147 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [mul_53, matmul_40], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf146, buf145, out=buf147)
            buf148 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [mul_52, B_13], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf148, buf147, 1179648, stream=stream0)
            buf150 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, mul_52, B_13, matmul_41], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf148, buf149, out=buf150)
            buf151 = buf149; del buf149  # reuse
            buf152 = reinterpret_tensor(buf148, (32, 192, 192), (36864, 1, 192), 0); del buf148  # reuse
            buf157 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, mul_54, X_19, A_14, transpose_20, matmul_44], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_19.run(buf134, buf142, buf150, buf151, buf152, buf157, 1179648, stream=stream0)
            buf153 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, mul_54, X_19, A_14, transpose_20], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf151, buf152, out=buf153)
            buf154 = reinterpret_tensor(buf152, (32, 192, 192), (36864, 192, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [mul_56], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf153, buf154, 1179648, stream=stream0)
            buf155 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [mul_56, matmul_43], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf154, buf153, out=buf155)
            buf156 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [mul_55, B_14], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf156, buf155, 1179648, stream=stream0)
            buf158 = buf155; del buf155  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, mul_54, X_19, mul_55, B_14, matmul_44], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf156, buf157, out=buf158)
            buf173 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [m_i_2, m_i_3], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_28.run(arg3_1, buf173, 32, 1024, stream=stream0)
            buf168 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [norm_6, add_40, truediv_3, w0_norm, w0, gate_before_act_1, ki_1, transpose_21], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.slice, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf167, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf168)
            buf221 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [v, norm_7, add_41, truediv_4, w1_norm, w1, vi_1, transpose_23, dhidden_1], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf220, reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf221)
            buf273 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [silu_5, dhidden_before_mul_1], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_29.run(buf221, buf168, buf273, 6291456, stream=stream0)
            buf223 = buf63; del buf63  # reuse
            buf274 = reinterpret_tensor(buf62, (32, 1024, 192), (196608, 192, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [ki_1, lr0i_1, mul_69, type_as_4, lr2i_1, mul_70, type_as_5], Original ATen: [aten.slice, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_30.run(arg6_1, arg9_1, arg8_1, buf223, buf274, 6291456, stream=stream0)
            buf275 = reinterpret_tensor(buf220, (32, 192, 192), (36864, 192, 1), 0); del buf220  # reuse
            # Topologically Sorted Source Nodes: [ki_1, silu_5, dhidden_before_mul_1, lr2i_1, mul_70, type_as_5, dw2_2], Original ATen: [aten.slice, aten.silu, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf273, buf274, out=buf275)
            buf276 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, m_i_2, m_i_3, mul_73, dw2_3, X_35, norm_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_31.run(buf275, buf115, buf13, buf173, buf276, 160, 7373, stream=stream0)
            buf277 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, m_i_2, m_i_3, mul_73, dw2_3, X_35, norm_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf276, buf277, 32, 5, stream=stream0)
            buf278 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf279 = buf167; del buf167  # reuse
            buf280 = reinterpret_tensor(buf157, (32, 192, 192), (36864, 1, 192), 0); del buf157  # reuse
            buf285 = buf156; del buf156  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, m_i_2, m_i_3, mul_73, dw2_3, X_35, norm_11, add_69, X_36, A_25, transpose_35, matmul_77], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_32.run(buf275, buf115, buf13, buf173, buf277, buf278, buf279, buf280, buf285, 1179648, stream=stream0)
            buf281 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [A_25, transpose_35], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf279, buf280, out=buf281)
            buf282 = reinterpret_tensor(buf280, (32, 192, 192), (36864, 192, 1), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [mul_105], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf281, buf282, 1179648, stream=stream0)
            buf283 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [mul_105, matmul_76], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf282, buf281, out=buf283)
            buf284 = buf281; del buf281  # reuse
            # Topologically Sorted Source Nodes: [mul_104, B_25], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf284, buf283, 1179648, stream=stream0)
            buf286 = buf283; del buf283  # reuse
            # Topologically Sorted Source Nodes: [mul_104, B_25, matmul_77], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf284, buf285, out=buf286)
            buf287 = buf285; del buf285  # reuse
            buf288 = reinterpret_tensor(buf284, (32, 192, 192), (36864, 1, 192), 0); del buf284  # reuse
            buf293 = buf282; del buf282  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, A_26, transpose_36, matmul_80], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_33.run(buf278, buf286, buf287, buf288, buf293, 1179648, stream=stream0)
            buf289 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, A_26, transpose_36], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf287, buf288, out=buf289)
            buf290 = reinterpret_tensor(buf288, (32, 192, 192), (36864, 192, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [mul_108], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf289, buf290, 1179648, stream=stream0)
            buf291 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [mul_108, matmul_79], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf290, buf289, out=buf291)
            buf292 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [mul_107, B_26], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf292, buf291, 1179648, stream=stream0)
            buf294 = buf291; del buf291  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_107, B_26, matmul_80], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf292, buf293, out=buf294)
            buf295 = buf293; del buf293  # reuse
            buf296 = reinterpret_tensor(buf292, (32, 192, 192), (36864, 1, 192), 0); del buf292  # reuse
            buf301 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, A_27, transpose_37, matmul_83], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_34.run(buf278, buf286, buf294, buf295, buf296, buf301, 1179648, stream=stream0)
            buf297 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, A_27, transpose_37], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf295, buf296, out=buf297)
            buf298 = reinterpret_tensor(buf296, (32, 192, 192), (36864, 192, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [mul_111], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf297, buf298, 1179648, stream=stream0)
            buf299 = buf295; del buf295  # reuse
            # Topologically Sorted Source Nodes: [mul_111, matmul_82], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf298, buf297, out=buf299)
            buf300 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [mul_110, B_27], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf300, buf299, 1179648, stream=stream0)
            buf302 = buf299; del buf299  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_110, B_27, matmul_83], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf300, buf301, out=buf302)
            buf303 = buf301; del buf301  # reuse
            buf304 = reinterpret_tensor(buf300, (32, 192, 192), (36864, 1, 192), 0); del buf300  # reuse
            buf309 = buf298; del buf298  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_112, X_39, A_28, transpose_38, matmul_86], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_35.run(buf278, buf286, buf294, buf302, buf303, buf304, buf309, 1179648, stream=stream0)
            buf305 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_112, X_39, A_28, transpose_38], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf303, buf304, out=buf305)
            buf306 = reinterpret_tensor(buf304, (32, 192, 192), (36864, 192, 1), 0); del buf304  # reuse
            # Topologically Sorted Source Nodes: [mul_114], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf305, buf306, 1179648, stream=stream0)
            buf307 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [mul_114, matmul_85], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf306, buf305, out=buf307)
            buf308 = buf305; del buf305  # reuse
            # Topologically Sorted Source Nodes: [mul_113, B_28], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf308, buf307, 1179648, stream=stream0)
            buf310 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_112, X_39, mul_113, B_28, matmul_86], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf308, buf309, out=buf310)
            buf311 = buf278; del buf278  # reuse
            buf312 = buf309; del buf309  # reuse
            buf313 = reinterpret_tensor(buf308, (32, 192, 192), (36864, 1, 192), 0); del buf308  # reuse
            buf318 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [mul_106, X_37, mul_109, X_38, mul_112, X_39, mul_115, X_40, A_29, transpose_39, matmul_89], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_36.run(buf311, buf286, buf294, buf302, buf310, buf312, buf313, buf318, 1179648, stream=stream0)
            del buf286
            del buf294
            buf314 = buf310; del buf310  # reuse
            # Topologically Sorted Source Nodes: [A_29, transpose_39], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf312, buf313, out=buf314)
            buf315 = reinterpret_tensor(buf313, (32, 192, 192), (36864, 192, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [mul_117], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf314, buf315, 1179648, stream=stream0)
            buf316 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [mul_117, matmul_88], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf315, buf314, out=buf316)
            buf317 = buf314; del buf314  # reuse
            # Topologically Sorted Source Nodes: [mul_116, B_29], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf317, buf316, 1179648, stream=stream0)
            buf319 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [mul_116, B_29, matmul_89], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf317, buf318, out=buf319)
            buf161 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf159 = buf134; del buf134  # reuse
            buf162 = buf318; del buf318  # reuse
            buf169 = buf317; del buf317  # reuse
            buf322 = buf315; del buf315  # reuse
            buf329 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [mul_51, X_18, mul_54, X_19, mul_57, X_20, w2_main, norm_8, add_42, truediv_5, w2_norm, w2, h_1, hidden_before_mul_1, mul_118, X_41, w2_main_1, norm_14, add_85, truediv_11, w2_1, h_2, hidden_before_mul_2], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_37.run(buf159, arg2_1, buf142, buf150, buf158, buf311, buf319, buf161, buf162, buf169, buf322, buf329, 6144, 192, stream=stream0)
            del arg2_1
            del buf142
            buf163 = reinterpret_tensor(buf274, (32, 192, 1024), (196608, 1024, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [q, qi_1, norm_8, add_42, truediv_5, w2_norm, w2, h_1], Original ATen: [aten.transpose, aten.slice, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf162, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf163)
            buf165 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [gate_1, mul_61], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_23.run(buf165, buf163, 6291456, stream=stream0)
            buf166 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [norm_7, add_41, truediv_4, w1_norm, w1, bmm_11, gate_1, mul_61], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.silu, aten.bmm]
            extern_kernels.bmm(buf164, buf165, out=buf166)
            buf170 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [norm_8, add_42, truediv_5, w2_norm, w2, ki_1, hidden_before_mul_1, transpose_22], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.slice, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf169, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf170)
            buf171 = reinterpret_tensor(buf273, (32, 1024, 192), (196608, 1, 1024), 0); del buf273  # reuse
            buf222 = buf221; del buf221  # reuse
            # Topologically Sorted Source Nodes: [silu_4, hidden_1, transpose_24, lr1i_1, mul_68, type_as_3, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43, dx_1], Original ATen: [aten.silu, aten.mul, aten.transpose, aten.slice, aten._to_copy, aten.sigmoid, aten.rsub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_38.run(buf222, buf168, buf170, arg7_1, buf171, 6291456, stream=stream0)
            del buf168
            del buf170
            buf172 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [v, vi_1, silu_4, hidden_1, transpose_24, lr1i_1, mul_68, type_as_3, dw1_2], Original ATen: [aten.transpose, aten.slice, aten.silu, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 196608), buf171, out=buf172)
            buf224 = buf164; del buf164  # reuse
            # Topologically Sorted Source Nodes: [ki_1, dgate_1, sigma_1, mul_65, sub_1, mul_66, add_43, dx_1, lr0i_1, mul_69, type_as_4, dw0_2], Original ATen: [aten.slice, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf222, buf223, out=buf224)
            buf174 = buf276; del buf276  # reuse
            buf225 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, dw0_momentum, mul_10, dw0_1, m_i_2, m_i_3, mul_72, dw1_3, X_21, norm_9, mul_71, dw0_3, X_28, norm_10], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_39.run(buf172, buf12, buf13, buf173, buf224, buf64, buf174, buf225, 160, 7373, stream=stream0)
            buf175 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, m_i_2, m_i_3, mul_72, dw1_3, X_21, norm_9], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf174, buf175, 32, 5, stream=stream0)
            del buf174
            buf226 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, m_i_2, m_i_3, mul_71, dw0_3, X_28, norm_10], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf225, buf226, 32, 5, stream=stream0)
            del buf225
            buf176 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf177 = buf162; del buf162  # reuse
            buf178 = reinterpret_tensor(buf158, (32, 192, 192), (36864, 1, 192), 0); del buf158  # reuse
            buf183 = buf150; del buf150  # reuse
            buf227 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf228 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf229 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf234 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, dw0_momentum, mul_10, dw0_1, m_i_2, m_i_3, mul_72, dw1_3, X_21, norm_9, add_47, X_22, A_15, transpose_25, matmul_47, mul_71, dw0_3, X_28, norm_10, add_58, X_29, A_20, transpose_30, matmul_62], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_40.run(buf172, buf12, buf13, buf173, buf175, buf224, buf64, buf226, buf176, buf177, buf178, buf183, buf227, buf228, buf229, buf234, 1179648, stream=stream0)
            del buf175
            del buf226
            buf179 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_15, transpose_25], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf177, buf178, out=buf179)
            buf180 = reinterpret_tensor(buf178, (32, 192, 192), (36864, 192, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [mul_75], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf179, buf180, 1179648, stream=stream0)
            buf181 = buf177; del buf177  # reuse
            # Topologically Sorted Source Nodes: [mul_75, matmul_46], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf180, buf179, out=buf181)
            buf182 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [mul_74, B_15], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf182, buf181, 1179648, stream=stream0)
            buf184 = buf181; del buf181  # reuse
            # Topologically Sorted Source Nodes: [mul_74, B_15, matmul_47], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf182, buf183, out=buf184)
            buf185 = buf183; del buf183  # reuse
            buf186 = reinterpret_tensor(buf182, (32, 192, 192), (36864, 1, 192), 0); del buf182  # reuse
            buf191 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, A_16, transpose_26, matmul_50], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_33.run(buf176, buf184, buf185, buf186, buf191, 1179648, stream=stream0)
            buf187 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_76, X_23, A_16, transpose_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf185, buf186, out=buf187)
            buf188 = reinterpret_tensor(buf186, (32, 192, 192), (36864, 192, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [mul_78], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf187, buf188, 1179648, stream=stream0)
            buf189 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [mul_78, matmul_49], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf188, buf187, out=buf189)
            buf190 = buf187; del buf187  # reuse
            # Topologically Sorted Source Nodes: [mul_77, B_16], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf190, buf189, 1179648, stream=stream0)
            buf192 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_77, B_16, matmul_50], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf190, buf191, out=buf192)
            buf193 = buf191; del buf191  # reuse
            buf194 = reinterpret_tensor(buf190, (32, 192, 192), (36864, 1, 192), 0); del buf190  # reuse
            buf199 = buf188; del buf188  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, A_17, transpose_27, matmul_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_34.run(buf176, buf184, buf192, buf193, buf194, buf199, 1179648, stream=stream0)
            buf195 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, A_17, transpose_27], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf193, buf194, out=buf195)
            buf196 = reinterpret_tensor(buf194, (32, 192, 192), (36864, 192, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [mul_81], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf195, buf196, 1179648, stream=stream0)
            buf197 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [mul_81, matmul_52], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf196, buf195, out=buf197)
            buf198 = buf195; del buf195  # reuse
            # Topologically Sorted Source Nodes: [mul_80, B_17], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf198, buf197, 1179648, stream=stream0)
            buf200 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, mul_80, B_17, matmul_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf198, buf199, out=buf200)
            buf201 = buf199; del buf199  # reuse
            buf202 = reinterpret_tensor(buf198, (32, 192, 192), (36864, 1, 192), 0); del buf198  # reuse
            buf207 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, mul_82, X_25, A_18, transpose_28, matmul_56], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_35.run(buf176, buf184, buf192, buf200, buf201, buf202, buf207, 1179648, stream=stream0)
            buf203 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, mul_82, X_25, A_18, transpose_28], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf201, buf202, out=buf203)
            buf204 = reinterpret_tensor(buf202, (32, 192, 192), (36864, 192, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [mul_84], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf203, buf204, 1179648, stream=stream0)
            buf205 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [mul_84, matmul_55], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf204, buf203, out=buf205)
            buf206 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [mul_83, B_18], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf206, buf205, 1179648, stream=stream0)
            buf208 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, mul_82, X_25, mul_83, B_18, matmul_56], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf206, buf207, out=buf208)
            buf209 = buf176; del buf176  # reuse
            buf210 = buf207; del buf207  # reuse
            buf211 = reinterpret_tensor(buf206, (32, 192, 192), (36864, 1, 192), 0); del buf206  # reuse
            buf216 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [mul_76, X_23, mul_79, X_24, mul_82, X_25, mul_85, X_26, A_19, transpose_29, matmul_59], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_36.run(buf209, buf184, buf192, buf200, buf208, buf210, buf211, buf216, 1179648, stream=stream0)
            buf212 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [A_19, transpose_29], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf210, buf211, out=buf212)
            buf213 = reinterpret_tensor(buf211, (32, 192, 192), (36864, 192, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [mul_87], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf212, buf213, 1179648, stream=stream0)
            buf214 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [mul_87, matmul_58], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf213, buf212, out=buf214)
            buf215 = buf212; del buf212  # reuse
            # Topologically Sorted Source Nodes: [mul_86, B_19], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf215, buf214, 1179648, stream=stream0)
            buf217 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [mul_86, B_19, matmul_59], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf215, buf216, out=buf217)
            buf324 = buf216; del buf216  # reuse
            buf380 = reinterpret_tensor(buf215, (32, 192, 192), (36864, 1, 192), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, mul_88, X_27, w1_main_1, norm_13, add_84, truediv_10, w1_1, bmm_20, transpose_42, dhidden_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_41.run(buf57, buf209, buf217, buf59, buf324, buf380, 6144, 192, stream=stream0)
            buf230 = buf213; del buf213  # reuse
            # Topologically Sorted Source Nodes: [A_20, transpose_30], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf228, buf229, out=buf230)
            buf231 = reinterpret_tensor(buf229, (32, 192, 192), (36864, 192, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [mul_90], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf230, buf231, 1179648, stream=stream0)
            buf232 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [mul_90, matmul_61], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf231, buf230, out=buf232)
            buf233 = buf230; del buf230  # reuse
            # Topologically Sorted Source Nodes: [mul_89, B_20], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf233, buf232, 1179648, stream=stream0)
            buf235 = buf232; del buf232  # reuse
            # Topologically Sorted Source Nodes: [mul_89, B_20, matmul_62], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf233, buf234, out=buf235)
            buf236 = buf234; del buf234  # reuse
            buf237 = reinterpret_tensor(buf233, (32, 192, 192), (36864, 1, 192), 0); del buf233  # reuse
            buf242 = buf231; del buf231  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, A_21, transpose_31, matmul_65], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_33.run(buf227, buf235, buf236, buf237, buf242, 1179648, stream=stream0)
            buf238 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, A_21, transpose_31], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf236, buf237, out=buf238)
            buf239 = reinterpret_tensor(buf237, (32, 192, 192), (36864, 192, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [mul_93], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf238, buf239, 1179648, stream=stream0)
            buf240 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [mul_93, matmul_64], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf239, buf238, out=buf240)
            buf241 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [mul_92, B_21], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf241, buf240, 1179648, stream=stream0)
            buf243 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_92, B_21, matmul_65], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf241, buf242, out=buf243)
            buf244 = buf242; del buf242  # reuse
            buf245 = reinterpret_tensor(buf241, (32, 192, 192), (36864, 1, 192), 0); del buf241  # reuse
            buf250 = buf239; del buf239  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, A_22, transpose_32, matmul_68], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_34.run(buf227, buf235, buf243, buf244, buf245, buf250, 1179648, stream=stream0)
            buf246 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, A_22, transpose_32], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf244, buf245, out=buf246)
            buf247 = reinterpret_tensor(buf245, (32, 192, 192), (36864, 192, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [mul_96], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf246, buf247, 1179648, stream=stream0)
            buf248 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [mul_96, matmul_67], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf247, buf246, out=buf248)
            buf249 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [mul_95, B_22], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf249, buf248, 1179648, stream=stream0)
            buf251 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, mul_95, B_22, matmul_68], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf249, buf250, out=buf251)
            buf252 = buf250; del buf250  # reuse
            buf253 = reinterpret_tensor(buf249, (32, 192, 192), (36864, 1, 192), 0); del buf249  # reuse
            buf258 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, mul_97, X_32, A_23, transpose_33, matmul_71], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_35.run(buf227, buf235, buf243, buf251, buf252, buf253, buf258, 1179648, stream=stream0)
            buf254 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, mul_97, X_32, A_23, transpose_33], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf252, buf253, out=buf254)
            buf255 = reinterpret_tensor(buf253, (32, 192, 192), (36864, 192, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [mul_99], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf254, buf255, 1179648, stream=stream0)
            buf256 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [mul_99, matmul_70], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf255, buf254, out=buf256)
            buf257 = buf254; del buf254  # reuse
            # Topologically Sorted Source Nodes: [mul_98, B_23], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf257, buf256, 1179648, stream=stream0)
            buf259 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, mul_97, X_32, mul_98, B_23, matmul_71], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf257, buf258, out=buf259)
            buf260 = buf227; del buf227  # reuse
            buf261 = buf258; del buf258  # reuse
            buf262 = reinterpret_tensor(buf257, (32, 192, 192), (36864, 1, 192), 0); del buf257  # reuse
            buf267 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [mul_91, X_30, mul_94, X_31, mul_97, X_32, mul_100, X_33, A_24, transpose_34, matmul_74], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_36.run(buf260, buf235, buf243, buf251, buf259, buf261, buf262, buf267, 1179648, stream=stream0)
            del buf235
            del buf243
            del buf251
            buf263 = buf259; del buf259  # reuse
            # Topologically Sorted Source Nodes: [A_24, transpose_34], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf261, buf262, out=buf263)
            buf264 = reinterpret_tensor(buf262, (32, 192, 192), (36864, 192, 1), 0); del buf262  # reuse
            # Topologically Sorted Source Nodes: [mul_102], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf263, buf264, 1179648, stream=stream0)
            buf265 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [mul_102, matmul_73], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf264, buf263, out=buf265)
            del buf264
            buf266 = buf263; del buf263  # reuse
            # Topologically Sorted Source Nodes: [mul_101, B_24], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf266, buf265, 1179648, stream=stream0)
            buf268 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [mul_101, B_24, matmul_74], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf266, buf267, out=buf268)
            buf271 = buf267; del buf267  # reuse
            buf327 = buf266; del buf266  # reuse
            # Topologically Sorted Source Nodes: [w0_norm, mul_103, X_34, w0_main_1, norm_12, add_83, truediv_9, w0_1, bmm_19, gate_before_act_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_41.run(buf108, buf260, buf268, buf110, buf271, buf327, 6144, 192, stream=stream0)
            buf272 = reinterpret_tensor(buf223, (32, 192, 1024), (196608, 1024, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [q, bmm_19, qi_2], Original ATen: [aten.transpose, aten._to_copy, aten.slice, aten.bmm]
            extern_kernels.bmm(buf271, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf272)
            del buf271
            buf323 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [q, qi_2, h_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf322, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf323)
            del buf322
            buf325 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [gate_2, mul_122], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_23.run(buf325, buf323, 6291456, stream=stream0)
            buf326 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [bmm_20, gate_2, mul_122], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.bmm]
            extern_kernels.bmm(buf324, buf325, out=buf326)
            del buf324
            buf328 = buf325; del buf325  # reuse
            # Topologically Sorted Source Nodes: [gate_before_act_2, ki_2, transpose_40], Original ATen: [aten._to_copy, aten.slice, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf327, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf328)
            del buf327
            buf330 = reinterpret_tensor(buf171, (32, 192, 1024), (196608, 1024, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [ki_2, hidden_before_mul_2, transpose_41], Original ATen: [aten.slice, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf329, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf330)
            del buf329
            buf381 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi_2, transpose_42, dhidden_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf380, reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf381)
            del buf380
            buf331 = empty_strided_cuda((32, 1024, 192), (196608, 1, 1024), torch.bfloat16)
            buf382 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_7, hidden_2, transpose_43, lr1i_2, mul_129, type_as_6, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86, dx_2], Original ATen: [aten.silu, aten.mul, aten.transpose, aten.slice, aten._to_copy, aten.sigmoid, aten.rsub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_rsub_sigmoid_silu_slice_transpose_42.run(buf328, buf330, arg7_1, buf381, buf331, buf382, 6291456, stream=stream0)
            del arg7_1
            del buf330
            buf332 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi_2, silu_7, hidden_2, transpose_43, lr1i_2, mul_129, type_as_6, dw1_4], Original ATen: [aten.transpose, aten.slice, aten.silu, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 393216), buf331, out=buf332)
            del arg5_1
            del buf331
            buf333 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i_4, m_i_5], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_43.run(arg3_1, buf333, 32, 1024, stream=stream0)
            del arg3_1
            buf334 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_11, dw1_1, m_i_2, m_i_3, mul_72, dw1_3, m_i_4, m_i_5, mul_133, dw1_5, X_42], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44.run(buf334, buf172, buf12, buf13, buf173, buf333, 1179648, stream=stream0)
            buf335 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_45.run(buf334, buf335, 160, 7373, stream=stream0)
            buf336 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf335, buf336, 32, 5, stream=stream0)
            buf337 = buf172; del buf172  # reuse
            buf338 = reinterpret_tensor(buf12, (32, 192, 192), (36864, 1, 192), 0); del buf12  # reuse
            buf343 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, A_30, transpose_44, matmul_92], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46.run(buf334, buf336, buf337, buf338, buf343, 1179648, stream=stream0)
            buf339 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, A_30, transpose_44], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf337, buf338, out=buf339)
            buf340 = reinterpret_tensor(buf338, (32, 192, 192), (36864, 192, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [mul_136], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf339, buf340, 1179648, stream=stream0)
            buf341 = buf337; del buf337  # reuse
            # Topologically Sorted Source Nodes: [mul_136, matmul_91], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf340, buf339, out=buf341)
            buf342 = buf339; del buf339  # reuse
            # Topologically Sorted Source Nodes: [mul_135, B_30], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf342, buf341, 1179648, stream=stream0)
            buf344 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_135, B_30, matmul_92], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf342, buf343, out=buf344)
            buf345 = buf343; del buf343  # reuse
            buf346 = reinterpret_tensor(buf342, (32, 192, 192), (36864, 1, 192), 0); del buf342  # reuse
            buf351 = buf340; del buf340  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, A_31, transpose_45, matmul_95], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47.run(buf334, buf336, buf344, buf345, buf346, buf351, 1179648, stream=stream0)
            buf347 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, A_31, transpose_45], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf345, buf346, out=buf347)
            buf348 = reinterpret_tensor(buf346, (32, 192, 192), (36864, 192, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [mul_139], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf347, buf348, 1179648, stream=stream0)
            buf349 = buf345; del buf345  # reuse
            # Topologically Sorted Source Nodes: [mul_139, matmul_94], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf348, buf347, out=buf349)
            buf350 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [mul_138, B_31], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf350, buf349, 1179648, stream=stream0)
            buf352 = buf349; del buf349  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_138, B_31, matmul_95], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf350, buf351, out=buf352)
            buf353 = buf351; del buf351  # reuse
            buf354 = reinterpret_tensor(buf350, (32, 192, 192), (36864, 1, 192), 0); del buf350  # reuse
            buf359 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_140, X_45, A_32, transpose_46, matmul_98], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48.run(buf334, buf336, buf344, buf352, buf353, buf354, buf359, 1179648, stream=stream0)
            buf355 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_140, X_45, A_32, transpose_46], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf353, buf354, out=buf355)
            buf356 = reinterpret_tensor(buf354, (32, 192, 192), (36864, 192, 1), 0); del buf354  # reuse
            # Topologically Sorted Source Nodes: [mul_142], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf355, buf356, 1179648, stream=stream0)
            buf357 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [mul_142, matmul_97], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf356, buf355, out=buf357)
            buf358 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [mul_141, B_32], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf358, buf357, 1179648, stream=stream0)
            buf360 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_140, X_45, mul_141, B_32, matmul_98], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf358, buf359, out=buf360)
            buf361 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf362 = buf359; del buf359  # reuse
            buf363 = reinterpret_tensor(buf358, (32, 192, 192), (36864, 1, 192), 0); del buf358  # reuse
            buf368 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_90, X_43, mul_137, X_44, mul_140, X_45, mul_143, X_46, A_33, transpose_47, matmul_101], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49.run(buf334, buf336, buf344, buf352, buf360, buf361, buf362, buf363, buf368, 1179648, stream=stream0)
            del buf334
            del buf344
            buf364 = buf360; del buf360  # reuse
            # Topologically Sorted Source Nodes: [A_33, transpose_47], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf362, buf363, out=buf364)
            buf365 = reinterpret_tensor(buf363, (32, 192, 192), (36864, 192, 1), 0); del buf363  # reuse
            # Topologically Sorted Source Nodes: [mul_145], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf364, buf365, 1179648, stream=stream0)
            buf366 = buf362; del buf362  # reuse
            # Topologically Sorted Source Nodes: [mul_145, matmul_100], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf365, buf364, out=buf366)
            buf367 = buf364; del buf364  # reuse
            # Topologically Sorted Source Nodes: [mul_144, B_33], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf367, buf366, 1179648, stream=stream0)
            buf369 = buf366; del buf366  # reuse
            # Topologically Sorted Source Nodes: [mul_144, B_33, matmul_101], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf367, buf368, out=buf369)
            buf370 = buf368; del buf368  # reuse
            buf371 = reinterpret_tensor(buf367, (32, 192, 192), (36864, 1, 192), 0); del buf367  # reuse
            buf376 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [mul_146, X_47, A_34, transpose_48, matmul_104], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_50.run(buf361, buf369, buf370, buf371, buf376, 1179648, stream=stream0)
            buf372 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [mul_146, X_47, A_34, transpose_48], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf370, buf371, out=buf372)
            buf373 = reinterpret_tensor(buf371, (32, 192, 192), (36864, 192, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [mul_148], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf372, buf373, 1179648, stream=stream0)
            buf374 = buf370; del buf370  # reuse
            # Topologically Sorted Source Nodes: [mul_148, matmul_103], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf373, buf372, out=buf374)
            buf375 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [mul_147, B_34], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf375, buf374, 1179648, stream=stream0)
            buf377 = buf374; del buf374  # reuse
            # Topologically Sorted Source Nodes: [mul_146, X_47, mul_147, B_34, matmul_104], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf375, buf376, out=buf377)
            buf378 = buf57; del buf57  # reuse
            buf484 = buf376; del buf376  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, mul_88, X_27, w1_main_1, mul_146, X_47, mul_149, X_48, w1_main_2, norm_19, add_127, truediv_16, w1_2, bmm_29], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51.run(buf378, buf209, buf217, buf361, buf369, buf377, buf59, buf484, 6144, 192, stream=stream0)
            del buf209
            del buf361
            del buf378
            del buf59
            buf383 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            buf434 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki_2, lr0i_2, mul_130, type_as_7, lr2i_2, mul_131, type_as_8], Original ATen: [aten.slice, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_52.run(arg6_1, arg9_1, arg8_1, buf383, buf434, 6291456, stream=stream0)
            del arg6_1
            del arg8_1
            del arg9_1
            buf384 = buf377; del buf377  # reuse
            # Topologically Sorted Source Nodes: [ki_2, dgate_2, sigma_2, mul_126, sub_2, mul_127, add_86, dx_2, lr0i_2, mul_130, type_as_7, dw0_4], Original ATen: [aten.slice, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf382, buf383, out=buf384)
            del buf382
            buf385 = buf384; del buf384  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_10, dw0_1, m_i_2, m_i_3, mul_71, dw0_3, m_i_4, m_i_5, mul_132, dw0_5, X_49], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44.run(buf385, buf224, buf64, buf13, buf173, buf333, 1179648, stream=stream0)
            buf386 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [norm_16], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_45.run(buf385, buf386, 160, 7373, stream=stream0)
            buf387 = buf336; del buf336  # reuse
            # Topologically Sorted Source Nodes: [norm_16], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf386, buf387, 32, 5, stream=stream0)
            buf388 = buf64; del buf64  # reuse
            buf389 = reinterpret_tensor(buf224, (32, 192, 192), (36864, 1, 192), 0); del buf224  # reuse
            buf394 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, A_35, transpose_49, matmul_107], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46.run(buf385, buf387, buf388, buf389, buf394, 1179648, stream=stream0)
            buf390 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, A_35, transpose_49], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf388, buf389, out=buf390)
            buf391 = reinterpret_tensor(buf389, (32, 192, 192), (36864, 192, 1), 0); del buf389  # reuse
            # Topologically Sorted Source Nodes: [mul_151], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf390, buf391, 1179648, stream=stream0)
            buf392 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [mul_151, matmul_106], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf391, buf390, out=buf392)
            buf393 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [mul_150, B_35], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf393, buf392, 1179648, stream=stream0)
            buf395 = buf392; del buf392  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_150, B_35, matmul_107], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf393, buf394, out=buf395)
            buf396 = buf394; del buf394  # reuse
            buf397 = reinterpret_tensor(buf393, (32, 192, 192), (36864, 1, 192), 0); del buf393  # reuse
            buf402 = buf391; del buf391  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, A_36, transpose_50, matmul_110], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47.run(buf385, buf387, buf395, buf396, buf397, buf402, 1179648, stream=stream0)
            buf398 = buf375; del buf375  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, A_36, transpose_50], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf396, buf397, out=buf398)
            buf399 = reinterpret_tensor(buf397, (32, 192, 192), (36864, 192, 1), 0); del buf397  # reuse
            # Topologically Sorted Source Nodes: [mul_154], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf398, buf399, 1179648, stream=stream0)
            buf400 = buf396; del buf396  # reuse
            # Topologically Sorted Source Nodes: [mul_154, matmul_109], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf399, buf398, out=buf400)
            buf401 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [mul_153, B_36], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf401, buf400, 1179648, stream=stream0)
            buf403 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, mul_153, B_36, matmul_110], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf401, buf402, out=buf403)
            buf404 = buf402; del buf402  # reuse
            buf405 = reinterpret_tensor(buf401, (32, 192, 192), (36864, 1, 192), 0); del buf401  # reuse
            buf410 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, mul_155, X_52, A_37, transpose_51, matmul_113], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48.run(buf385, buf387, buf395, buf403, buf404, buf405, buf410, 1179648, stream=stream0)
            buf406 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, mul_155, X_52, A_37, transpose_51], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf404, buf405, out=buf406)
            buf407 = reinterpret_tensor(buf405, (32, 192, 192), (36864, 192, 1), 0); del buf405  # reuse
            # Topologically Sorted Source Nodes: [mul_157], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf406, buf407, 1179648, stream=stream0)
            buf408 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [mul_157, matmul_112], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf407, buf406, out=buf408)
            buf409 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [mul_156, B_37], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf409, buf408, 1179648, stream=stream0)
            buf411 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, mul_155, X_52, mul_156, B_37, matmul_113], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf409, buf410, out=buf411)
            buf412 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf413 = buf410; del buf410  # reuse
            buf414 = reinterpret_tensor(buf409, (32, 192, 192), (36864, 1, 192), 0); del buf409  # reuse
            buf419 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_101, X_50, mul_152, X_51, mul_155, X_52, mul_158, X_53, A_38, transpose_52, matmul_116], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49.run(buf385, buf387, buf395, buf403, buf411, buf412, buf413, buf414, buf419, 1179648, stream=stream0)
            del buf385
            del buf387
            del buf395
            buf415 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [A_38, transpose_52], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf413, buf414, out=buf415)
            buf416 = reinterpret_tensor(buf414, (32, 192, 192), (36864, 192, 1), 0); del buf414  # reuse
            # Topologically Sorted Source Nodes: [mul_160], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf415, buf416, 1179648, stream=stream0)
            buf417 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [mul_160, matmul_115], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf416, buf415, out=buf417)
            buf418 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [mul_159, B_38], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf418, buf417, 1179648, stream=stream0)
            buf420 = buf417; del buf417  # reuse
            # Topologically Sorted Source Nodes: [mul_159, B_38, matmul_116], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf418, buf419, out=buf420)
            buf421 = buf419; del buf419  # reuse
            buf422 = reinterpret_tensor(buf418, (32, 192, 192), (36864, 1, 192), 0); del buf418  # reuse
            buf427 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [mul_161, X_54, A_39, transpose_53, matmul_119], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_50.run(buf412, buf420, buf421, buf422, buf427, 1179648, stream=stream0)
            buf423 = buf403; del buf403  # reuse
            # Topologically Sorted Source Nodes: [mul_161, X_54, A_39, transpose_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf421, buf422, out=buf423)
            buf424 = reinterpret_tensor(buf422, (32, 192, 192), (36864, 192, 1), 0); del buf422  # reuse
            # Topologically Sorted Source Nodes: [mul_163], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf423, buf424, 1179648, stream=stream0)
            buf425 = buf421; del buf421  # reuse
            # Topologically Sorted Source Nodes: [mul_163, matmul_118], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf424, buf423, out=buf425)
            del buf424
            buf426 = buf423; del buf423  # reuse
            # Topologically Sorted Source Nodes: [mul_162, B_39], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf426, buf425, 1179648, stream=stream0)
            buf428 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [mul_161, X_54, mul_162, B_39, matmul_119], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf426, buf427, out=buf428)
            buf429 = buf108; del buf108  # reuse
            buf431 = buf427; del buf427  # reuse
            # Topologically Sorted Source Nodes: [w0_norm, mul_103, X_34, w0_main_1, mul_161, X_54, mul_164, X_55, w0_main_2, norm_18, add_126, truediv_15, w0_2, bmm_28], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51.run(buf429, buf260, buf268, buf412, buf420, buf428, buf110, buf431, 6144, 192, stream=stream0)
            del buf110
            del buf260
            del buf412
            buf432 = reinterpret_tensor(buf383, (32, 192, 1024), (196608, 1024, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [q, w0_norm, norm_18, add_126, truediv_15, w0_2, bmm_28, qi_3], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.slice, aten.bmm]
            extern_kernels.bmm(buf431, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 589824), out=buf432)
            buf433 = buf381; del buf381  # reuse
            # Topologically Sorted Source Nodes: [silu_8, dhidden_before_mul_2], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_24.run(buf433, buf328, 6291456, stream=stream0)
            del buf328
            buf435 = buf431; del buf431  # reuse
            # Topologically Sorted Source Nodes: [ki_2, silu_8, dhidden_before_mul_2, lr2i_2, mul_131, type_as_8, dw2_4], Original ATen: [aten.slice, aten.silu, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf433, buf434, out=buf435)
            del buf433
            buf436 = buf435; del buf435  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_12, dw2_1, m_i_2, m_i_3, mul_73, dw2_3, m_i_4, m_i_5, mul_134, dw2_5, X_56], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_44.run(buf436, buf275, buf115, buf13, buf173, buf333, 1179648, stream=stream0)
            del buf13
            del buf173
            buf437 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [norm_17], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_45.run(buf436, buf437, 160, 7373, stream=stream0)
            buf438 = buf333; del buf333  # reuse
            # Topologically Sorted Source Nodes: [norm_17], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_6.run(buf437, buf438, 32, 5, stream=stream0)
            del buf437
            buf439 = buf275; del buf275  # reuse
            buf440 = reinterpret_tensor(buf115, (32, 192, 192), (36864, 1, 192), 0); del buf115  # reuse
            buf445 = buf428; del buf428  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, A_40, transpose_54, matmul_122], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_46.run(buf436, buf438, buf439, buf440, buf445, 1179648, stream=stream0)
            buf441 = buf420; del buf420  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, A_40, transpose_54], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf439, buf440, out=buf441)
            buf442 = reinterpret_tensor(buf440, (32, 192, 192), (36864, 192, 1), 0); del buf440  # reuse
            # Topologically Sorted Source Nodes: [mul_166], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_8.run(buf441, buf442, 1179648, stream=stream0)
            buf443 = buf439; del buf439  # reuse
            # Topologically Sorted Source Nodes: [mul_166, matmul_121], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf442, buf441, out=buf443)
            buf444 = buf441; del buf441  # reuse
            # Topologically Sorted Source Nodes: [mul_165, B_40], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf444, buf443, 1179648, stream=stream0)
            buf446 = buf443; del buf443  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_165, B_40, matmul_122], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf444, buf445, out=buf446)
            buf447 = buf445; del buf445  # reuse
            buf448 = reinterpret_tensor(buf444, (32, 192, 192), (36864, 1, 192), 0); del buf444  # reuse
            buf453 = buf442; del buf442  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, A_41, transpose_55, matmul_125], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_47.run(buf436, buf438, buf446, buf447, buf448, buf453, 1179648, stream=stream0)
            buf449 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, A_41, transpose_55], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf447, buf448, out=buf449)
            buf450 = reinterpret_tensor(buf448, (32, 192, 192), (36864, 192, 1), 0); del buf448  # reuse
            # Topologically Sorted Source Nodes: [mul_169], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_11.run(buf449, buf450, 1179648, stream=stream0)
            buf451 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [mul_169, matmul_124], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf450, buf449, out=buf451)
            buf452 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [mul_168, B_41], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_12.run(buf452, buf451, 1179648, stream=stream0)
            buf454 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, mul_168, B_41, matmul_125], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf452, buf453, out=buf454)
            buf455 = buf453; del buf453  # reuse
            buf456 = reinterpret_tensor(buf452, (32, 192, 192), (36864, 1, 192), 0); del buf452  # reuse
            buf461 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, mul_170, X_59, A_42, transpose_56, matmul_128], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_48.run(buf436, buf438, buf446, buf454, buf455, buf456, buf461, 1179648, stream=stream0)
            buf457 = buf426; del buf426  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, mul_170, X_59, A_42, transpose_56], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf455, buf456, out=buf457)
            buf458 = reinterpret_tensor(buf456, (32, 192, 192), (36864, 192, 1), 0); del buf456  # reuse
            # Topologically Sorted Source Nodes: [mul_172], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_14.run(buf457, buf458, 1179648, stream=stream0)
            buf459 = buf455; del buf455  # reuse
            # Topologically Sorted Source Nodes: [mul_172, matmul_127], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf458, buf457, out=buf459)
            buf460 = buf457; del buf457  # reuse
            # Topologically Sorted Source Nodes: [mul_171, B_42], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_15.run(buf460, buf459, 1179648, stream=stream0)
            buf462 = buf459; del buf459  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, mul_170, X_59, mul_171, B_42, matmul_128], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf460, buf461, out=buf462)
            buf463 = buf429; del buf429  # reuse
            buf464 = buf461; del buf461  # reuse
            buf465 = reinterpret_tensor(buf460, (32, 192, 192), (36864, 1, 192), 0); del buf460  # reuse
            buf470 = buf458; del buf458  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_112, X_57, mul_167, X_58, mul_170, X_59, mul_173, X_60, A_43, transpose_57, matmul_131], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_49.run(buf436, buf438, buf446, buf454, buf462, buf463, buf464, buf465, buf470, 1179648, stream=stream0)
            del buf436
            del buf438
            del buf446
            buf466 = buf462; del buf462  # reuse
            # Topologically Sorted Source Nodes: [A_43, transpose_57], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf464, buf465, out=buf466)
            buf467 = reinterpret_tensor(buf465, (32, 192, 192), (36864, 192, 1), 0); del buf465  # reuse
            # Topologically Sorted Source Nodes: [mul_175], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf466, buf467, 1179648, stream=stream0)
            buf468 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [mul_175, matmul_130], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf467, buf466, out=buf468)
            buf469 = buf466; del buf466  # reuse
            # Topologically Sorted Source Nodes: [mul_174, B_43], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_18.run(buf469, buf468, 1179648, stream=stream0)
            buf471 = buf468; del buf468  # reuse
            # Topologically Sorted Source Nodes: [mul_174, B_43, matmul_131], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf469, buf470, out=buf471)
            buf472 = buf470; del buf470  # reuse
            buf473 = reinterpret_tensor(buf469, (32, 192, 192), (36864, 1, 192), 0); del buf469  # reuse
            buf478 = buf467; del buf467  # reuse
            # Topologically Sorted Source Nodes: [mul_176, X_61, A_44, transpose_58, matmul_134], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_50.run(buf463, buf471, buf472, buf473, buf478, 1179648, stream=stream0)
            buf474 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [mul_176, X_61, A_44, transpose_58], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf472, buf473, out=buf474)
            buf475 = reinterpret_tensor(buf473, (32, 192, 192), (36864, 192, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [mul_178], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_20.run(buf474, buf475, 1179648, stream=stream0)
            buf476 = buf472; del buf472  # reuse
            # Topologically Sorted Source Nodes: [mul_178, matmul_133], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf475, buf474, out=buf476)
            del buf475
            buf477 = buf474; del buf474  # reuse
            # Topologically Sorted Source Nodes: [mul_177, B_44], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_21.run(buf477, buf476, 1179648, stream=stream0)
            buf479 = buf476; del buf476  # reuse
            # Topologically Sorted Source Nodes: [mul_176, X_61, mul_177, B_44, matmul_134], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf477, buf478, out=buf479)
            del buf477
            buf480 = buf159; del buf159  # reuse
            buf482 = buf478; del buf478  # reuse
            # Topologically Sorted Source Nodes: [w2_norm, mul_118, X_41, w2_main_1, mul_176, X_61, mul_179, X_62, w2_main_2, norm_20, add_128, truediv_17, w2_2, h_3], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_51.run(buf480, buf311, buf319, buf463, buf471, buf479, buf161, buf482, 6144, 192, stream=stream0)
            del buf161
            del buf311
            del buf319
            del buf463
            del buf471
            del buf479
            del buf480
            buf483 = reinterpret_tensor(buf434, (32, 192, 1024), (196608, 1024, 1), 0); del buf434  # reuse
            # Topologically Sorted Source Nodes: [q, w2_norm, qi_3, norm_20, add_128, truediv_17, w2_2, h_3], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.slice, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf482, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 589824), out=buf483)
            del arg4_1
            del buf482
            buf485 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [gate_3, mul_183], Original ATen: [aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_23.run(buf485, buf483, 6291456, stream=stream0)
            buf486 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, norm_19, add_127, truediv_16, w1_2, bmm_29, gate_3, mul_183], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.silu, aten.bmm]
            extern_kernels.bmm(buf484, buf485, out=buf486)
            del buf484
            del buf485
            buf487 = empty_strided_cuda((32, 192, 4096), (786432, 4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_zeros_like_53.run(buf486, buf326, buf166, buf6, buf487, 25165824, stream=stream0)
            del buf166
            del buf326
            del buf486
            del buf6
            buf488 = empty_strided_cuda((32, 4096, 192), (786432, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_zeros_like_54.run(buf487, buf488, 131072, 192, stream=stream0)
            del buf487
        return (buf488, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 192, 192), (36864, 192, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 4096, 1), (4096, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((32, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((32, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((32, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, 4096, 1), (12288, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
