# AOT ID: ['3_inference']
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
#   gate_before_act => convert_element_type_13
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_3,%convert_element_type_13
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/hv/chvei5pphgs2nlzco3cqmio65mvyutdmcjqmxalpvjkcviycfnh5.py
# Topologically Sorted Source Nodes: [gate, mul, getitem_8, float_1, x_rot, x1, hci, c, mul_1, x2, hsi, s_, mul_2, sub, mul_3, mul_4, add, y, silu_1, hidden, getitem_14, float_2, x_rot_1, x1_1, c_1, mul_6, x2_1, s__1, mul_7, sub_1, mul_8, mul_9, add_1, y_2], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   c => unsqueeze
#   c_1 => unsqueeze_2
#   float_1 => convert_element_type_8
#   float_2 => convert_element_type_21
#   gate => convert_element_type_6, convert_element_type_7, mul, sigmoid
#   getitem_14 => slice_13
#   getitem_8 => slice_9
#   hci => slice_7
#   hidden => mul_7
#   hsi => slice_8
#   mul => mul_1
#   mul_1 => mul_2
#   mul_2 => mul_3
#   mul_3 => mul_4
#   mul_4 => mul_5
#   mul_6 => mul_8
#   mul_7 => mul_9
#   mul_8 => mul_10
#   mul_9 => mul_11
#   s_ => unsqueeze_1
#   s__1 => unsqueeze_3
#   silu_1 => convert_element_type_19, convert_element_type_20, mul_6, sigmoid_1
#   sub => sub
#   sub_1 => sub_1
#   x1 => select
#   x1_1 => select_2
#   x2 => select_1
#   x2_1 => select_3
#   x_rot => view
#   x_rot_1 => view_3
#   y => cat
#   y_2 => cat_2
# Graph fragment:
#   %bmm_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm]
#   %arg10_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %bmm_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %bmm_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %convert_element_type_6 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_1, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %bmm), kwargs = {})
#   %slice_9 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1, 1, 0, 96), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_9, torch.float32), kwargs = {})
#   %view : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_8, [32, 48, 2, 1024]), kwargs = {})
#   %select : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view, 2, 0), kwargs = {})
#   %slice_7 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg10_1, 1, 0, 1024), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_7, 0), kwargs = {})
#   %mul_2 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %unsqueeze), kwargs = {})
#   %select_1 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view, 2, 1), kwargs = {})
#   %slice_8 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg11_1, 1, 0, 1024), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_8, 0), kwargs = {})
#   %mul_3 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %unsqueeze_1), kwargs = {})
#   %sub : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_4 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %unsqueeze_1), kwargs = {})
#   %mul_5 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %cat : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %add], 2), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_6 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
#   %mul_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %bmm_4), kwargs = {})
#   %slice_13 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_7, 1, 0, 96), kwargs = {})
#   %convert_element_type_21 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_13, torch.float32), kwargs = {})
#   %view_3 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_21, [32, 48, 2, 1024]), kwargs = {})
#   %select_2 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_3, 2, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_7, 0), kwargs = {})
#   %mul_8 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %unsqueeze_2), kwargs = {})
#   %select_3 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_3, 2, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_8, 0), kwargs = {})
#   %mul_9 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %unsqueeze_3), kwargs = {})
#   %sub_1 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %unsqueeze_3), kwargs = {})
#   %mul_11 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %unsqueeze_2), kwargs = {})
#   %add_1 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %cat_2 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_1, %add_1], 2), kwargs = {})
#   return %cat,%cat_2
triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 127401984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 - tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 2048, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + (4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.sigmoid(tmp42)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tmp45 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr2 + (4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp40 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tmp55 = tl.load(in_ptr4 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.sigmoid(tmp56)
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.load(in_ptr5 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp62 * tmp13
    tmp64 = tl.load(in_ptr4 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tl.sigmoid(tmp65)
    tmp67 = tmp65 * tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.load(in_ptr5 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp70 = tmp68 * tmp69
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp23
    tmp73 = tmp63 - tmp72
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp4, tmp73, tmp74)
    tmp76 = tl.load(in_ptr4 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tl.sigmoid(tmp77)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tl.load(in_ptr5 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp82 = tmp80 * tmp81
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp83 * tmp39
    tmp85 = tl.load(in_ptr4 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tl.sigmoid(tmp86)
    tmp88 = tmp86 * tmp87
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tl.load(in_ptr5 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp91 = tmp89 * tmp90
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp92 * tmp49
    tmp94 = tmp84 + tmp93
    tmp95 = tl.full(tmp94.shape, 0.0, tmp94.dtype)
    tmp96 = tl.where(tmp28, tmp94, tmp95)
    tmp97 = tl.where(tmp4, tmp75, tmp96)
    tl.store(out_ptr0 + (x3), tmp54, None)
    tl.store(out_ptr1 + (x3), tmp97, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/4w/c4w774ylihulpp5eyiqyav7i366gvn6miybbufzs3q7btwv5veej.py
# Topologically Sorted Source Nodes: [silu_1, hidden, y_2, reshape_3, y_3, getitem_19, hidden_rot, transpose_5, lr1i, mul_19, type_as_3], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.transpose]
# Source node to ATen node mapping:
#   getitem_19 => slice_14
#   hidden => mul_7
#   hidden_rot => cat_3
#   lr1i => slice_4
#   mul_19 => mul_22
#   reshape_3 => view_5
#   silu_1 => convert_element_type_19, convert_element_type_20, mul_6, sigmoid_1
#   transpose_5 => permute_6
#   type_as_3 => convert_element_type_30
#   y_2 => view_4
#   y_3 => convert_element_type_22
# Graph fragment:
#   %cat_2 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %bmm_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %bmm_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %arg7_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_19 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_6 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
#   %mul_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %bmm_4), kwargs = {})
#   %view_4 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_2, [32, 48, 2, 1024]), kwargs = {})
#   %view_5 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [32, 96, 1024]), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.bfloat16), kwargs = {})
#   %slice_14 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_7, 1, 96, 9223372036854775807), kwargs = {})
#   %cat_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_22, %slice_14], 1), kwargs = {})
#   %permute_6 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%cat_3, [0, 2, 1]), kwargs = {})
#   %slice_4 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg7_1, 1, 0, 1024), kwargs = {})
#   %mul_22 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, %slice_4), kwargs = {})
#   %convert_element_type_30 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   return %convert_element_type_30
triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_2 = async_compile.triton('triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331648, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp23 = tl.load(in_ptr3 + (3*y0 + 12288*y1), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 1024*(x2) + 98304*y1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1, 1], 192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp8, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp25, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/mo/cmof3luhrubafgkhwgolyv5lzsbenjkyn53heop2jxd65spc6sot.py
# Topologically Sorted Source Nodes: [m_i, m_i_1], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i => slice_17
#   m_i_1 => mean
# Graph fragment:
#   %arg3_1 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   return %buf15
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/eg/ceg2fyyefcvhyvhmu35fwjrdh73skw46p526mbfg3r4g2dllm4yx.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type_39
#   dw1_1 => add_5
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   mul_23 => mul_26
#   norm_3 => convert_element_type_40, pow_7, sum_4
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_39, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_40, 2), kwargs = {})
#   %sum_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   return %buf16
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_4 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 2359360}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_4(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/wq/cwqke4akwblzyfm3yskdhrphsnh6o6yw6hmecfdyfoq4whyeyxxa.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type_39
#   dw1_1 => add_5
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   mul_23 => mul_26
#   norm_3 => convert_element_type_40, pow_7, sum_4
# Graph fragment:
#   %buf16 : Tensor "f32[32, 1, 1, 5][5, 160, 160, 1]cuda:0" = PlaceHolder[target=buf16]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_39, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_40, 2), kwargs = {})
#   %sum_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   return %sum_4
triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5 = async_compile.triton('triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 400}}
)
@triton.jit
def triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/vy/cvyfaezus2vhbe3n3g5qkco6k2oyzldwnqc45gmcaq2mwvzmaqfl.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, A, transpose_6, matmul_2], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A => convert_element_type_41, convert_element_type_42
#   X => convert_element_type_39
#   X_1 => div
#   add_7 => add_7
#   dw1_1 => add_5
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   matmul_2 => convert_element_type_47
#   mul_23 => mul_26
#   norm_3 => pow_8
#   transpose_6 => permute_7
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %sum_4 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_4]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add_7 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_39, %add_7), kwargs = {})
#   %convert_element_type_42 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %permute_7 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1]), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.bfloat16), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %expand,%expand_1,%expand_5
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_6 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16515072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/s4/cs4p7rh3fhopnos2owzt7jqsqjggo2lr4oy3xxr2n6igtuhvsfn5.py
# Topologically Sorted Source Nodes: [mul_26], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_26 => mul_29
# Graph fragment:
#   %expand_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %mul_29 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 2.927), kwargs = {})
#   return %expand_2
triton_poi_fused_mul_7 = async_compile.triton('triton_poi_fused_mul_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/v2/cv2azj4zegyhykhn3luzxnu7io3f6gufzg2v4npuumhfndpgce32.py
# Topologically Sorted Source Nodes: [mul_25, B], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B => add_8
#   mul_25 => mul_28
# Graph fragment:
#   %expand_3 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %bmm_10 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %mul_28 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, -6.8946), kwargs = {})
#   %add_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %bmm_10), kwargs = {})
#   return %expand_4
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/rr/crr7qcs3faywppzeoptl3xykochaijykqee4fvmyksoj3zihirkg.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_27, X_2, A_1, transpose_7, matmul_5], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_1 => convert_element_type_50, convert_element_type_51
#   X => convert_element_type_39
#   X_1 => div
#   X_2 => add_9
#   add_7 => add_7
#   dw1_1 => add_5
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   matmul_5 => convert_element_type_56
#   mul_23 => mul_26
#   mul_27 => mul_30
#   norm_3 => pow_8
#   transpose_7 => permute_8
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %sum_4 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_4]
#   %bmm_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add_7 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_39, %add_7), kwargs = {})
#   %mul_30 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_9 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %bmm_11), kwargs = {})
#   %convert_element_type_51 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   %permute_8 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_9, [0, 2, 1]), kwargs = {})
#   %convert_element_type_50 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_8, torch.bfloat16), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   return %expand_6,%expand_7,%expand_11
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/uz/cuzx7fo3ea4fg7nsqw3sgvkxarf34pb5kg7f64yyosiul2xjne7v.py
# Topologically Sorted Source Nodes: [mul_29], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_29 => mul_32
# Graph fragment:
#   %expand_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %mul_32 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, 2.6377), kwargs = {})
#   return %expand_8
triton_poi_fused_mul_10 = async_compile.triton('triton_poi_fused_mul_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/pg/cpgxh2hyvwnqkdjux3wnumbakpg3yu6vgtjpntghnpqwzq53vtli.py
# Topologically Sorted Source Nodes: [mul_28, B_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_1 => add_10
#   mul_28 => mul_31
# Graph fragment:
#   %expand_9 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %bmm_13 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %mul_31 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, -6.3029), kwargs = {})
#   %add_10 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %bmm_13), kwargs = {})
#   return %expand_10
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ad/cadvmisgiluns3iwf35ln6zrquzx5hynprgmlogoyjylqk5ihwuc.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_27, X_2, mul_30, X_3, A_2, transpose_8, matmul_8], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_2 => convert_element_type_59, convert_element_type_60
#   X => convert_element_type_39
#   X_1 => div
#   X_2 => add_9
#   X_3 => add_11
#   add_7 => add_7
#   dw1_1 => add_5
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   matmul_8 => convert_element_type_65
#   mul_23 => mul_26
#   mul_27 => mul_30
#   mul_30 => mul_33
#   norm_3 => pow_8
#   transpose_8 => permute_9
# Graph fragment:
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %sum_4 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_4]
#   %bmm_11 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %add_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_11]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %pow_8 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %add_7 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_8, 1e-07), kwargs = {})
#   %div : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_39, %add_7), kwargs = {})
#   %mul_30 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4.0848), kwargs = {})
#   %add_9 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %bmm_11), kwargs = {})
#   %mul_33 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 3.9505), kwargs = {})
#   %add_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %bmm_14), kwargs = {})
#   %convert_element_type_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_11, [0, 2, 1]), kwargs = {})
#   %convert_element_type_59 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_9, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.bfloat16), kwargs = {})
#   return %add_11,%expand_12,%expand_13,%expand_17
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_12 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30670848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7v/c7vfm4sx42kxft2pvutloxds6x3uxz2egjxagyl6osxobnw4625n.py
# Topologically Sorted Source Nodes: [mul_32], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_32 => mul_35
# Graph fragment:
#   %expand_15 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %mul_35 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_15, 2.3037), kwargs = {})
#   return %expand_14
triton_poi_fused_mul_13 = async_compile.triton('triton_poi_fused_mul_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/la/claaje4ttau3xlcpis2kcvlubjn27r6psvi5mzd4tjwubt6t7zj7.py
# Topologically Sorted Source Nodes: [mul_31, B_2], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_2 => add_12
#   mul_31 => mul_34
# Graph fragment:
#   %expand_15 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %bmm_16 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_16]
#   %mul_34 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_15, -5.5913), kwargs = {})
#   %add_12 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %bmm_16), kwargs = {})
#   return %expand_16
triton_poi_fused_add_mul_14 = async_compile.triton('triton_poi_fused_add_mul_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/v6/cv6qeeakorobfnyzow2g6webutfsufrpr35e3644n3es4dgo7zvx.py
# Topologically Sorted Source Nodes: [mul_33, X_4, A_3, transpose_9, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_3 => convert_element_type_68, convert_element_type_69
#   X_4 => add_13
#   matmul_11 => convert_element_type_74
#   mul_33 => mul_36
#   transpose_9 => permute_10
# Graph fragment:
#   %add_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_11]
#   %bmm_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %mul_36 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 3.7418), kwargs = {})
#   %add_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %bmm_17), kwargs = {})
#   %convert_element_type_69 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.bfloat16), kwargs = {})
#   %permute_10 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_13, [0, 2, 1]), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_10, torch.bfloat16), kwargs = {})
#   %convert_element_type_74 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.bfloat16), kwargs = {})
#   return %expand_18,%expand_19,%expand_23
triton_poi_fused__to_copy_add_mul_transpose_15 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/2q/c2q7f733lvbq3uvwriayt2ogzcyjbzing3avlbtczktutvr4sigk.py
# Topologically Sorted Source Nodes: [mul_35], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_35 => mul_38
# Graph fragment:
#   %expand_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %mul_38 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_18, 1.2046), kwargs = {})
#   return %expand_20
triton_poi_fused_mul_16 = async_compile.triton('triton_poi_fused_mul_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/qv/cqvhe5sdyac4z4jhro4tvd4e7xyexlb332co56rvuctrgsbiktmn.py
# Topologically Sorted Source Nodes: [mul_34, B_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_3 => add_14
#   mul_34 => mul_37
# Graph fragment:
#   %expand_21 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %bmm_19 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_19]
#   %mul_37 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_18, -3.1427), kwargs = {})
#   %add_14 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %bmm_19), kwargs = {})
#   return %expand_22
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/df/cdfualzso655vi62tlstfp7nsqhfr3yupyyol4kdelj5ss7cms6a.py
# Topologically Sorted Source Nodes: [mul_33, X_4, mul_36, X_5, A_4, transpose_10, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_4 => convert_element_type_77, convert_element_type_78
#   X_4 => add_13
#   X_5 => add_15
#   matmul_14 => convert_element_type_83
#   mul_33 => mul_36
#   mul_36 => mul_39
#   transpose_10 => permute_11
# Graph fragment:
#   %add_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_11]
#   %bmm_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %bmm_20 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %mul_36 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 3.7418), kwargs = {})
#   %add_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %bmm_17), kwargs = {})
#   %mul_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, 2.8769), kwargs = {})
#   %add_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %bmm_20), kwargs = {})
#   %convert_element_type_78 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.bfloat16), kwargs = {})
#   %permute_11 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_15, [0, 2, 1]), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_11, torch.bfloat16), kwargs = {})
#   %convert_element_type_83 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.bfloat16), kwargs = {})
#   return %expand_24,%expand_25,%expand_29
triton_poi_fused__to_copy_add_mul_transpose_18 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/3o/c3ok5fa2ezetfithrpdda45uqkmddukwimznuf3fpn4n2zrugxvr.py
# Topologically Sorted Source Nodes: [mul_38], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_38 => mul_41
# Graph fragment:
#   %expand_27 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %mul_41 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_21, 1.2012), kwargs = {})
#   return %expand_26
triton_poi_fused_mul_19 = async_compile.triton('triton_poi_fused_mul_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7077888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ng/cng7gpk6lpdwvtm3siv2xy3fgbivztujjh3a32qrjahqt3s5qjvb.py
# Topologically Sorted Source Nodes: [mul_37, B_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_4 => add_16
#   mul_37 => mul_40
# Graph fragment:
#   %expand_27 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %bmm_22 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_22]
#   %mul_40 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_21, -3.0525), kwargs = {})
#   %add_16 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, %bmm_22), kwargs = {})
#   return %expand_28
triton_poi_fused_add_mul_20 = async_compile.triton('triton_poi_fused_add_mul_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/jk/cjktkz3et2xpibggw2eijcv6w4b33dkkzxkh6xrvzzmtmjsauryp.py
# Topologically Sorted Source Nodes: [bmm_2, mul_33, X_4, mul_36, X_5, mul_39, X_6, w1_main, w1_norm, transpose_4, dhidden_rot], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.linalg_vector_norm, aten.transpose]
# Source node to ATen node mapping:
#   X_4 => add_13
#   X_5 => add_15
#   X_6 => add_17
#   bmm_2 => convert_element_type_10
#   dhidden_rot => convert_element_type_23
#   mul_33 => mul_36
#   mul_36 => mul_39
#   mul_39 => mul_42
#   transpose_4 => permute_5
#   w1_main => add_40
#   w1_norm => pow_3, sum_2
# Graph fragment:
#   %arg1_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_11]
#   %bmm_17 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_17]
#   %bmm_20 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %bmm_23 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_23]
#   %convert_element_type_10 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   %mul_36 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 3.7418), kwargs = {})
#   %add_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %bmm_17), kwargs = {})
#   %mul_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, 2.8769), kwargs = {})
#   %add_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %bmm_20), kwargs = {})
#   %mul_42 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, 2.8366), kwargs = {})
#   %add_17 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %bmm_23), kwargs = {})
#   %add_40 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %add_17), kwargs = {})
#   %pow_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [2], True), kwargs = {})
#   %permute_5 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 2, 1]), kwargs = {})
#   %convert_element_type_23 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_5, torch.bfloat16), kwargs = {})
#   return %sum_2,%convert_element_type_10,%add_40,%convert_element_type_23
triton_per_fused__to_copy_add_linalg_vector_norm_mul_transpose_21 = async_compile.triton('triton_per_fused__to_copy_add_linalg_vector_norm_mul_transpose_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_linalg_vector_norm_mul_transpose_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 35389440}}
)
@triton.jit
def triton_per_fused__to_copy_add_linalg_vector_norm_mul_transpose_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr3 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = tmp0.to(tl.float32)
    tmp8 = 3.7418
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = 2.8769
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = 2.8366
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp0 + tmp22
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp6, r0_mask & xmask)
    tl.store(in_out_ptr0 + (r0_1 + 192*x0), tmp23, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 192*x0), tmp6, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/gg/cggfcufjykpaa47h5p6cxsxydgmanbpeuvqbux7jtq27wuh4c6gj.py
# Topologically Sorted Source Nodes: [gate, mul, y, reshape_1, y_1, getitem_13, hq_rot], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat]
# Source node to ATen node mapping:
#   gate => convert_element_type_6, convert_element_type_7, mul, sigmoid
#   getitem_13 => slice_10
#   hq_rot => cat_1
#   mul => mul_1
#   reshape_1 => view_2
#   y => view_1
#   y_1 => convert_element_type_9
# Graph fragment:
#   %cat : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0" = PlaceHolder[target=cat]
#   %bmm_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm]
#   %convert_element_type_6 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_1, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %bmm), kwargs = {})
#   %view_1 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [32, 48, 2, 1024]), kwargs = {})
#   %view_2 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [32, 96, 1024]), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.bfloat16), kwargs = {})
#   %slice_10 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1, 1, 96, 9223372036854775807), kwargs = {})
#   %cat_1 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_9, %slice_10], 1), kwargs = {})
#   return %cat_1
triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22 = async_compile.triton('triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 75497472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 192)
    x0 = (xindex % 1024)
    x2 = xindex // 196608
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (98304 + x0 + 1024*((-96) + x1) + 196608*x2), tmp9, other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (98304 + x0 + 1024*((-96) + x1) + 196608*x2), tmp9, other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp8, tmp20)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/5x/c5xnj44u3zl2c6hqlcgs4doip5jrzd5c6wdnc5igarthmgr4ixei.py
# Topologically Sorted Source Nodes: [hci, hsi, getitem_20, float_3, x_rot_2, x1_2, c_2, mul_10, x2_2, neg, s__2, mul_11, sub_2, mul_12, mul_13, add_2, y_4], Original ATen: [aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.mul, aten.neg, aten.sub, aten.add, aten.stack]
# Source node to ATen node mapping:
#   add_2 => add_2
#   c_2 => unsqueeze_4
#   float_3 => convert_element_type_26
#   getitem_20 => slice_15
#   hci => slice_7
#   hsi => slice_8
#   mul_10 => mul_12
#   mul_11 => mul_13
#   mul_12 => mul_14
#   mul_13 => mul_15
#   neg => neg
#   s__2 => unsqueeze_5
#   sub_2 => sub_2
#   x1_2 => select_4
#   x2_2 => select_5
#   x_rot_2 => view_6
#   y_4 => cat_4
# Graph fragment:
#   %bmm_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %arg10_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %slice_7 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg10_1, 1, 0, 1024), kwargs = {})
#   %slice_8 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg11_1, 1, 0, 1024), kwargs = {})
#   %slice_15 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_5, 1, 0, 96), kwargs = {})
#   %convert_element_type_26 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_15, torch.float32), kwargs = {})
#   %view_6 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_26, [32, 48, 2, 1024]), kwargs = {})
#   %select_4 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_6, 2, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_7, 0), kwargs = {})
#   %mul_12 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %unsqueeze_4), kwargs = {})
#   %select_5 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_6, 2, 1), kwargs = {})
#   %neg : Tensor "f32[48, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_8,), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 48, 1024][49152, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%neg, 0), kwargs = {})
#   %mul_13 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %unsqueeze_5), kwargs = {})
#   %sub_2 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %mul_14 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %unsqueeze_5), kwargs = {})
#   %mul_15 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %unsqueeze_4), kwargs = {})
#   %add_2 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %mul_15), kwargs = {})
#   %cat_4 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_2, %add_2], 2), kwargs = {})
#   return %cat_4
triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_23 = async_compile.triton('triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 45613056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = -tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 - tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 2048, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr2 + (4096*x1 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp23 = -tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (4096*x1 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp24 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp17, tmp29, tmp30)
    tmp32 = tl.where(tmp4, tmp16, tmp31)
    tl.store(out_ptr0 + (x3), tmp32, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/gz/cgzcbqyg5bojdfv3nrfx2svfuil3mrydjlws5fmgzkazixa6ujbn.py
# Topologically Sorted Source Nodes: [y_4, reshape_5, y_5, getitem_25, dhidden, dgate, sigma, mul_16, sub_3, mul_17, add_3, dx, silu_2, dhidden_before_mul], Original ATen: [aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_3 => add_3
#   dgate => mul_18
#   dhidden => cat_5
#   dhidden_before_mul => mul_17
#   dx => mul_21
#   getitem_25 => slice_16
#   mul_16 => mul_19
#   mul_17 => mul_20
#   reshape_5 => view_8
#   sigma => sigmoid_3
#   silu_2 => convert_element_type_28, convert_element_type_29, mul_16, sigmoid_2
#   sub_3 => sub_3
#   y_4 => view_7
#   y_5 => convert_element_type_27
# Graph fragment:
#   %cat_4 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0" = PlaceHolder[target=cat_4]
#   %bmm_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_4 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %bmm_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %view_7 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_4, [32, 48, 2, 1024]), kwargs = {})
#   %view_8 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [32, 96, 1024]), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.bfloat16), kwargs = {})
#   %slice_16 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_5, 1, 96, 9223372036854775807), kwargs = {})
#   %cat_5 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_27, %slice_16], 1), kwargs = {})
#   %mul_18 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_5, %bmm_4), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm_3,), kwargs = {})
#   %mul_19 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %sigmoid_3), kwargs = {})
#   %sub_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_3), kwargs = {})
#   %mul_20 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, %sub_3), kwargs = {})
#   %add_3 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, 1), kwargs = {})
#   %mul_21 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %add_3), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_3, torch.float32), kwargs = {})
#   %sigmoid_2 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_28,), kwargs = {})
#   %mul_16 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_28, %sigmoid_2), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   %mul_17 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_5, %convert_element_type_29), kwargs = {})
#   return %mul_21,%mul_17
triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24 = async_compile.triton('triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 192)
    x0 = (xindex % 1024)
    x2 = xindex // 196608
    x3 = xindex
    tmp14 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (98304 + x0 + 1024*((-96) + x1) + 196608*x2), tmp9, other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp4, tmp8, tmp12)
    tmp15 = tmp13 * tmp14
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 1.0
    tmp20 = tmp19 - tmp17
    tmp21 = tmp16 * tmp20
    tmp22 = tmp21 + tmp19
    tmp23 = tmp18 * tmp22
    tmp24 = tmp16.to(tl.float32)
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp13 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp23, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/t6/ct63xk3s3sn6ud3vb7ye5tl4z6sf7wrknc3dihum6tonpqg32s4i.py
# Topologically Sorted Source Nodes: [ki, lr0i, mul_20, type_as_4, lr2i, mul_21, type_as_5], Original ATen: [aten.slice, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   ki => slice_1
#   lr0i => slice_6
#   lr2i => slice_5
#   mul_20 => mul_23
#   mul_21 => mul_24
#   type_as_4 => convert_element_type_33
#   type_as_5 => convert_element_type_36
# Graph fragment:
#   %arg6_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg8_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %slice_1 : Tensor "bf16[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 0, 1024), kwargs = {})
#   %slice_6 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg9_1, 1, 0, 1024), kwargs = {})
#   %mul_23 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, %slice_6), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   %slice_5 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg8_1, 1, 0, 1024), kwargs = {})
#   %mul_24 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, %slice_5), kwargs = {})
#   %convert_element_type_36 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   return %convert_element_type_33,%convert_element_type_36
triton_poi_fused__to_copy_mul_slice_25 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62914560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/de/cdesgm2ytjwmz5p3aw2ozdhbrcz75dj2aed7xet4rj567duwtuqd.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, dw2_momentum, mul_24, dw2_1, X_14, norm_5], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_14 => convert_element_type_133
#   X_7 => convert_element_type_86
#   dw0_1 => add_4
#   dw0_momentum => full_default_1
#   dw2_1 => add_6
#   dw2_momentum => full_default_2
#   m_i => slice_17
#   m_i_1 => mean
#   mul_22 => mul_25
#   mul_24 => mul_27
#   norm_4 => convert_element_type_87, pow_9, sum_5
#   norm_5 => convert_element_type_134, pow_11, sum_6
# Graph fragment:
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_25), kwargs = {})
#   %convert_element_type_86 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_87 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_86, torch.float32), kwargs = {})
#   %pow_9 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_87, 2), kwargs = {})
#   %sum_5 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2], True), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_27), kwargs = {})
#   %convert_element_type_133 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   %convert_element_type_134 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_133, torch.float32), kwargs = {})
#   %pow_11 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_134, 2), kwargs = {})
#   %sum_6 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2], True), kwargs = {})
#   return %buf68,%buf119
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_26 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 4718720}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/wm/cwm6o2xpgvnti6ixt64n22xjujao7i6n5x567hask3qi2wlaceov.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, mul_42, X_9, mul_45, X_10, A_7, transpose_13, matmul_23, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, A_10, transpose_16, matmul_32], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_10 => convert_element_type_135, convert_element_type_136
#   A_7 => convert_element_type_106, convert_element_type_107
#   X_10 => add_22
#   X_14 => convert_element_type_133
#   X_15 => div_2
#   X_7 => convert_element_type_86
#   X_8 => div_1
#   X_9 => add_20
#   add_18 => add_18
#   add_29 => add_29
#   dw0_1 => add_4
#   dw0_momentum => full_default_1
#   dw2_1 => add_6
#   dw2_momentum => full_default_2
#   m_i => slice_17
#   m_i_1 => mean
#   matmul_23 => convert_element_type_112
#   matmul_32 => convert_element_type_141
#   mul_22 => mul_25
#   mul_24 => mul_27
#   mul_42 => mul_45
#   mul_45 => mul_48
#   norm_4 => pow_10
#   norm_5 => pow_12
#   transpose_13 => permute_14
#   transpose_16 => permute_17
# Graph fragment:
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %sum_5 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_5]
#   %bmm_26 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_26]
#   %bmm_29 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_29]
#   %add_22 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_22]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %sum_6 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_6]
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_25), kwargs = {})
#   %convert_element_type_86 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %pow_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   %add_18 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_10, 1e-07), kwargs = {})
#   %div_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_86, %add_18), kwargs = {})
#   %mul_45 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, 4.0848), kwargs = {})
#   %add_20 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %bmm_26), kwargs = {})
#   %mul_48 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, 3.9505), kwargs = {})
#   %add_22 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %bmm_29), kwargs = {})
#   %convert_element_type_107 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %permute_14 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_22, [0, 2, 1]), kwargs = {})
#   %convert_element_type_106 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_14, torch.bfloat16), kwargs = {})
#   %convert_element_type_112 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_27), kwargs = {})
#   %convert_element_type_133 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   %pow_12 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %add_29 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, 1e-07), kwargs = {})
#   %div_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_133, %add_29), kwargs = {})
#   %convert_element_type_136 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.bfloat16), kwargs = {})
#   %permute_17 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_2, [0, 2, 1]), kwargs = {})
#   %convert_element_type_135 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_17, torch.bfloat16), kwargs = {})
#   %convert_element_type_141 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.bfloat16), kwargs = {})
#   return %add_22,%expand_42,%expand_43,%expand_47,%expand_60,%expand_61,%expand_65
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 47185920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/yv/cyvst5fs3wehwrvmmjqg5ie2wk7khwytafgf52j43gpd3zbg2fvc.py
# Topologically Sorted Source Nodes: [mul_48, X_11, mul_51, X_12, mul_54, X_13, w0_main, norm_6, add_43, truediv_3, w0_norm, w0, bmm_10, gate_before_act_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_11 => add_24
#   X_12 => add_26
#   X_13 => add_28
#   add_43 => add_43
#   bmm_10 => convert_element_type_183
#   gate_before_act_1 => convert_element_type_193
#   mul_48 => mul_51
#   mul_51 => mul_54
#   mul_54 => mul_57
#   norm_6 => pow_13, pow_14, sum_7
#   truediv_3 => div_3
#   w0 => mul_73
#   w0_main => add_41
#   w0_norm => pow_1, pow_2, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %add_22 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_22]
#   %bmm_32 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_32]
#   %bmm_35 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_35]
#   %bmm_38 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_38]
#   %add_41 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_41]
#   %sum_7 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_7]
#   %sum_1 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_1]
#   %mul_51 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 3.7418), kwargs = {})
#   %add_24 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %bmm_32), kwargs = {})
#   %mul_54 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, 2.8769), kwargs = {})
#   %add_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %bmm_35), kwargs = {})
#   %mul_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, 2.8366), kwargs = {})
#   %add_28 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %bmm_38), kwargs = {})
#   %add_41 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %add_28), kwargs = {})
#   %pow_13 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_41, 2), kwargs = {})
#   %sum_7 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [2], True), kwargs = {})
#   %pow_14 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %add_43 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_14, 1e-05), kwargs = {})
#   %div_3 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_41, %add_43), kwargs = {})
#   %pow_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %sum_1 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2], True), kwargs = {})
#   %pow_2 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mul_73 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %pow_2), kwargs = {})
#   %convert_element_type_183 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_73, torch.bfloat16), kwargs = {})
#   %convert_element_type_193 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_73, torch.bfloat16), kwargs = {})
#   return %sum_1,%add_41,%sum_7,%convert_element_type_183,%convert_element_type_193
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_28 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 35389440}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ok/cokcxd6xrlbkpgrm6mhcq3iyhskaijqh5wtgupe3tyivkzvw3zcz.py
# Topologically Sorted Source Nodes: [gate_1, mul_73, getitem_35, float_4, x_rot_3, x1_3, hci_1, c_3, mul_74, x2_3, hsi_1, s__3, mul_75, sub_4, mul_76, mul_77, add_46, y_6, silu_4, hidden_1, getitem_41, float_5, x_rot_4, x1_4, c_4, mul_79, x2_4, s__4, mul_80, sub_5, mul_81, mul_82, add_47, y_8], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack]
# Source node to ATen node mapping:
#   add_46 => add_46
#   add_47 => add_47
#   c_3 => unsqueeze_6
#   c_4 => unsqueeze_8
#   float_4 => convert_element_type_188
#   float_5 => convert_element_type_201
#   gate_1 => convert_element_type_186, convert_element_type_187, mul_76, sigmoid_4
#   getitem_35 => slice_26
#   getitem_41 => slice_31
#   hci_1 => slice_24
#   hidden_1 => mul_83
#   hsi_1 => slice_25
#   mul_73 => mul_77
#   mul_74 => mul_78
#   mul_75 => mul_79
#   mul_76 => mul_80
#   mul_77 => mul_81
#   mul_79 => mul_84
#   mul_80 => mul_85
#   mul_81 => mul_86
#   mul_82 => mul_87
#   s__3 => unsqueeze_7
#   s__4 => unsqueeze_9
#   silu_4 => convert_element_type_199, convert_element_type_200, mul_82, sigmoid_5
#   sub_4 => sub_4
#   sub_5 => sub_5
#   x1_3 => select_6
#   x1_4 => select_8
#   x2_3 => select_7
#   x2_4 => select_9
#   x_rot_3 => view_144
#   x_rot_4 => view_147
#   y_6 => cat_6
#   y_8 => cat_8
# Graph fragment:
#   %bmm_55 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_55]
#   %bmm_54 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_54]
#   %arg10_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %bmm_57 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %bmm_58 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_58]
#   %convert_element_type_186 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_55, torch.float32), kwargs = {})
#   %sigmoid_4 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_186,), kwargs = {})
#   %mul_76 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_186, %sigmoid_4), kwargs = {})
#   %convert_element_type_187 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_76, torch.bfloat16), kwargs = {})
#   %mul_77 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_187, %bmm_54), kwargs = {})
#   %slice_26 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_77, 1, 0, 96), kwargs = {})
#   %convert_element_type_188 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_26, torch.float32), kwargs = {})
#   %view_144 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_188, [32, 48, 2, 1024]), kwargs = {})
#   %select_6 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_144, 2, 0), kwargs = {})
#   %slice_24 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg10_1, 1, 1024, 2048), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_24, 0), kwargs = {})
#   %mul_78 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %unsqueeze_6), kwargs = {})
#   %select_7 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_144, 2, 1), kwargs = {})
#   %slice_25 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg11_1, 1, 1024, 2048), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_25, 0), kwargs = {})
#   %mul_79 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %unsqueeze_7), kwargs = {})
#   %sub_4 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_78, %mul_79), kwargs = {})
#   %mul_80 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %unsqueeze_7), kwargs = {})
#   %mul_81 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %unsqueeze_6), kwargs = {})
#   %add_46 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %mul_81), kwargs = {})
#   %cat_6 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_4, %add_46], 2), kwargs = {})
#   %convert_element_type_199 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_57, torch.float32), kwargs = {})
#   %sigmoid_5 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_199,), kwargs = {})
#   %mul_82 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_199, %sigmoid_5), kwargs = {})
#   %convert_element_type_200 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_82, torch.bfloat16), kwargs = {})
#   %mul_83 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_200, %bmm_58), kwargs = {})
#   %slice_31 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_83, 1, 0, 96), kwargs = {})
#   %convert_element_type_201 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_31, torch.float32), kwargs = {})
#   %view_147 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_201, [32, 48, 2, 1024]), kwargs = {})
#   %select_8 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_147, 2, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_24, 0), kwargs = {})
#   %mul_84 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %unsqueeze_8), kwargs = {})
#   %select_9 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_147, 2, 1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_25, 0), kwargs = {})
#   %mul_85 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %unsqueeze_9), kwargs = {})
#   %sub_5 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_84, %mul_85), kwargs = {})
#   %mul_86 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %unsqueeze_9), kwargs = {})
#   %mul_87 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %unsqueeze_8), kwargs = {})
#   %add_47 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %mul_87), kwargs = {})
#   %cat_8 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_5, %add_47], 2), kwargs = {})
#   return %cat_6,%cat_8
triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_29 = async_compile.triton('triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 127401984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (1024 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (1024 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 - tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 2048, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + (1024 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.sigmoid(tmp42)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tmp45 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr2 + (1024 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp40 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tmp55 = tl.load(in_ptr4 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.sigmoid(tmp56)
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.load(in_ptr5 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp62 * tmp13
    tmp64 = tl.load(in_ptr4 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tl.sigmoid(tmp65)
    tmp67 = tmp65 * tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.load(in_ptr5 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp70 = tmp68 * tmp69
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp23
    tmp73 = tmp63 - tmp72
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp4, tmp73, tmp74)
    tmp76 = tl.load(in_ptr4 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tl.sigmoid(tmp77)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tl.load(in_ptr5 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp82 = tmp80 * tmp81
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp83 * tmp39
    tmp85 = tl.load(in_ptr4 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tl.sigmoid(tmp86)
    tmp88 = tmp86 * tmp87
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tl.load(in_ptr5 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp91 = tmp89 * tmp90
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp92 * tmp49
    tmp94 = tmp84 + tmp93
    tmp95 = tl.full(tmp94.shape, 0.0, tmp94.dtype)
    tmp96 = tl.where(tmp28, tmp94, tmp95)
    tmp97 = tl.where(tmp4, tmp75, tmp96)
    tl.store(out_ptr0 + (x3), tmp54, None)
    tl.store(out_ptr1 + (x3), tmp97, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/e7/ce7667xyj7fu6tn7n7huv6f2uhkfip7rlaewsczn6ojdgnrv44ep.py
# Topologically Sorted Source Nodes: [silu_4, hidden_1, y_8, reshape_9, y_9, getitem_46, hidden_rot_1, transpose_24, lr1i_1, mul_92, type_as_9], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.transpose]
# Source node to ATen node mapping:
#   getitem_46 => slice_32
#   hidden_1 => mul_83
#   hidden_rot_1 => cat_9
#   lr1i_1 => slice_21
#   mul_92 => mul_98
#   reshape_9 => view_149
#   silu_4 => convert_element_type_199, convert_element_type_200, mul_82, sigmoid_5
#   transpose_24 => permute_25
#   type_as_9 => convert_element_type_210
#   y_8 => view_148
#   y_9 => convert_element_type_202
# Graph fragment:
#   %cat_8 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0" = PlaceHolder[target=cat_8]
#   %bmm_57 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %bmm_58 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_58]
#   %arg7_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_199 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_57, torch.float32), kwargs = {})
#   %sigmoid_5 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_199,), kwargs = {})
#   %mul_82 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_199, %sigmoid_5), kwargs = {})
#   %convert_element_type_200 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_82, torch.bfloat16), kwargs = {})
#   %mul_83 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_200, %bmm_58), kwargs = {})
#   %view_148 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_8, [32, 48, 2, 1024]), kwargs = {})
#   %view_149 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_148, [32, 96, 1024]), kwargs = {})
#   %convert_element_type_202 : Tensor "bf16[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_149, torch.bfloat16), kwargs = {})
#   %slice_32 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_83, 1, 96, 9223372036854775807), kwargs = {})
#   %cat_9 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_202, %slice_32], 1), kwargs = {})
#   %permute_25 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%cat_9, [0, 2, 1]), kwargs = {})
#   %slice_21 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg7_1, 1, 1024, 2048), kwargs = {})
#   %mul_98 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_25, %slice_21), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_98, torch.bfloat16), kwargs = {})
#   return %convert_element_type_210
triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30 = async_compile.triton('triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331648, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp23 = tl.load(in_ptr3 + (3072 + 3*y0 + 12288*y1), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 1024*(x2) + 98304*y1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1, 1], 192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp8, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp25, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/so/cso7dpjlcm26isrolcg4xikajxinnrf7uwsnlbs634kahj4d7cc2.py
# Topologically Sorted Source Nodes: [m_i_2, m_i_3], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i_2 => slice_35
#   m_i_3 => mean_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %slice_35 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_35, [1], True), kwargs = {})
#   return %buf178
triton_per_fused_mean_slice_31 = async_compile.triton('triton_per_fused_mean_slice_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_mean_slice_31(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/uj/cujnigyl75cmek5m3ddydpbtekxjsu6jw4gwxf3vzbcuaasn4d5w.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, X_21, norm_9], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_21 => convert_element_type_219
#   dw1_1 => add_5
#   dw1_3 => add_51
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   m_i_2 => slice_35
#   m_i_3 => mean_1
#   mul_23 => mul_26
#   mul_96 => mul_102
#   norm_9 => convert_element_type_220, pow_19, sum_10
# Graph fragment:
#   %bmm_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %buf178 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf178]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %slice_35 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_35, [1], True), kwargs = {})
#   %mul_102 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %mean_1), kwargs = {})
#   %add_51 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_60, %mul_102), kwargs = {})
#   %convert_element_type_219 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.bfloat16), kwargs = {})
#   %convert_element_type_220 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_219, torch.float32), kwargs = {})
#   %pow_19 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_220, 2), kwargs = {})
#   %sum_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_19, [1, 2], True), kwargs = {})
#   return %buf179
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_32 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_32', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 4718720}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/6t/c6t63x4wsb3l5vwmauydcd2qcpcxqcaz42f66nmtvvvoosrydbvw.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, X_21, norm_9, add_53, X_22, A_15, transpose_25, matmul_47], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_15 => convert_element_type_221, convert_element_type_222
#   X_21 => convert_element_type_219
#   X_22 => div_6
#   add_53 => add_53
#   dw1_1 => add_5
#   dw1_3 => add_51
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   m_i_2 => slice_35
#   m_i_3 => mean_1
#   matmul_47 => convert_element_type_227
#   mul_23 => mul_26
#   mul_96 => mul_102
#   norm_9 => pow_20
#   transpose_25 => permute_26
# Graph fragment:
#   %bmm_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %buf178 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf178]
#   %sum_10 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_10]
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_6]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %slice_35 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_35, [1], True), kwargs = {})
#   %mul_102 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %mean_1), kwargs = {})
#   %add_51 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_60, %mul_102), kwargs = {})
#   %convert_element_type_219 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.bfloat16), kwargs = {})
#   %pow_20 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_10, 0.5), kwargs = {})
#   %add_53 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, 1e-07), kwargs = {})
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_219, %add_53), kwargs = {})
#   %convert_element_type_222 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   %permute_26 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_6, [0, 2, 1]), kwargs = {})
#   %convert_element_type_221 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_26, torch.bfloat16), kwargs = {})
#   %convert_element_type_227 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   return %div_6,%expand_90,%expand_91,%expand_95
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_33 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28311552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/jh/cjhhansccujjzcpjp2vwi3x6p2mlk6sszel77dyd4gbh6o7gb55d.py
# Topologically Sorted Source Nodes: [mul_100, X_23, A_16, transpose_26, matmul_50], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_16 => convert_element_type_230, convert_element_type_231
#   X_23 => add_55
#   matmul_50 => convert_element_type_236
#   mul_100 => mul_106
#   transpose_26 => permute_27
# Graph fragment:
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_6]
#   %bmm_65 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %mul_106 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_55 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %bmm_65), kwargs = {})
#   %convert_element_type_231 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.bfloat16), kwargs = {})
#   %permute_27 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_55, [0, 2, 1]), kwargs = {})
#   %convert_element_type_230 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_27, torch.bfloat16), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.bfloat16), kwargs = {})
#   return %expand_96,%expand_97,%expand_101
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_34(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/3c/c3co2z4ok3eqehh43pwefy2wdjw36p4qrm3qqqi2cqv7olliw3vq.py
# Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, A_17, transpose_27, matmul_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_17 => convert_element_type_239, convert_element_type_240
#   X_23 => add_55
#   X_24 => add_57
#   matmul_53 => convert_element_type_245
#   mul_100 => mul_106
#   mul_103 => mul_109
#   transpose_27 => permute_28
# Graph fragment:
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_6]
#   %bmm_65 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %bmm_68 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_68]
#   %mul_106 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_55 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %bmm_65), kwargs = {})
#   %mul_109 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 3.9505), kwargs = {})
#   %add_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %bmm_68), kwargs = {})
#   %convert_element_type_240 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.bfloat16), kwargs = {})
#   %permute_28 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_57, [0, 2, 1]), kwargs = {})
#   %convert_element_type_239 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_28, torch.bfloat16), kwargs = {})
#   %convert_element_type_245 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.bfloat16), kwargs = {})
#   return %expand_102,%expand_103,%expand_107
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/hb/chbrgtprw735zh7hopdiwrpvl2i55zwbeng7362ps7wpyrddt7jr.py
# Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_106, X_25, A_18, transpose_28, matmul_56], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_18 => convert_element_type_248, convert_element_type_249
#   X_23 => add_55
#   X_24 => add_57
#   X_25 => add_59
#   matmul_56 => convert_element_type_254
#   mul_100 => mul_106
#   mul_103 => mul_109
#   mul_106 => mul_112
#   transpose_28 => permute_29
# Graph fragment:
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_6]
#   %bmm_65 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %bmm_68 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_68]
#   %bmm_71 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_71]
#   %mul_106 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_55 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %bmm_65), kwargs = {})
#   %mul_109 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 3.9505), kwargs = {})
#   %add_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %bmm_68), kwargs = {})
#   %mul_112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, 3.7418), kwargs = {})
#   %add_59 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %bmm_71), kwargs = {})
#   %convert_element_type_249 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.bfloat16), kwargs = {})
#   %permute_29 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_59, [0, 2, 1]), kwargs = {})
#   %convert_element_type_248 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_29, torch.bfloat16), kwargs = {})
#   %convert_element_type_254 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.bfloat16), kwargs = {})
#   return %expand_108,%expand_109,%expand_113
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25952256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/su/csuugn4fzgxfhisbmsxd66nhwrgzmgu7wbrb7euaw3kogij2subx.py
# Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_106, X_25, mul_109, X_26, A_19, transpose_29, matmul_59], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_19 => convert_element_type_257, convert_element_type_258
#   X_23 => add_55
#   X_24 => add_57
#   X_25 => add_59
#   X_26 => add_61
#   matmul_59 => convert_element_type_263
#   mul_100 => mul_106
#   mul_103 => mul_109
#   mul_106 => mul_112
#   mul_109 => mul_115
#   transpose_29 => permute_30
# Graph fragment:
#   %div_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_6]
#   %bmm_65 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %bmm_68 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_68]
#   %bmm_71 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_71]
#   %bmm_74 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_74]
#   %add_61 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_61]
#   %mul_106 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 4.0848), kwargs = {})
#   %add_55 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %bmm_65), kwargs = {})
#   %mul_109 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 3.9505), kwargs = {})
#   %add_57 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %bmm_68), kwargs = {})
#   %mul_112 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, 3.7418), kwargs = {})
#   %add_59 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %bmm_71), kwargs = {})
#   %mul_115 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, 2.8769), kwargs = {})
#   %add_61 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, %bmm_74), kwargs = {})
#   %convert_element_type_258 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_61, torch.bfloat16), kwargs = {})
#   %permute_30 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_61, [0, 2, 1]), kwargs = {})
#   %convert_element_type_257 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_30, torch.bfloat16), kwargs = {})
#   %convert_element_type_263 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_61, torch.bfloat16), kwargs = {})
#   return %add_61,%expand_114,%expand_115,%expand_119
triton_poi_fused__to_copy_add_mul_transpose_37 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_37', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/6r/c6r7qao7xct6zfiz365y4hpl7zbrqwzlevv4t2wp3tklayfau4en.py
# Topologically Sorted Source Nodes: [norm_7, add_44, truediv_4, w1_norm, w1, bmm_11, mul_112, X_27, w1_main_1, norm_13, add_90, truediv_10, w1_1, bmm_20, transpose_23, dhidden_rot_1, transpose_42, dhidden_rot_2], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   X_27 => add_63
#   add_44 => add_44
#   add_90 => add_90
#   bmm_11 => convert_element_type_190
#   bmm_20 => convert_element_type_370
#   dhidden_rot_1 => convert_element_type_203
#   dhidden_rot_2 => convert_element_type_383
#   mul_112 => mul_118
#   norm_13 => pow_27, pow_28, sum_14
#   norm_7 => pow_15, pow_16, sum_8
#   transpose_23 => permute_24
#   transpose_42 => permute_43
#   truediv_10 => div_10
#   truediv_4 => div_4
#   w1 => mul_74
#   w1_1 => mul_150
#   w1_main_1 => add_86
#   w1_norm => pow_4
# Graph fragment:
#   %add_40 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_40]
#   %add_61 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_61]
#   %bmm_77 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_77]
#   %sum_14 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_14]
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_8 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_8]
#   %mul_150 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=mul_150]
#   %pow_15 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_40, 2), kwargs = {})
#   %sum_8 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_15, [2], True), kwargs = {})
#   %pow_16 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_8, 0.5), kwargs = {})
#   %add_44 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_16, 1e-05), kwargs = {})
#   %div_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_40, %add_44), kwargs = {})
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mul_74 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %pow_4), kwargs = {})
#   %convert_element_type_190 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_74, torch.bfloat16), kwargs = {})
#   %mul_118 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, 2.8366), kwargs = {})
#   %add_63 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %bmm_77), kwargs = {})
#   %add_86 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %add_63), kwargs = {})
#   %pow_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_86, 2), kwargs = {})
#   %sum_14 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_27, [2], True), kwargs = {})
#   %pow_28 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_14, 0.5), kwargs = {})
#   %add_90 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_28, 1e-05), kwargs = {})
#   %div_10 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_86, %add_90), kwargs = {})
#   %mul_150 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %pow_4), kwargs = {})
#   %convert_element_type_370 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_150, torch.bfloat16), kwargs = {})
#   %permute_24 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_74, [0, 2, 1]), kwargs = {})
#   %convert_element_type_203 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_24, torch.bfloat16), kwargs = {})
#   %permute_43 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_150, [0, 2, 1]), kwargs = {})
#   %convert_element_type_383 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_43, torch.bfloat16), kwargs = {})
#   return %sum_8,%sum_14,%mul_150,%convert_element_type_190,%convert_element_type_203,%convert_element_type_370,%convert_element_type_383
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_38 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 30670848}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
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
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp21 * tmp23
    tmp25 = libdevice.sqrt(tmp5)
    tmp26 = tmp25 + tmp19
    tmp27 = (tmp0 / tmp26)
    tmp28 = tmp27 * tmp23
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp24.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 192*x0), tmp29, r0_mask & xmask)
    tl.store(out_ptr4 + (r0_1 + 192*x0), tmp29, r0_mask & xmask)
    tl.store(out_ptr5 + (r0_1 + 192*x0), tmp30, r0_mask & xmask)
    tl.store(out_ptr6 + (r0_1 + 192*x0), tmp30, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ef/cefubpzejppbeakjldvl645ptawxzjsteejubtqmd2vpi356rtdg.py
# Topologically Sorted Source Nodes: [hci_1, hsi_1, getitem_47, float_6, x_rot_5, x1_5, c_5, mul_83, x2_5, neg_1, s__5, mul_84, sub_6, mul_85, mul_86, add_48, y_10], Original ATen: [aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.mul, aten.neg, aten.sub, aten.add, aten.stack]
# Source node to ATen node mapping:
#   add_48 => add_48
#   c_5 => unsqueeze_10
#   float_6 => convert_element_type_206
#   getitem_47 => slice_33
#   hci_1 => slice_24
#   hsi_1 => slice_25
#   mul_83 => mul_88
#   mul_84 => mul_89
#   mul_85 => mul_90
#   mul_86 => mul_91
#   neg_1 => neg_1
#   s__5 => unsqueeze_11
#   sub_6 => sub_6
#   x1_5 => select_10
#   x2_5 => select_11
#   x_rot_5 => view_150
#   y_10 => cat_10
# Graph fragment:
#   %bmm_59 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %arg10_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %slice_24 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg10_1, 1, 1024, 2048), kwargs = {})
#   %slice_25 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg11_1, 1, 1024, 2048), kwargs = {})
#   %slice_33 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_59, 1, 0, 96), kwargs = {})
#   %convert_element_type_206 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_33, torch.float32), kwargs = {})
#   %view_150 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_206, [32, 48, 2, 1024]), kwargs = {})
#   %select_10 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_150, 2, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_24, 0), kwargs = {})
#   %mul_88 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %unsqueeze_10), kwargs = {})
#   %select_11 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_150, 2, 1), kwargs = {})
#   %neg_1 : Tensor "f32[48, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_25,), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 48, 1024][49152, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%neg_1, 0), kwargs = {})
#   %mul_89 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %unsqueeze_11), kwargs = {})
#   %sub_6 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_88, %mul_89), kwargs = {})
#   %mul_90 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %unsqueeze_11), kwargs = {})
#   %mul_91 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %unsqueeze_10), kwargs = {})
#   %add_48 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_91), kwargs = {})
#   %cat_10 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_6, %add_48], 2), kwargs = {})
#   return %cat_10
triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39 = async_compile.triton('triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 45613056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (1024 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (1024 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = -tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 - tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 2048, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr2 + (1024 + 4096*x1 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp23 = -tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (1024 + 4096*x1 + ((-1024) + x0)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp24 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp17, tmp29, tmp30)
    tmp32 = tl.where(tmp4, tmp16, tmp31)
    tl.store(out_ptr0 + (x3), tmp32, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/5o/c5o5vgy2szucfkkdlhbt3ysl54rvffqu7mtfjovnvvtujsbacxds.py
# Topologically Sorted Source Nodes: [ki_1, lr0i_1, mul_93, type_as_10, lr2i_1, mul_94, type_as_11], Original ATen: [aten.slice, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   ki_1 => slice_18
#   lr0i_1 => slice_23
#   lr2i_1 => slice_22
#   mul_93 => mul_99
#   mul_94 => mul_100
#   type_as_10 => convert_element_type_213
#   type_as_11 => convert_element_type_216
# Graph fragment:
#   %arg6_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg8_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %slice_18 : Tensor "bf16[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 1024, 2048), kwargs = {})
#   %slice_23 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg9_1, 1, 1024, 2048), kwargs = {})
#   %mul_99 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_18, %slice_23), kwargs = {})
#   %convert_element_type_213 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_99, torch.bfloat16), kwargs = {})
#   %slice_22 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg8_1, 1, 1024, 2048), kwargs = {})
#   %mul_100 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_18, %slice_22), kwargs = {})
#   %convert_element_type_216 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   return %convert_element_type_213,%convert_element_type_216
triton_poi_fused__to_copy_mul_slice_40 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62914560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/k6/ck6m5bb5z7x2jhllbj7czh5jv7pxjmbeji5hygfc43l52n7hgril.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, dw2_momentum, mul_24, dw2_1, m_i_2, m_i_3, mul_95, dw0_3, X_28, norm_10, mul_97, dw2_3, X_35, norm_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_28 => convert_element_type_266
#   X_35 => convert_element_type_313
#   dw0_1 => add_4
#   dw0_3 => add_50
#   dw0_momentum => full_default_1
#   dw2_1 => add_6
#   dw2_3 => add_52
#   dw2_momentum => full_default_2
#   m_i => slice_17
#   m_i_1 => mean
#   m_i_2 => slice_35
#   m_i_3 => mean_1
#   mul_22 => mul_25
#   mul_24 => mul_27
#   mul_95 => mul_101
#   mul_97 => mul_103
#   norm_10 => convert_element_type_267, pow_21, sum_11
#   norm_11 => convert_element_type_314, pow_23, sum_12
# Graph fragment:
#   %bmm_61 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_61]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %buf178 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf178]
#   %bmm_62 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_62]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_25), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_27), kwargs = {})
#   %slice_35 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_35, [1], True), kwargs = {})
#   %mul_101 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %mean_1), kwargs = {})
#   %add_50 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_61, %mul_101), kwargs = {})
#   %convert_element_type_266 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.bfloat16), kwargs = {})
#   %convert_element_type_267 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_266, torch.float32), kwargs = {})
#   %pow_21 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_267, 2), kwargs = {})
#   %sum_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_21, [1, 2], True), kwargs = {})
#   %mul_103 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %mean_1), kwargs = {})
#   %add_52 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_62, %mul_103), kwargs = {})
#   %convert_element_type_313 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.bfloat16), kwargs = {})
#   %convert_element_type_314 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_313, torch.float32), kwargs = {})
#   %pow_23 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_314, 2), kwargs = {})
#   %sum_12 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_23, [1, 2], True), kwargs = {})
#   return %buf231,%buf282
triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41 = async_compile.triton('triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 9437440}}
)
@triton.jit
def triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/m6/cm6bh26hqfoc2mqb44hnme33pnnzhuoyl5eig5qxrpru2bvgftuy.py
# Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, dw2_momentum, mul_24, dw2_1, m_i_2, m_i_3, mul_95, dw0_3, X_28, norm_10, add_64, X_29, A_20, transpose_30, matmul_62, mul_97, dw2_3, X_35, norm_11, add_75, X_36, A_25, transpose_35, matmul_77], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
# Source node to ATen node mapping:
#   A_20 => convert_element_type_268, convert_element_type_269
#   A_25 => convert_element_type_315, convert_element_type_316
#   X_28 => convert_element_type_266
#   X_29 => div_7
#   X_35 => convert_element_type_313
#   X_36 => div_8
#   add_64 => add_64
#   add_75 => add_75
#   dw0_1 => add_4
#   dw0_3 => add_50
#   dw0_momentum => full_default_1
#   dw2_1 => add_6
#   dw2_3 => add_52
#   dw2_momentum => full_default_2
#   m_i => slice_17
#   m_i_1 => mean
#   m_i_2 => slice_35
#   m_i_3 => mean_1
#   matmul_62 => convert_element_type_274
#   matmul_77 => convert_element_type_321
#   mul_22 => mul_25
#   mul_24 => mul_27
#   mul_95 => mul_101
#   mul_97 => mul_103
#   norm_10 => pow_22
#   norm_11 => pow_24
#   transpose_30 => permute_31
#   transpose_35 => permute_36
# Graph fragment:
#   %bmm_61 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_61]
#   %bmm_7 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %buf178 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf178]
#   %sum_11 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_11]
#   %div_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_7]
#   %bmm_62 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_62]
#   %bmm_8 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %sum_12 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_12]
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=div_8]
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %full_default_1 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %mean), kwargs = {})
#   %add_4 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_7, %mul_25), kwargs = {})
#   %full_default_2 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_27 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %mean), kwargs = {})
#   %add_6 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_8, %mul_27), kwargs = {})
#   %slice_35 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_35, [1], True), kwargs = {})
#   %mul_101 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %mean_1), kwargs = {})
#   %add_50 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_61, %mul_101), kwargs = {})
#   %convert_element_type_266 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.bfloat16), kwargs = {})
#   %pow_22 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_11, 0.5), kwargs = {})
#   %add_64 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_22, 1e-07), kwargs = {})
#   %div_7 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_266, %add_64), kwargs = {})
#   %convert_element_type_269 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.bfloat16), kwargs = {})
#   %permute_31 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_7, [0, 2, 1]), kwargs = {})
#   %convert_element_type_268 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_31, torch.bfloat16), kwargs = {})
#   %convert_element_type_274 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.bfloat16), kwargs = {})
#   %mul_103 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %mean_1), kwargs = {})
#   %add_52 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_62, %mul_103), kwargs = {})
#   %convert_element_type_313 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.bfloat16), kwargs = {})
#   %pow_24 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_12, 0.5), kwargs = {})
#   %add_75 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_24, 1e-07), kwargs = {})
#   %div_8 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_313, %add_75), kwargs = {})
#   %convert_element_type_316 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.bfloat16), kwargs = {})
#   %permute_36 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_8, [0, 2, 1]), kwargs = {})
#   %convert_element_type_315 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_36, torch.bfloat16), kwargs = {})
#   %convert_element_type_321 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.bfloat16), kwargs = {})
#   return %div_7,%expand_120,%expand_121,%expand_125,%div_8,%expand_150,%expand_151,%expand_155
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 56623104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/ae/caeq2acz4etoz7mquqvmlqyvjiisn2m6taptyqjuqgvnvyhfye4a.py
# Topologically Sorted Source Nodes: [w0_norm, mul_127, X_34, w0_main_1, norm_12, add_89, truediv_9, w0_1, bmm_19, gate_before_act_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_34 => add_74
#   add_89 => add_89
#   bmm_19 => convert_element_type_363
#   gate_before_act_2 => convert_element_type_373
#   mul_127 => mul_133
#   norm_12 => pow_25, pow_26, sum_13
#   truediv_9 => div_9
#   w0_1 => mul_149
#   w0_main_1 => add_87
#   w0_norm => pow_2
# Graph fragment:
#   %add_41 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_41]
#   %add_72 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_72]
#   %bmm_92 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_92]
#   %sum_13 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_13]
#   %sum_1 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_1]
#   %mul_149 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=mul_149]
#   %pow_2 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mul_133 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_72, 2.8366), kwargs = {})
#   %add_74 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_133, %bmm_92), kwargs = {})
#   %add_87 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %add_74), kwargs = {})
#   %pow_25 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_87, 2), kwargs = {})
#   %sum_13 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_25, [2], True), kwargs = {})
#   %pow_26 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_13, 0.5), kwargs = {})
#   %add_89 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_26, 1e-05), kwargs = {})
#   %div_9 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_87, %add_89), kwargs = {})
#   %mul_149 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %pow_2), kwargs = {})
#   %convert_element_type_363 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_149, torch.bfloat16), kwargs = {})
#   %convert_element_type_373 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_149, torch.bfloat16), kwargs = {})
#   return %sum_13,%mul_149,%convert_element_type_363,%convert_element_type_373
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_43 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_43', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 21233664}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/3r/c3rp6zjgq2voenipnfmp7tbua6vfebf46bznunpwjkpku35sn2z7.py
# Topologically Sorted Source Nodes: [gate_2, mul_146, getitem_62, float_7, x_rot_6, x1_6, hci_2, c_6, mul_147, x2_6, hsi_2, s__6, mul_148, sub_8, mul_149, mul_150, add_92, y_12, silu_7, hidden_2, getitem_68, float_8, x_rot_7, x1_7, c_7, mul_152, x2_7, s__7, mul_153, sub_9, mul_154, mul_155, add_93, y_14, getitem_74, float_9, x_rot_8, x1_8, c_8, mul_156, x2_8, neg_2, s__8, mul_157, sub_10, mul_158, mul_159, add_94, y_16], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack, aten.neg]
# Source node to ATen node mapping:
#   add_92 => add_92
#   add_93 => add_93
#   add_94 => add_94
#   c_6 => unsqueeze_12
#   c_7 => unsqueeze_14
#   c_8 => unsqueeze_16
#   float_7 => convert_element_type_368
#   float_8 => convert_element_type_381
#   float_9 => convert_element_type_386
#   gate_2 => convert_element_type_366, convert_element_type_367, mul_152, sigmoid_8
#   getitem_62 => slice_44
#   getitem_68 => slice_49
#   getitem_74 => slice_51
#   hci_2 => slice_42
#   hidden_2 => mul_159
#   hsi_2 => slice_43
#   mul_146 => mul_153
#   mul_147 => mul_154
#   mul_148 => mul_155
#   mul_149 => mul_156
#   mul_150 => mul_157
#   mul_152 => mul_160
#   mul_153 => mul_161
#   mul_154 => mul_162
#   mul_155 => mul_163
#   mul_156 => mul_164
#   mul_157 => mul_165
#   mul_158 => mul_166
#   mul_159 => mul_167
#   neg_2 => neg_2
#   s__6 => unsqueeze_13
#   s__7 => unsqueeze_15
#   s__8 => unsqueeze_17
#   silu_7 => convert_element_type_379, convert_element_type_380, mul_158, sigmoid_9
#   sub_10 => sub_10
#   sub_8 => sub_8
#   sub_9 => sub_9
#   x1_6 => select_12
#   x1_7 => select_14
#   x1_8 => select_16
#   x2_6 => select_13
#   x2_7 => select_15
#   x2_8 => select_17
#   x_rot_6 => view_288
#   x_rot_7 => view_291
#   x_rot_8 => view_294
#   y_12 => cat_12
#   y_14 => cat_14
#   y_16 => cat_16
# Graph fragment:
#   %bmm_109 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_109]
#   %bmm_108 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_108]
#   %arg10_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %bmm_111 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_111]
#   %bmm_112 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %bmm_113 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_113]
#   %convert_element_type_366 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_109, torch.float32), kwargs = {})
#   %sigmoid_8 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_366,), kwargs = {})
#   %mul_152 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_366, %sigmoid_8), kwargs = {})
#   %convert_element_type_367 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_152, torch.bfloat16), kwargs = {})
#   %mul_153 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_367, %bmm_108), kwargs = {})
#   %slice_44 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_153, 1, 0, 96), kwargs = {})
#   %convert_element_type_368 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_44, torch.float32), kwargs = {})
#   %view_288 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_368, [32, 48, 2, 1024]), kwargs = {})
#   %select_12 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_288, 2, 0), kwargs = {})
#   %slice_42 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg10_1, 1, 2048, 3072), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_42, 0), kwargs = {})
#   %mul_154 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %unsqueeze_12), kwargs = {})
#   %select_13 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_288, 2, 1), kwargs = {})
#   %slice_43 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg11_1, 1, 2048, 3072), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_43, 0), kwargs = {})
#   %mul_155 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %unsqueeze_13), kwargs = {})
#   %sub_8 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_154, %mul_155), kwargs = {})
#   %mul_156 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %unsqueeze_13), kwargs = {})
#   %mul_157 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %unsqueeze_12), kwargs = {})
#   %add_92 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %mul_157), kwargs = {})
#   %cat_12 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_8, %add_92], 2), kwargs = {})
#   %convert_element_type_379 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_111, torch.float32), kwargs = {})
#   %sigmoid_9 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_379,), kwargs = {})
#   %mul_158 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_379, %sigmoid_9), kwargs = {})
#   %convert_element_type_380 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_158, torch.bfloat16), kwargs = {})
#   %mul_159 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_380, %bmm_112), kwargs = {})
#   %slice_49 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_159, 1, 0, 96), kwargs = {})
#   %convert_element_type_381 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_49, torch.float32), kwargs = {})
#   %view_291 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_381, [32, 48, 2, 1024]), kwargs = {})
#   %select_14 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_291, 2, 0), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_42, 0), kwargs = {})
#   %mul_160 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %unsqueeze_14), kwargs = {})
#   %select_15 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_291, 2, 1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_43, 0), kwargs = {})
#   %mul_161 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %unsqueeze_15), kwargs = {})
#   %sub_9 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_160, %mul_161), kwargs = {})
#   %mul_162 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %unsqueeze_15), kwargs = {})
#   %mul_163 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %unsqueeze_14), kwargs = {})
#   %add_93 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_162, %mul_163), kwargs = {})
#   %cat_14 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_9, %add_93], 2), kwargs = {})
#   %slice_51 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_113, 1, 0, 96), kwargs = {})
#   %convert_element_type_386 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_51, torch.float32), kwargs = {})
#   %view_294 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_386, [32, 48, 2, 1024]), kwargs = {})
#   %select_16 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_294, 2, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_42, 0), kwargs = {})
#   %mul_164 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %unsqueeze_16), kwargs = {})
#   %select_17 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_294, 2, 1), kwargs = {})
#   %neg_2 : Tensor "f32[48, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_43,), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 48, 1024][49152, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%neg_2, 0), kwargs = {})
#   %mul_165 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %unsqueeze_17), kwargs = {})
#   %sub_10 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_164, %mul_165), kwargs = {})
#   %mul_166 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %unsqueeze_17), kwargs = {})
#   %mul_167 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %unsqueeze_16), kwargs = {})
#   %add_94 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %mul_167), kwargs = {})
#   %cat_16 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_10, %add_94], 2), kwargs = {})
#   return %cat_12,%cat_14,%cat_16
triton_poi_fused__to_copy_add_mul_neg_select_silu_slice_stack_sub_unsqueeze_view_44 = async_compile.triton('triton_poi_fused__to_copy_add_mul_neg_select_silu_slice_stack_sub_unsqueeze_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_select_silu_slice_stack_sub_unsqueeze_view_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 171442176}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_neg_select_silu_slice_stack_sub_unsqueeze_view_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (2048 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (2048 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 - tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 2048, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + (2048 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.sigmoid(tmp42)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tmp45 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr2 + (2048 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp40 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tmp55 = tl.load(in_ptr4 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.sigmoid(tmp56)
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.load(in_ptr5 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp62 * tmp13
    tmp64 = tl.load(in_ptr4 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tl.sigmoid(tmp65)
    tmp67 = tmp65 * tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.load(in_ptr5 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp70 = tmp68 * tmp69
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp23
    tmp73 = tmp63 - tmp72
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp4, tmp73, tmp74)
    tmp76 = tl.load(in_ptr4 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tl.sigmoid(tmp77)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tl.load(in_ptr5 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp82 = tmp80 * tmp81
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp83 * tmp39
    tmp85 = tl.load(in_ptr4 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tl.sigmoid(tmp86)
    tmp88 = tmp86 * tmp87
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tl.load(in_ptr5 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp91 = tmp89 * tmp90
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp92 * tmp49
    tmp94 = tmp84 + tmp93
    tmp95 = tl.full(tmp94.shape, 0.0, tmp94.dtype)
    tmp96 = tl.where(tmp28, tmp94, tmp95)
    tmp97 = tl.where(tmp4, tmp75, tmp96)
    tmp98 = tl.load(in_ptr6 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp99 * tmp13
    tmp101 = tl.load(in_ptr6 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp102 = tmp101.to(tl.float32)
    tmp103 = -tmp23
    tmp104 = tmp102 * tmp103
    tmp105 = tmp100 - tmp104
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp4, tmp105, tmp106)
    tmp108 = tl.load(in_ptr6 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp109 = tmp108.to(tl.float32)
    tmp110 = -tmp39
    tmp111 = tmp109 * tmp110
    tmp112 = tl.load(in_ptr6 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp113 = tmp112.to(tl.float32)
    tmp114 = tmp113 * tmp49
    tmp115 = tmp111 + tmp114
    tmp116 = tl.full(tmp115.shape, 0.0, tmp115.dtype)
    tmp117 = tl.where(tmp28, tmp115, tmp116)
    tmp118 = tl.where(tmp4, tmp107, tmp117)
    tl.store(out_ptr0 + (x3), tmp54, None)
    tl.store(out_ptr1 + (x3), tmp97, None)
    tl.store(out_ptr2 + (x3), tmp118, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/h4/ch4mu2vp76xtfrftmsqvpm2gd3zyrnjnorsjzkxi3b75niqku5fr.py
# Topologically Sorted Source Nodes: [silu_7, hidden_2, y_14, reshape_15, y_15, getitem_73, hidden_rot_2, transpose_43, lr1i_2, mul_165, type_as_15], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.transpose]
# Source node to ATen node mapping:
#   getitem_73 => slice_50
#   hidden_2 => mul_159
#   hidden_rot_2 => cat_15
#   lr1i_2 => slice_39
#   mul_165 => mul_174
#   reshape_15 => view_293
#   silu_7 => convert_element_type_379, convert_element_type_380, mul_158, sigmoid_9
#   transpose_43 => permute_44
#   type_as_15 => convert_element_type_390
#   y_14 => view_292
#   y_15 => convert_element_type_382
# Graph fragment:
#   %cat_14 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0" = PlaceHolder[target=cat_14]
#   %bmm_111 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_111]
#   %bmm_112 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %arg7_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_379 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_111, torch.float32), kwargs = {})
#   %sigmoid_9 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_379,), kwargs = {})
#   %mul_158 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_379, %sigmoid_9), kwargs = {})
#   %convert_element_type_380 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_158, torch.bfloat16), kwargs = {})
#   %mul_159 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_380, %bmm_112), kwargs = {})
#   %view_292 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_14, [32, 48, 2, 1024]), kwargs = {})
#   %view_293 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_292, [32, 96, 1024]), kwargs = {})
#   %convert_element_type_382 : Tensor "bf16[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_293, torch.bfloat16), kwargs = {})
#   %slice_50 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_159, 1, 96, 9223372036854775807), kwargs = {})
#   %cat_15 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_382, %slice_50], 1), kwargs = {})
#   %permute_44 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%cat_15, [0, 2, 1]), kwargs = {})
#   %slice_39 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg7_1, 1, 2048, 3072), kwargs = {})
#   %mul_174 : Tensor "f32[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_44, %slice_39), kwargs = {})
#   %convert_element_type_390 : Tensor "bf16[32, 1024, 192][196608, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_174, torch.bfloat16), kwargs = {})
#   return %convert_element_type_390
triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_45 = async_compile.triton('triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331648, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp23 = tl.load(in_ptr3 + (6144 + 3*y0 + 12288*y1), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 1024*(x2) + 98304*y1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1, 1], 192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (98304 + y0 + 1024*((-96) + x2) + 196608*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp8, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 192*y3), tmp25, xmask)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/7c/c7c2ljd4dlz3c57safvklqznwhi7ffs5ltnujenu4svdhm5p75ba.py
# Topologically Sorted Source Nodes: [m_i_4, m_i_5], Original ATen: [aten.slice, aten.mean]
# Source node to ATen node mapping:
#   m_i_4 => slice_53
#   m_i_5 => mean_2
# Graph fragment:
#   %arg3_1 : Tensor "f32[32, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %slice_53 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 2048, 3072), kwargs = {})
#   %mean_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_53, [1], True), kwargs = {})
#   return %buf341
triton_per_fused_mean_slice_46 = async_compile.triton('triton_per_fused_mean_slice_46', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_slice_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_mean_slice_46(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/uw/cuwtfbnpnijcggsbaxakyanvuswht4vsnbiof5frsdkjvk7ar6xr.py
# Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, m_i_4, m_i_5, mul_169, dw1_5, X_42], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   X_42 => convert_element_type_399
#   dw1_1 => add_5
#   dw1_3 => add_51
#   dw1_5 => add_97
#   dw1_momentum => full_default
#   m_i => slice_17
#   m_i_1 => mean
#   m_i_2 => slice_35
#   m_i_3 => mean_1
#   m_i_4 => slice_53
#   m_i_5 => mean_2
#   mul_169 => mul_178
#   mul_23 => mul_26
#   mul_96 => mul_102
# Graph fragment:
#   %bmm_114 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_114]
#   %bmm_60 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_60]
#   %bmm_6 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_6]
#   %buf15 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf15]
#   %buf178 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf178]
#   %buf341 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=buf341]
#   %full_default : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 192, 192], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_17 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 0, 1024), kwargs = {})
#   %mean : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_17, [1], True), kwargs = {})
#   %mul_26 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mean), kwargs = {})
#   %add_5 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_6, %mul_26), kwargs = {})
#   %slice_35 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 1024, 2048), kwargs = {})
#   %mean_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_35, [1], True), kwargs = {})
#   %mul_102 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %mean_1), kwargs = {})
#   %add_51 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_60, %mul_102), kwargs = {})
#   %slice_53 : Tensor "f32[32, 1024, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 1, 2048, 3072), kwargs = {})
#   %mean_2 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_53, [1], True), kwargs = {})
#   %mul_178 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, %mean_2), kwargs = {})
#   %add_97 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_114, %mul_178), kwargs = {})
#   %convert_element_type_399 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_97, torch.bfloat16), kwargs = {})
#   return %convert_element_type_399
triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47 = async_compile.triton('triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/x7/cx7jtygx772dbgyqjll5i3xq2queyrymf5krx5xlr2vtdygqgolk.py
# Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm_15 => convert_element_type_400, pow_31, sum_16
# Graph fragment:
#   %convert_element_type_399 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_399]
#   %convert_element_type_400 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_399, torch.float32), kwargs = {})
#   %pow_31 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_400, 2), kwargs = {})
#   %sum_16 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_31, [1, 2], True), kwargs = {})
#   return %buf343
triton_red_fused_linalg_vector_norm_48 = async_compile.triton('triton_red_fused_linalg_vector_norm_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280, 'r0_': 2359360}}
)
@triton.jit
def triton_red_fused_linalg_vector_norm_48(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/db/cdbx7gwx5g5ddjvzi4xd4oqmmcq5kjkuhupcuhukpsem5ialoy55.py
# Topologically Sorted Source Nodes: [norm_15, add_99, X_43, A_30, transpose_44, matmul_92], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_30 => convert_element_type_401, convert_element_type_402
#   X_43 => div_12
#   add_99 => add_99
#   matmul_92 => convert_element_type_407
#   norm_15 => pow_32
#   transpose_44 => permute_45
# Graph fragment:
#   %convert_element_type_399 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_399]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_99 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_399, %add_99), kwargs = {})
#   %convert_element_type_402 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.bfloat16), kwargs = {})
#   %permute_45 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_12, [0, 2, 1]), kwargs = {})
#   %convert_element_type_401 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_45, torch.bfloat16), kwargs = {})
#   %convert_element_type_407 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.bfloat16), kwargs = {})
#   return %expand_180,%expand_181,%expand_185
triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16515072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/wj/cwjsvo6gl2m2wqa6tzxjan2q6meup6ylkkaymlavopymyvmdcgvi.py
# Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, A_31, transpose_45, matmul_95], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_31 => convert_element_type_410, convert_element_type_411
#   X_43 => div_12
#   X_44 => add_101
#   add_99 => add_99
#   matmul_95 => convert_element_type_416
#   mul_173 => mul_182
#   norm_15 => pow_32
#   transpose_45 => permute_46
# Graph fragment:
#   %convert_element_type_399 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_399]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %bmm_119 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_99 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_399, %add_99), kwargs = {})
#   %mul_182 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, 4.0848), kwargs = {})
#   %add_101 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %bmm_119), kwargs = {})
#   %convert_element_type_411 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_101, torch.bfloat16), kwargs = {})
#   %permute_46 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_101, [0, 2, 1]), kwargs = {})
#   %convert_element_type_410 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_46, torch.bfloat16), kwargs = {})
#   %convert_element_type_416 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_101, torch.bfloat16), kwargs = {})
#   return %expand_186,%expand_187,%expand_191
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/bm/cbmop5ptjq6hlwnieunlpwg33xu77csflf7bgqysax2rpw7xmjqt.py
# Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_176, X_45, A_32, transpose_46, matmul_98], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_32 => convert_element_type_419, convert_element_type_420
#   X_43 => div_12
#   X_44 => add_101
#   X_45 => add_103
#   add_99 => add_99
#   matmul_98 => convert_element_type_425
#   mul_173 => mul_182
#   mul_176 => mul_185
#   norm_15 => pow_32
#   transpose_46 => permute_47
# Graph fragment:
#   %convert_element_type_399 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_399]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %bmm_119 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %bmm_122 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_122]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_99 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_399, %add_99), kwargs = {})
#   %mul_182 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, 4.0848), kwargs = {})
#   %add_101 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %bmm_119), kwargs = {})
#   %mul_185 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_101, 3.9505), kwargs = {})
#   %add_103 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %bmm_122), kwargs = {})
#   %convert_element_type_420 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_103, torch.bfloat16), kwargs = {})
#   %permute_47 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_103, [0, 2, 1]), kwargs = {})
#   %convert_element_type_419 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_47, torch.bfloat16), kwargs = {})
#   %convert_element_type_425 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_103, torch.bfloat16), kwargs = {})
#   return %expand_192,%expand_193,%expand_197
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/k5/ck55mpzz4yrsi5ah4ebdryucnxgrci5nviexfdg7nuewmwjewx4m.py
# Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_176, X_45, mul_179, X_46, A_33, transpose_47, matmul_101], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_33 => convert_element_type_428, convert_element_type_429
#   X_43 => div_12
#   X_44 => add_101
#   X_45 => add_103
#   X_46 => add_105
#   add_99 => add_99
#   matmul_101 => convert_element_type_434
#   mul_173 => mul_182
#   mul_176 => mul_185
#   mul_179 => mul_188
#   norm_15 => pow_32
#   transpose_47 => permute_48
# Graph fragment:
#   %convert_element_type_399 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=convert_element_type_399]
#   %sum_16 : Tensor "f32[32, 1, 1][1, 32, 32]cuda:0" = PlaceHolder[target=sum_16]
#   %bmm_119 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %bmm_122 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_122]
#   %bmm_125 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_125]
#   %add_105 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_105]
#   %pow_32 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %add_99 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_32, 1e-07), kwargs = {})
#   %div_12 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_399, %add_99), kwargs = {})
#   %mul_182 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, 4.0848), kwargs = {})
#   %add_101 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %bmm_119), kwargs = {})
#   %mul_185 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_101, 3.9505), kwargs = {})
#   %add_103 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %bmm_122), kwargs = {})
#   %mul_188 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, 3.7418), kwargs = {})
#   %add_105 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %bmm_125), kwargs = {})
#   %convert_element_type_429 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_105, torch.bfloat16), kwargs = {})
#   %permute_48 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_105, [0, 2, 1]), kwargs = {})
#   %convert_element_type_428 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_48, torch.bfloat16), kwargs = {})
#   %convert_element_type_434 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_105, torch.bfloat16), kwargs = {})
#   return %add_105,%expand_198,%expand_199,%expand_203
triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33030144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/6u/c6u5l442g3ynb6aacyz2vyp6ccq5yytn6jj3gxmk57cq6xgie6bp.py
# Topologically Sorted Source Nodes: [mul_182, X_47, A_34, transpose_48, matmul_104], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
# Source node to ATen node mapping:
#   A_34 => convert_element_type_437, convert_element_type_438
#   X_47 => add_107
#   matmul_104 => convert_element_type_443
#   mul_182 => mul_191
#   transpose_48 => permute_49
# Graph fragment:
#   %add_105 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_105]
#   %bmm_128 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_128]
#   %mul_191 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, 2.8769), kwargs = {})
#   %add_107 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %bmm_128), kwargs = {})
#   %convert_element_type_438 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_107, torch.bfloat16), kwargs = {})
#   %permute_49 : Tensor "f32[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_107, [0, 2, 1]), kwargs = {})
#   %convert_element_type_437 : Tensor "bf16[32, 192, 192][36864, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_49, torch.bfloat16), kwargs = {})
#   %convert_element_type_443 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_107, torch.bfloat16), kwargs = {})
#   return %expand_204,%expand_205,%expand_209
triton_poi_fused__to_copy_add_mul_transpose_53 = async_compile.triton('triton_poi_fused__to_copy_add_mul_transpose_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_transpose_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21233664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_transpose_53(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/qd/cqdpb4nyptyr2vv53pkgdm32sccji37mfcvvqpjc62ugalemlh4v.py
# Topologically Sorted Source Nodes: [w1_norm, mul_112, X_27, w1_main_1, mul_182, X_47, mul_185, X_48, w1_main_2, norm_19, add_136, truediv_16, w1_2, bmm_29], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   X_27 => add_63
#   X_47 => add_107
#   X_48 => add_109
#   add_136 => add_136
#   bmm_29 => convert_element_type_550
#   mul_112 => mul_118
#   mul_182 => mul_191
#   mul_185 => mul_194
#   norm_19 => pow_39, pow_40, sum_20
#   truediv_16 => div_16
#   w1_2 => mul_226
#   w1_main_1 => add_86
#   w1_main_2 => add_132
#   w1_norm => pow_4
# Graph fragment:
#   %add_40 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_40]
#   %add_61 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_61]
#   %bmm_77 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_77]
#   %add_105 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_105]
#   %bmm_128 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_128]
#   %bmm_131 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=bmm_131]
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0" = PlaceHolder[target=add_132]
#   %sum_20 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_20]
#   %sum_2 : Tensor "f32[32, 192, 1][192, 1, 6144]cuda:0" = PlaceHolder[target=sum_2]
#   %pow_4 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mul_118 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, 2.8366), kwargs = {})
#   %add_63 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %bmm_77), kwargs = {})
#   %add_86 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %add_63), kwargs = {})
#   %mul_191 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, 2.8769), kwargs = {})
#   %add_107 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %bmm_128), kwargs = {})
#   %mul_194 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_107, 2.8366), kwargs = {})
#   %add_109 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %bmm_131), kwargs = {})
#   %add_132 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %add_109), kwargs = {})
#   %pow_39 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_132, 2), kwargs = {})
#   %sum_20 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_39, [2], True), kwargs = {})
#   %pow_40 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_20, 0.5), kwargs = {})
#   %add_136 : Tensor "f32[32, 192, 1][192, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_40, 1e-05), kwargs = {})
#   %div_16 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_132, %add_136), kwargs = {})
#   %mul_226 : Tensor "f32[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %pow_4), kwargs = {})
#   %convert_element_type_550 : Tensor "bf16[32, 192, 192][36864, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_226, torch.bfloat16), kwargs = {})
#   return %add_132,%sum_20,%convert_element_type_550
triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54 = async_compile.triton('triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 25952256}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/qs/cqs2cu5pzaxjbhpi5bwakoi37tjpiaffjpuwd4vpbr2azp5o3eo3.py
# Topologically Sorted Source Nodes: [ki_2, lr0i_2, mul_166, type_as_16, lr2i_2, mul_167, type_as_17], Original ATen: [aten.slice, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   ki_2 => slice_36
#   lr0i_2 => slice_41
#   lr2i_2 => slice_40
#   mul_166 => mul_175
#   mul_167 => mul_176
#   type_as_16 => convert_element_type_393
#   type_as_17 => convert_element_type_396
# Graph fragment:
#   %arg6_1 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg8_1 : Tensor "f32[32, 4096, 1][12288, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %slice_36 : Tensor "bf16[32, 1024, 192][786432, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 2048, 3072), kwargs = {})
#   %slice_41 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg9_1, 1, 2048, 3072), kwargs = {})
#   %mul_175 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_36, %slice_41), kwargs = {})
#   %convert_element_type_393 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_175, torch.bfloat16), kwargs = {})
#   %slice_40 : Tensor "f32[32, 1024, 1][12288, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg8_1, 1, 2048, 3072), kwargs = {})
#   %mul_176 : Tensor "f32[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_36, %slice_40), kwargs = {})
#   %convert_element_type_396 : Tensor "bf16[32, 1024, 192][196608, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_176, torch.bfloat16), kwargs = {})
#   return %convert_element_type_393,%convert_element_type_396
triton_poi_fused__to_copy_mul_slice_55 = async_compile.triton('triton_poi_fused__to_copy_mul_slice_55', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_slice_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62914560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_slice_55(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/tw/ctwzuib7wgtgk6shntswj43zglduq6fh5bx53qqum62ct44jhyo4.py
# Topologically Sorted Source Nodes: [gate_3, mul_219, getitem_84, float_10, x_rot_9, x1_9, getitem_82, c_9, mul_220, x2_9, getitem_83, s__9, mul_221, sub_12, mul_222, mul_223, add_138, y_18], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack]
# Source node to ATen node mapping:
#   add_138 => add_138
#   c_9 => unsqueeze_18
#   float_10 => convert_element_type_548
#   gate_3 => convert_element_type_546, convert_element_type_547, mul_228, sigmoid_12
#   getitem_82 => slice_55
#   getitem_83 => slice_56
#   getitem_84 => slice_57
#   mul_219 => mul_229
#   mul_220 => mul_230
#   mul_221 => mul_231
#   mul_222 => mul_232
#   mul_223 => mul_233
#   s__9 => unsqueeze_19
#   sub_12 => sub_12
#   x1_9 => select_18
#   x2_9 => select_19
#   x_rot_9 => view_432
#   y_18 => cat_18
# Graph fragment:
#   %bmm_163 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_163]
#   %bmm_162 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_162]
#   %arg10_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[48, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convert_element_type_546 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_163, torch.float32), kwargs = {})
#   %sigmoid_12 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_546,), kwargs = {})
#   %mul_228 : Tensor "f32[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_546, %sigmoid_12), kwargs = {})
#   %convert_element_type_547 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_228, torch.bfloat16), kwargs = {})
#   %mul_229 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_547, %bmm_162), kwargs = {})
#   %slice_57 : Tensor "bf16[32, 96, 1024][196608, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_229, 1, 0, 96), kwargs = {})
#   %convert_element_type_548 : Tensor "f32[32, 96, 1024][98304, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_57, torch.float32), kwargs = {})
#   %view_432 : Tensor "f32[32, 48, 2, 1024][98304, 2048, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_548, [32, 48, 2, 1024]), kwargs = {})
#   %select_18 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_432, 2, 0), kwargs = {})
#   %slice_55 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg10_1, 1, 3072, 4096), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_55, 0), kwargs = {})
#   %mul_230 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %unsqueeze_18), kwargs = {})
#   %select_19 : Tensor "f32[32, 48, 1024][98304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%view_432, 2, 1), kwargs = {})
#   %slice_56 : Tensor "f32[48, 1024][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg11_1, 1, 3072, 4096), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 48, 1024][196608, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_56, 0), kwargs = {})
#   %mul_231 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %unsqueeze_19), kwargs = {})
#   %sub_12 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_230, %mul_231), kwargs = {})
#   %mul_232 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %unsqueeze_19), kwargs = {})
#   %mul_233 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %unsqueeze_18), kwargs = {})
#   %add_138 : Tensor "f32[32, 48, 1024][49152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %mul_233), kwargs = {})
#   %cat_18 : Tensor "f32[32, 48, 2048][98304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_12, %add_138], 2), kwargs = {})
#   return %cat_18
triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56 = async_compile.triton('triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64487424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 48)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (3072 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (3072 + 4096*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 - tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 2048, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr0 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + (3072 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr0 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.sigmoid(tmp42)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (1024 + 2048*x1 + 196608*x2 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tmp45 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr2 + (3072 + 4096*x1 + ((-1024) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp40 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x3), tmp54, None)
''', device_str='cuda')


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/v3/cv3tzoabpoqaqrozyntcbhcw7po6gvyp4d725qdzezlxit7uuqeg.py
# Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   output => full_3, permute_2
#   setitem => copy, slice_11
#   setitem_1 => copy_1, slice_29
#   setitem_2 => copy_2, slice_47
#   setitem_3 => copy_3, slice_60
# Graph fragment:
#   %bmm_164 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_164]
#   %bmm_110 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_110]
#   %bmm_56 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_56]
#   %bmm_2 : Tensor "bf16[32, 192, 1024][196608, 1024, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %full_3 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_3, [0, 2, 1]), kwargs = {})
#   %slice_11 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 1024), kwargs = {})
#   %copy : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_11, %bmm_2), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_2, %copy, 2, 0, 1024), kwargs = {})
#   %slice_29 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 1024, 2048), kwargs = {})
#   %copy_1 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_29, %bmm_56), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 1024, 2048), kwargs = {})
#   %slice_47 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_1, 2, 2048, 3072), kwargs = {})
#   %copy_2 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_47, %bmm_110), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %copy_2, 2, 2048, 3072), kwargs = {})
#   %slice_60 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 2, 3072, 4096), kwargs = {})
#   %copy_3 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_60, %bmm_164), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %copy_3, 2, 3072, 4096), kwargs = {})
#   return %slice_scatter_default_3
triton_poi_fused_copy_slice_zeros_like_57 = async_compile.triton('triton_poi_fused_copy_slice_zeros_like_57', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_like_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 301989888}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_like_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/.cache_inductor/mh/cmh624nfblvpxfj2hcjdotlp6s7woyl22tzttndzi5k6ue4cd7v3.py
# Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   output => full_3, permute_2
#   setitem => copy, slice_11
#   setitem_1 => copy_1, slice_29
#   setitem_2 => copy_2, slice_47
#   setitem_3 => copy_3, slice_60
# Graph fragment:
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 4096, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_3]
#   %full_3 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 4096, 192], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_3, [0, 2, 1]), kwargs = {})
#   %slice_11 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 1024), kwargs = {})
#   %copy : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_11, %bmm_2), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_2, %copy, 2, 0, 1024), kwargs = {})
#   %slice_29 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default, 2, 1024, 2048), kwargs = {})
#   %copy_1 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_29, %bmm_56), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 1024, 2048), kwargs = {})
#   %slice_47 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_1, 2, 2048, 3072), kwargs = {})
#   %copy_2 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_47, %bmm_110), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %copy_2, 2, 2048, 3072), kwargs = {})
#   %slice_60 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 2, 3072, 4096), kwargs = {})
#   %copy_3 : Tensor "bf16[32, 192, 1024][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_60, %bmm_164), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[32, 192, 4096][786432, 1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %copy_3, 2, 3072, 4096), kwargs = {})
#   %permute_61 : Tensor "bf16[32, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_scatter_default_3, [0, 2, 1]), kwargs = {})
#   return %permute_61
triton_poi_fused_copy_slice_zeros_like_58 = async_compile.triton('triton_poi_fused_copy_slice_zeros_like_58', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_like_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DA0E6AA63EF71E4A4BF9C5D8922226266DA6579CC4D07F5EF744CA207F7CCA1F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331648, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_like_58(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
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
        assert_size_stride(arg10_1, (48, 4096), (4096, 1))
        assert_size_stride(arg11_1, (48, 4096), (4096, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf8 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_1, gate_before_act], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg0_1, buf0, buf8, 1179648, stream=stream0)
            buf1 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [bmm_1, q, qi], Original ATen: [aten._to_copy, aten.transpose, aten.slice, aten.bmm]
            extern_kernels.bmm(buf0, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf1)
            buf2 = buf0; del buf0  # reuse
            buf10 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h, hidden_before_mul], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg2_1, buf2, buf10, 1179648, stream=stream0)
            buf3 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, qi, h], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf2, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf3)
            buf11 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ki, hidden_before_mul, transpose_3], Original ATen: [aten.slice, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf10, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf11)
            buf9 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_before_act, ki, transpose_2], Original ATen: [aten._to_copy, aten.slice, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf8, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf9)
            buf4 = empty_strided_cuda((32, 48, 2048), (98304, 2048, 1), torch.float32)
            buf12 = empty_strided_cuda((32, 48, 2048), (98304, 2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [gate, mul, getitem_8, float_1, x_rot, x1, hci, c, mul_1, x2, hsi, s_, mul_2, sub, mul_3, mul_4, add, y, silu_1, hidden, getitem_14, float_2, x_rot_1, x1_1, c_1, mul_6, x2_1, s__1, mul_7, sub_1, mul_8, mul_9, add_1, y_2], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_1.run(buf1, buf3, arg10_1, arg11_1, buf9, buf11, buf4, buf12, 3145728, stream=stream0)
            buf13 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_1, hidden, y_2, reshape_3, y_3, getitem_19, hidden_rot, transpose_5, lr1i, mul_19, type_as_3], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_2.run(buf12, buf9, buf11, arg7_1, buf13, 32768, 192, stream=stream0)
            buf14 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [v, vi, silu_1, hidden, y_2, reshape_3, y_3, getitem_19, hidden_rot, transpose_5, lr1i, mul_19, type_as_3, dw1], Original ATen: [aten.transpose, aten.slice, aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.cat, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 0), buf13, out=buf14)
            buf15 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_3.run(arg3_1, buf15, 32, 1024, stream=stream0)
            buf16 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_4.run(buf14, buf15, buf16, 160, 7373, stream=stream0)
            buf17 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf16, buf17, 32, 5, stream=stream0)
            buf18 = buf10; del buf10  # reuse
            buf19 = reinterpret_tensor(buf2, (32, 192, 192), (36864, 1, 192), 0); del buf2  # reuse
            buf24 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, A, transpose_6, matmul_2], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_6.run(buf14, buf15, buf17, buf18, buf19, buf24, 1179648, stream=stream0)
            buf20 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, A, transpose_6], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf18, buf19, out=buf20)
            buf21 = reinterpret_tensor(buf19, (32, 192, 192), (36864, 192, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [mul_26], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf20, buf21, 1179648, stream=stream0)
            buf22 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [mul_26, matmul_1], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf21, buf20, out=buf22)
            buf23 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [mul_25, B], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf23, buf22, 1179648, stream=stream0)
            buf25 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_25, B, matmul_2], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf23, buf24, out=buf25)
            buf26 = buf24; del buf24  # reuse
            buf27 = reinterpret_tensor(buf23, (32, 192, 192), (36864, 1, 192), 0); del buf23  # reuse
            buf32 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_27, X_2, A_1, transpose_7, matmul_5], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9.run(buf14, buf15, buf17, buf25, buf26, buf27, buf32, 1179648, stream=stream0)
            buf28 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_27, X_2, A_1, transpose_7], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf26, buf27, out=buf28)
            buf29 = reinterpret_tensor(buf27, (32, 192, 192), (36864, 192, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [mul_29], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf28, buf29, 1179648, stream=stream0)
            buf30 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [mul_29, matmul_4], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf29, buf28, out=buf30)
            buf31 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [mul_28, B_1], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf31, buf30, 1179648, stream=stream0)
            buf33 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_27, X_2, mul_28, B_1, matmul_5], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf31, buf32, out=buf33)
            buf34 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf35 = buf32; del buf32  # reuse
            buf36 = reinterpret_tensor(buf31, (32, 192, 192), (36864, 1, 192), 0); del buf31  # reuse
            buf41 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, X, norm_3, add_7, X_1, mul_27, X_2, mul_30, X_3, A_2, transpose_8, matmul_8], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_12.run(buf14, buf15, buf17, buf25, buf33, buf34, buf35, buf36, buf41, 1179648, stream=stream0)
            buf37 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [A_2, transpose_8], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf35, buf36, out=buf37)
            buf38 = reinterpret_tensor(buf36, (32, 192, 192), (36864, 192, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [mul_32], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf37, buf38, 1179648, stream=stream0)
            buf39 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [mul_32, matmul_7], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf38, buf37, out=buf39)
            buf40 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [mul_31, B_2], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf40, buf39, 1179648, stream=stream0)
            buf42 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [mul_31, B_2, matmul_8], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf40, buf41, out=buf42)
            buf43 = buf41; del buf41  # reuse
            buf44 = reinterpret_tensor(buf40, (32, 192, 192), (36864, 1, 192), 0); del buf40  # reuse
            buf49 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [mul_33, X_4, A_3, transpose_9, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_15.run(buf34, buf42, buf43, buf44, buf49, 1179648, stream=stream0)
            buf45 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [mul_33, X_4, A_3, transpose_9], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf43, buf44, out=buf45)
            buf46 = reinterpret_tensor(buf44, (32, 192, 192), (36864, 192, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [mul_35], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf45, buf46, 1179648, stream=stream0)
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [mul_35, matmul_10], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf46, buf45, out=buf47)
            buf48 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [mul_34, B_3], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf48, buf47, 1179648, stream=stream0)
            buf50 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [mul_33, X_4, mul_34, B_3, matmul_11], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf48, buf49, out=buf50)
            buf51 = buf49; del buf49  # reuse
            buf52 = reinterpret_tensor(buf48, (32, 192, 192), (36864, 1, 192), 0); del buf48  # reuse
            buf57 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [mul_33, X_4, mul_36, X_5, A_4, transpose_10, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_18.run(buf34, buf42, buf50, buf51, buf52, buf57, 1179648, stream=stream0)
            buf53 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_33, X_4, mul_36, X_5, A_4, transpose_10], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf51, buf52, out=buf53)
            buf54 = reinterpret_tensor(buf52, (32, 192, 192), (36864, 192, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [mul_38], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf53, buf54, 1179648, stream=stream0)
            buf55 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [mul_38, matmul_13], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf54, buf53, out=buf55)
            buf56 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [mul_37, B_4], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf56, buf55, 1179648, stream=stream0)
            buf58 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [mul_33, X_4, mul_36, X_5, mul_37, B_4, matmul_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf56, buf57, out=buf58)
            buf61 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf5 = buf57; del buf57  # reuse
            buf59 = buf34; del buf34  # reuse
            buf62 = reinterpret_tensor(buf56, (32, 192, 192), (36864, 1, 192), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [bmm_2, mul_33, X_4, mul_36, X_5, mul_39, X_6, w1_main, w1_norm, transpose_4, dhidden_rot], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.linalg_vector_norm, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_transpose_21.run(buf59, arg1_1, buf42, buf50, buf58, buf61, buf5, buf62, 6144, 192, stream=stream0)
            del arg1_1
            buf6 = reinterpret_tensor(buf13, (32, 192, 1024), (196608, 1024, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [gate, mul, y, reshape_1, y_1, getitem_13, hq_rot], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22.run(buf4, buf1, buf3, buf6, 6291456, stream=stream0)
            buf7 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [bmm_2, gate, mul, y, reshape_1, y_1, getitem_13, hq_rot], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.stack, aten.view, aten.slice, aten.cat, aten.bmm]
            extern_kernels.bmm(buf5, buf6, out=buf7)
            buf63 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [v, vi, transpose_4, dhidden_rot], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf62, reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 0), out=buf63)
            buf64 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [hci, hsi, getitem_20, float_3, x_rot_2, x1_2, c_2, mul_10, x2_2, neg, s__2, mul_11, sub_2, mul_12, mul_13, add_2, y_4], Original ATen: [aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.mul, aten.neg, aten.sub, aten.add, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_23.run(buf63, arg10_1, arg11_1, buf64, 3145728, stream=stream0)
            buf65 = buf11; del buf11  # reuse
            buf116 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [y_4, reshape_5, y_5, getitem_25, dhidden, dgate, sigma, mul_16, sub_3, mul_17, add_3, dx, silu_2, dhidden_before_mul], Original ATen: [aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24.run(buf65, buf64, buf63, buf9, buf116, 6291456, stream=stream0)
            buf66 = reinterpret_tensor(buf9, (32, 1024, 192), (196608, 192, 1), 0); del buf9  # reuse
            buf117 = reinterpret_tensor(buf63, (32, 1024, 192), (196608, 192, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [ki, lr0i, mul_20, type_as_4, lr2i, mul_21, type_as_5], Original ATen: [aten.slice, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_25.run(arg6_1, arg9_1, arg8_1, buf66, buf117, 6291456, stream=stream0)
            buf118 = reinterpret_tensor(buf62, (32, 192, 192), (36864, 192, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [ki, y_4, reshape_5, y_5, getitem_25, dhidden, silu_2, dhidden_before_mul, lr2i, mul_21, type_as_5, dw2], Original ATen: [aten.slice, aten.stack, aten.view, aten._to_copy, aten.cat, aten.silu, aten.mul, aten.bmm]
            extern_kernels.bmm(buf116, buf117, out=buf118)
            buf67 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [ki, y_4, reshape_5, y_5, getitem_25, dhidden, dgate, sigma, mul_16, sub_3, mul_17, add_3, dx, lr0i, mul_20, type_as_4, dw0], Original ATen: [aten.slice, aten.stack, aten.view, aten._to_copy, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.bmm]
            extern_kernels.bmm(buf65, buf66, out=buf67)
            buf68 = buf16; del buf16  # reuse
            buf119 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, dw2_momentum, mul_24, dw2_1, X_14, norm_5], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_26.run(buf67, buf15, buf118, buf68, buf119, 160, 7373, stream=stream0)
            buf120 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf119, buf120, 32, 5, stream=stream0)
            buf69 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf68, buf69, 32, 5, stream=stream0)
            buf70 = buf58; del buf58  # reuse
            buf71 = reinterpret_tensor(buf50, (32, 192, 192), (36864, 1, 192), 0); del buf50  # reuse
            buf76 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, A_5, transpose_11, matmul_17], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_6.run(buf67, buf15, buf69, buf70, buf71, buf76, 1179648, stream=stream0)
            buf72 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, A_5, transpose_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf70, buf71, out=buf72)
            buf73 = reinterpret_tensor(buf71, (32, 192, 192), (36864, 192, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [mul_41], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf72, buf73, 1179648, stream=stream0)
            buf74 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [mul_41, matmul_16], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf73, buf72, out=buf74)
            buf75 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [mul_40, B_5], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf75, buf74, 1179648, stream=stream0)
            buf77 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, mul_40, B_5, matmul_17], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf75, buf76, out=buf77)
            buf78 = buf76; del buf76  # reuse
            buf79 = reinterpret_tensor(buf75, (32, 192, 192), (36864, 1, 192), 0); del buf75  # reuse
            buf84 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, mul_42, X_9, A_6, transpose_12, matmul_20], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9.run(buf67, buf15, buf69, buf77, buf78, buf79, buf84, 1179648, stream=stream0)
            buf80 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, mul_42, X_9, A_6, transpose_12], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf78, buf79, out=buf80)
            buf81 = reinterpret_tensor(buf79, (32, 192, 192), (36864, 192, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [mul_44], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf80, buf81, 1179648, stream=stream0)
            buf82 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [mul_44, matmul_19], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf81, buf80, out=buf82)
            buf83 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [mul_43, B_6], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf83, buf82, 1179648, stream=stream0)
            buf85 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, mul_42, X_9, mul_43, B_6, matmul_20], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf83, buf84, out=buf85)
            buf86 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf87 = buf84; del buf84  # reuse
            buf88 = reinterpret_tensor(buf83, (32, 192, 192), (36864, 1, 192), 0); del buf83  # reuse
            buf93 = buf81; del buf81  # reuse
            buf121 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf122 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf127 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, X_7, norm_4, add_18, X_8, mul_42, X_9, mul_45, X_10, A_7, transpose_13, matmul_23, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, A_10, transpose_16, matmul_32], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_27.run(buf67, buf15, buf69, buf77, buf85, buf118, buf120, buf86, buf87, buf88, buf93, buf121, buf122, buf127, 1179648, stream=stream0)
            buf89 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [A_7, transpose_13], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf87, buf88, out=buf89)
            buf90 = reinterpret_tensor(buf88, (32, 192, 192), (36864, 192, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [mul_47], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf89, buf90, 1179648, stream=stream0)
            buf91 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [mul_47, matmul_22], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf90, buf89, out=buf91)
            buf92 = buf89; del buf89  # reuse
            # Topologically Sorted Source Nodes: [mul_46, B_7], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf92, buf91, 1179648, stream=stream0)
            buf94 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [mul_46, B_7, matmul_23], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf92, buf93, out=buf94)
            buf95 = buf93; del buf93  # reuse
            buf96 = reinterpret_tensor(buf92, (32, 192, 192), (36864, 1, 192), 0); del buf92  # reuse
            buf101 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [mul_48, X_11, A_8, transpose_14, matmul_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_15.run(buf86, buf94, buf95, buf96, buf101, 1179648, stream=stream0)
            buf97 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [mul_48, X_11, A_8, transpose_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf95, buf96, out=buf97)
            buf98 = reinterpret_tensor(buf96, (32, 192, 192), (36864, 192, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [mul_50], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf97, buf98, 1179648, stream=stream0)
            buf99 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [mul_50, matmul_25], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf98, buf97, out=buf99)
            buf100 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [mul_49, B_8], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf100, buf99, 1179648, stream=stream0)
            buf102 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [mul_48, X_11, mul_49, B_8, matmul_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf100, buf101, out=buf102)
            buf103 = buf101; del buf101  # reuse
            buf104 = reinterpret_tensor(buf100, (32, 192, 192), (36864, 1, 192), 0); del buf100  # reuse
            buf109 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [mul_48, X_11, mul_51, X_12, A_9, transpose_15, matmul_29], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_18.run(buf86, buf94, buf102, buf103, buf104, buf109, 1179648, stream=stream0)
            buf105 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_48, X_11, mul_51, X_12, A_9, transpose_15], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf103, buf104, out=buf105)
            buf106 = reinterpret_tensor(buf104, (32, 192, 192), (36864, 192, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [mul_53], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf105, buf106, 1179648, stream=stream0)
            buf107 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [mul_53, matmul_28], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf106, buf105, out=buf107)
            buf108 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [mul_52, B_9], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf108, buf107, 1179648, stream=stream0)
            buf110 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [mul_48, X_11, mul_51, X_12, mul_52, B_9, matmul_29], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf108, buf109, out=buf110)
            buf113 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf111 = buf86; del buf86  # reuse
            buf114 = buf109; del buf109  # reuse
            buf171 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [mul_48, X_11, mul_51, X_12, mul_54, X_13, w0_main, norm_6, add_43, truediv_3, w0_norm, w0, bmm_10, gate_before_act_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_28.run(buf111, arg0_1, buf94, buf102, buf110, buf113, buf114, buf171, 6144, 192, stream=stream0)
            del arg0_1
            buf172 = reinterpret_tensor(buf66, (32, 192, 1024), (196608, 1024, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [norm_6, add_43, truediv_3, w0_norm, w0, gate_before_act_1, ki_1, transpose_21], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.slice, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf171, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf172)
            buf123 = buf171; del buf171  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, A_10, transpose_16], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf121, buf122, out=buf123)
            buf124 = reinterpret_tensor(buf122, (32, 192, 192), (36864, 192, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [mul_56], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf123, buf124, 1179648, stream=stream0)
            buf125 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [mul_56, matmul_31], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf124, buf123, out=buf125)
            buf126 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [mul_55, B_10], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf126, buf125, 1179648, stream=stream0)
            buf128 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, mul_55, B_10, matmul_32], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf126, buf127, out=buf128)
            buf129 = buf127; del buf127  # reuse
            buf130 = reinterpret_tensor(buf126, (32, 192, 192), (36864, 1, 192), 0); del buf126  # reuse
            buf135 = buf124; del buf124  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, mul_57, X_16, A_11, transpose_17, matmul_35], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_9.run(buf118, buf15, buf120, buf128, buf129, buf130, buf135, 1179648, stream=stream0)
            buf131 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, mul_57, X_16, A_11, transpose_17], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf129, buf130, out=buf131)
            buf132 = reinterpret_tensor(buf130, (32, 192, 192), (36864, 192, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [mul_59], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf131, buf132, 1179648, stream=stream0)
            buf133 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [mul_59, matmul_34], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf132, buf131, out=buf133)
            buf134 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [mul_58, B_11], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf134, buf133, 1179648, stream=stream0)
            buf136 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, mul_57, X_16, mul_58, B_11, matmul_35], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.bmm]
            extern_kernels.bmm(buf134, buf135, out=buf136)
            buf137 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf138 = buf135; del buf135  # reuse
            buf139 = reinterpret_tensor(buf134, (32, 192, 192), (36864, 1, 192), 0); del buf134  # reuse
            buf144 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, X_14, norm_5, add_29, X_15, mul_57, X_16, mul_60, X_17, A_12, transpose_18, matmul_38], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_12.run(buf118, buf15, buf120, buf128, buf136, buf137, buf138, buf139, buf144, 1179648, stream=stream0)
            buf140 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [A_12, transpose_18], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf138, buf139, out=buf140)
            buf141 = reinterpret_tensor(buf139, (32, 192, 192), (36864, 192, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [mul_62], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf140, buf141, 1179648, stream=stream0)
            buf142 = buf138; del buf138  # reuse
            # Topologically Sorted Source Nodes: [mul_62, matmul_37], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf141, buf140, out=buf142)
            buf143 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [mul_61, B_12], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf143, buf142, 1179648, stream=stream0)
            buf145 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [mul_61, B_12, matmul_38], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf143, buf144, out=buf145)
            buf146 = buf144; del buf144  # reuse
            buf147 = reinterpret_tensor(buf143, (32, 192, 192), (36864, 1, 192), 0); del buf143  # reuse
            buf152 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, A_13, transpose_19, matmul_41], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_15.run(buf137, buf145, buf146, buf147, buf152, 1179648, stream=stream0)
            buf148 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, A_13, transpose_19], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf146, buf147, out=buf148)
            buf149 = reinterpret_tensor(buf147, (32, 192, 192), (36864, 192, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [mul_65], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf148, buf149, 1179648, stream=stream0)
            buf150 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [mul_65, matmul_40], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf149, buf148, out=buf150)
            buf151 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [mul_64, B_13], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf151, buf150, 1179648, stream=stream0)
            buf153 = buf150; del buf150  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, mul_64, B_13, matmul_41], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf151, buf152, out=buf153)
            buf154 = buf152; del buf152  # reuse
            buf155 = reinterpret_tensor(buf151, (32, 192, 192), (36864, 1, 192), 0); del buf151  # reuse
            buf160 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, mul_66, X_19, A_14, transpose_20, matmul_44], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_18.run(buf137, buf145, buf153, buf154, buf155, buf160, 1179648, stream=stream0)
            buf156 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, mul_66, X_19, A_14, transpose_20], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf154, buf155, out=buf156)
            buf157 = reinterpret_tensor(buf155, (32, 192, 192), (36864, 192, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [mul_68], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf156, buf157, 1179648, stream=stream0)
            buf158 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [mul_68, matmul_43], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf157, buf156, out=buf158)
            buf159 = buf156; del buf156  # reuse
            # Topologically Sorted Source Nodes: [mul_67, B_14], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf159, buf158, 1179648, stream=stream0)
            buf161 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, mul_66, X_19, mul_67, B_14, matmul_44], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf159, buf160, out=buf161)
            buf164 = empty_strided_cuda((32, 192, 1), (192, 1, 6144), torch.float32)
            buf162 = buf137; del buf137  # reuse
            buf165 = buf160; del buf160  # reuse
            buf173 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [mul_63, X_18, mul_66, X_19, mul_69, X_20, w2_main, norm_8, add_45, truediv_5, w2_norm, w2, h_1, hidden_before_mul_1], Original ATen: [aten.mul, aten.add, aten.linalg_vector_norm, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_28.run(buf162, arg2_1, buf145, buf153, buf161, buf164, buf165, buf173, 6144, 192, stream=stream0)
            del arg2_1
            buf174 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [norm_8, add_45, truediv_5, w2_norm, w2, ki_1, hidden_before_mul_1, transpose_22], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.slice, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf173, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf174)
            buf115 = reinterpret_tensor(buf117, (32, 192, 1024), (196608, 1024, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [q, norm_6, add_43, truediv_3, w0_norm, w0, bmm_10, qi_1], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.slice, aten.bmm]
            extern_kernels.bmm(buf114, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf115)
            buf166 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [q, qi_1, norm_8, add_45, truediv_5, w2_norm, w2, h_1], Original ATen: [aten.transpose, aten.slice, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf165, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf166)
            buf167 = buf64; del buf64  # reuse
            buf175 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [gate_1, mul_73, getitem_35, float_4, x_rot_3, x1_3, hci_1, c_3, mul_74, x2_3, hsi_1, s__3, mul_75, sub_4, mul_76, mul_77, add_46, y_6, silu_4, hidden_1, getitem_41, float_5, x_rot_4, x1_4, c_4, mul_79, x2_4, s__4, mul_80, sub_5, mul_81, mul_82, add_47, y_8], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_29.run(buf115, buf166, arg10_1, arg11_1, buf172, buf174, buf167, buf175, 3145728, stream=stream0)
            buf176 = empty_strided_cuda((32, 1024, 192), (196608, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_4, hidden_1, y_8, reshape_9, y_9, getitem_46, hidden_rot_1, transpose_24, lr1i_1, mul_92, type_as_9], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_30.run(buf175, buf172, buf174, arg7_1, buf176, 32768, 192, stream=stream0)
            buf177 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [v, vi_1, silu_4, hidden_1, y_8, reshape_9, y_9, getitem_46, hidden_rot_1, transpose_24, lr1i_1, mul_92, type_as_9, dw1_2], Original ATen: [aten.transpose, aten.slice, aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.cat, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 196608), buf176, out=buf177)
            buf178 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [m_i_2, m_i_3], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_31.run(arg3_1, buf178, 32, 1024, stream=stream0)
            buf179 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, X_21, norm_9], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_32.run(buf177, buf14, buf15, buf178, buf179, 160, 7373, stream=stream0)
            buf180 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, X_21, norm_9], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf179, buf180, 32, 5, stream=stream0)
            buf181 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf182 = buf114; del buf114  # reuse
            buf183 = reinterpret_tensor(buf173, (32, 192, 192), (36864, 1, 192), 0); del buf173  # reuse
            buf188 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, X_21, norm_9, add_53, X_22, A_15, transpose_25, matmul_47], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_33.run(buf177, buf14, buf15, buf178, buf180, buf181, buf182, buf183, buf188, 1179648, stream=stream0)
            buf184 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [A_15, transpose_25], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf182, buf183, out=buf184)
            buf185 = reinterpret_tensor(buf183, (32, 192, 192), (36864, 192, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [mul_99], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf184, buf185, 1179648, stream=stream0)
            buf186 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [mul_99, matmul_46], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf185, buf184, out=buf186)
            buf187 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [mul_98, B_15], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf187, buf186, 1179648, stream=stream0)
            buf189 = buf186; del buf186  # reuse
            # Topologically Sorted Source Nodes: [mul_98, B_15, matmul_47], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf187, buf188, out=buf189)
            buf190 = buf188; del buf188  # reuse
            buf191 = reinterpret_tensor(buf187, (32, 192, 192), (36864, 1, 192), 0); del buf187  # reuse
            buf196 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, A_16, transpose_26, matmul_50], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_34.run(buf181, buf189, buf190, buf191, buf196, 1179648, stream=stream0)
            buf192 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, A_16, transpose_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf190, buf191, out=buf192)
            buf193 = reinterpret_tensor(buf191, (32, 192, 192), (36864, 192, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [mul_102], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf192, buf193, 1179648, stream=stream0)
            buf194 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [mul_102, matmul_49], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf193, buf192, out=buf194)
            buf195 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [mul_101, B_16], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf195, buf194, 1179648, stream=stream0)
            buf197 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_101, B_16, matmul_50], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf195, buf196, out=buf197)
            buf198 = buf196; del buf196  # reuse
            buf199 = reinterpret_tensor(buf195, (32, 192, 192), (36864, 1, 192), 0); del buf195  # reuse
            buf204 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, A_17, transpose_27, matmul_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_35.run(buf181, buf189, buf197, buf198, buf199, buf204, 1179648, stream=stream0)
            buf200 = buf157; del buf157  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, A_17, transpose_27], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf198, buf199, out=buf200)
            buf201 = reinterpret_tensor(buf199, (32, 192, 192), (36864, 192, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [mul_105], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf200, buf201, 1179648, stream=stream0)
            buf202 = buf198; del buf198  # reuse
            # Topologically Sorted Source Nodes: [mul_105, matmul_52], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf201, buf200, out=buf202)
            buf203 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [mul_104, B_17], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf203, buf202, 1179648, stream=stream0)
            buf205 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_104, B_17, matmul_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf203, buf204, out=buf205)
            buf206 = buf204; del buf204  # reuse
            buf207 = reinterpret_tensor(buf203, (32, 192, 192), (36864, 1, 192), 0); del buf203  # reuse
            buf212 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_106, X_25, A_18, transpose_28, matmul_56], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_36.run(buf181, buf189, buf197, buf205, buf206, buf207, buf212, 1179648, stream=stream0)
            buf208 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_106, X_25, A_18, transpose_28], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf206, buf207, out=buf208)
            buf209 = reinterpret_tensor(buf207, (32, 192, 192), (36864, 192, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [mul_108], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf208, buf209, 1179648, stream=stream0)
            buf210 = buf206; del buf206  # reuse
            # Topologically Sorted Source Nodes: [mul_108, matmul_55], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf209, buf208, out=buf210)
            buf211 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [mul_107, B_18], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf211, buf210, 1179648, stream=stream0)
            buf213 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_106, X_25, mul_107, B_18, matmul_56], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf211, buf212, out=buf213)
            buf214 = buf181; del buf181  # reuse
            buf215 = buf212; del buf212  # reuse
            buf216 = reinterpret_tensor(buf211, (32, 192, 192), (36864, 1, 192), 0); del buf211  # reuse
            buf221 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [mul_100, X_23, mul_103, X_24, mul_106, X_25, mul_109, X_26, A_19, transpose_29, matmul_59], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_37.run(buf214, buf189, buf197, buf205, buf213, buf215, buf216, buf221, 1179648, stream=stream0)
            buf217 = buf213; del buf213  # reuse
            # Topologically Sorted Source Nodes: [A_19, transpose_29], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf215, buf216, out=buf217)
            buf218 = reinterpret_tensor(buf216, (32, 192, 192), (36864, 192, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [mul_111], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf217, buf218, 1179648, stream=stream0)
            buf219 = buf215; del buf215  # reuse
            # Topologically Sorted Source Nodes: [mul_111, matmul_58], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf218, buf217, out=buf219)
            buf220 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [mul_110, B_19], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf220, buf219, 1179648, stream=stream0)
            buf222 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [mul_110, B_19, matmul_59], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf220, buf221, out=buf222)
            buf168 = buf221; del buf221  # reuse
            buf225 = reinterpret_tensor(buf220, (32, 192, 192), (36864, 1, 192), 0); del buf220  # reuse
            buf331 = buf218; del buf218  # reuse
            buf388 = reinterpret_tensor(buf205, (32, 192, 192), (36864, 1, 192), 0); del buf205  # reuse
            # Topologically Sorted Source Nodes: [norm_7, add_44, truediv_4, w1_norm, w1, bmm_11, mul_112, X_27, w1_main_1, norm_13, add_90, truediv_10, w1_1, bmm_20, transpose_23, dhidden_rot_1, transpose_42, dhidden_rot_2], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_38.run(buf59, buf214, buf222, buf61, buf168, buf225, buf331, buf388, 6144, 192, stream=stream0)
            buf169 = reinterpret_tensor(buf176, (32, 192, 1024), (196608, 1024, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [gate_1, mul_73, y_6, reshape_7, y_7, getitem_40, hq_rot_1], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22.run(buf167, buf115, buf166, buf169, 6291456, stream=stream0)
            buf170 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [norm_7, add_44, truediv_4, w1_norm, w1, bmm_11, gate_1, mul_73, y_6, reshape_7, y_7, getitem_40, hq_rot_1], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.silu, aten.stack, aten.view, aten.slice, aten.cat, aten.bmm]
            extern_kernels.bmm(buf168, buf169, out=buf170)
            buf226 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [v, norm_7, add_44, truediv_4, w1_norm, w1, vi_1, transpose_23, dhidden_rot_1], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf225, reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 196608), out=buf226)
            buf227 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [hci_1, hsi_1, getitem_47, float_6, x_rot_5, x1_5, c_5, mul_83, x2_5, neg_1, s__5, mul_84, sub_6, mul_85, mul_86, add_48, y_10], Original ATen: [aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.mul, aten.neg, aten.sub, aten.add, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_neg_select_slice_stack_sub_unsqueeze_view_39.run(buf226, arg10_1, arg11_1, buf227, 3145728, stream=stream0)
            buf228 = buf174; del buf174  # reuse
            buf279 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [y_10, reshape_11, y_11, getitem_52, dhidden_1, dgate_1, sigma_1, mul_89, sub_7, mul_90, add_49, dx_1, silu_5, dhidden_before_mul_1], Original ATen: [aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24.run(buf228, buf227, buf226, buf172, buf279, 6291456, stream=stream0)
            buf229 = reinterpret_tensor(buf226, (32, 1024, 192), (196608, 192, 1), 0); del buf226  # reuse
            buf280 = reinterpret_tensor(buf172, (32, 1024, 192), (196608, 192, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [ki_1, lr0i_1, mul_93, type_as_10, lr2i_1, mul_94, type_as_11], Original ATen: [aten.slice, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_40.run(arg6_1, arg9_1, arg8_1, buf229, buf280, 6291456, stream=stream0)
            buf230 = reinterpret_tensor(buf225, (32, 192, 192), (36864, 192, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [ki_1, y_10, reshape_11, y_11, getitem_52, dhidden_1, dgate_1, sigma_1, mul_89, sub_7, mul_90, add_49, dx_1, lr0i_1, mul_93, type_as_10, dw0_2], Original ATen: [aten.slice, aten.stack, aten.view, aten._to_copy, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.bmm]
            extern_kernels.bmm(buf228, buf229, out=buf230)
            buf281 = buf168; del buf168  # reuse
            # Topologically Sorted Source Nodes: [ki_1, y_10, reshape_11, y_11, getitem_52, dhidden_1, silu_5, dhidden_before_mul_1, lr2i_1, mul_94, type_as_11, dw2_2], Original ATen: [aten.slice, aten.stack, aten.view, aten._to_copy, aten.cat, aten.silu, aten.mul, aten.bmm]
            extern_kernels.bmm(buf279, buf280, out=buf281)
            buf231 = buf179; del buf179  # reuse
            buf282 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, dw2_momentum, mul_24, dw2_1, m_i_2, m_i_3, mul_95, dw0_3, X_28, norm_10, mul_97, dw2_3, X_35, norm_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_41.run(buf230, buf67, buf15, buf178, buf281, buf118, buf231, buf282, 160, 7373, stream=stream0)
            buf232 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, m_i_2, m_i_3, mul_95, dw0_3, X_28, norm_10], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf231, buf232, 32, 5, stream=stream0)
            del buf231
            buf283 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, m_i_2, m_i_3, mul_97, dw2_3, X_35, norm_11], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf282, buf283, 32, 5, stream=stream0)
            del buf282
            buf233 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf234 = buf197; del buf197  # reuse
            buf235 = reinterpret_tensor(buf189, (32, 192, 192), (36864, 1, 192), 0); del buf189  # reuse
            buf240 = buf106; del buf106  # reuse
            buf284 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf285 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            buf286 = empty_strided_cuda((32, 192, 192), (36864, 1, 192), torch.bfloat16)
            buf291 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, dw2_momentum, mul_24, dw2_1, m_i_2, m_i_3, mul_95, dw0_3, X_28, norm_10, add_64, X_29, A_20, transpose_30, matmul_62, mul_97, dw2_3, X_35, norm_11, add_75, X_36, A_25, transpose_35, matmul_77], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mean_mul_slice_transpose_zeros_like_42.run(buf230, buf67, buf15, buf178, buf232, buf281, buf118, buf283, buf233, buf234, buf235, buf240, buf284, buf285, buf286, buf291, 1179648, stream=stream0)
            del buf232
            del buf283
            buf236 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [A_20, transpose_30], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf234, buf235, out=buf236)
            buf237 = reinterpret_tensor(buf235, (32, 192, 192), (36864, 192, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [mul_114], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf236, buf237, 1179648, stream=stream0)
            buf238 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [mul_114, matmul_61], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf237, buf236, out=buf238)
            buf239 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [mul_113, B_20], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf239, buf238, 1179648, stream=stream0)
            buf241 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [mul_113, B_20, matmul_62], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf239, buf240, out=buf241)
            buf242 = buf240; del buf240  # reuse
            buf243 = reinterpret_tensor(buf239, (32, 192, 192), (36864, 1, 192), 0); del buf239  # reuse
            buf248 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, A_21, transpose_31, matmul_65], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_34.run(buf233, buf241, buf242, buf243, buf248, 1179648, stream=stream0)
            buf244 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_115, X_30, A_21, transpose_31], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf242, buf243, out=buf244)
            buf245 = reinterpret_tensor(buf243, (32, 192, 192), (36864, 192, 1), 0); del buf243  # reuse
            # Topologically Sorted Source Nodes: [mul_117], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf244, buf245, 1179648, stream=stream0)
            buf246 = buf242; del buf242  # reuse
            # Topologically Sorted Source Nodes: [mul_117, matmul_64], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf245, buf244, out=buf246)
            buf247 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [mul_116, B_21], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf247, buf246, 1179648, stream=stream0)
            buf249 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_116, B_21, matmul_65], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf247, buf248, out=buf249)
            buf250 = buf248; del buf248  # reuse
            buf251 = reinterpret_tensor(buf247, (32, 192, 192), (36864, 1, 192), 0); del buf247  # reuse
            buf256 = buf245; del buf245  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, A_22, transpose_32, matmul_68], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_35.run(buf233, buf241, buf249, buf250, buf251, buf256, 1179648, stream=stream0)
            buf252 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, A_22, transpose_32], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf250, buf251, out=buf252)
            buf253 = reinterpret_tensor(buf251, (32, 192, 192), (36864, 192, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [mul_120], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf252, buf253, 1179648, stream=stream0)
            buf254 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [mul_120, matmul_67], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf253, buf252, out=buf254)
            buf255 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [mul_119, B_22], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf255, buf254, 1179648, stream=stream0)
            buf257 = buf254; del buf254  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, mul_119, B_22, matmul_68], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf255, buf256, out=buf257)
            buf258 = buf256; del buf256  # reuse
            buf259 = reinterpret_tensor(buf255, (32, 192, 192), (36864, 1, 192), 0); del buf255  # reuse
            buf264 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, mul_121, X_32, A_23, transpose_33, matmul_71], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_36.run(buf233, buf241, buf249, buf257, buf258, buf259, buf264, 1179648, stream=stream0)
            buf260 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, mul_121, X_32, A_23, transpose_33], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf258, buf259, out=buf260)
            buf261 = reinterpret_tensor(buf259, (32, 192, 192), (36864, 192, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [mul_123], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf260, buf261, 1179648, stream=stream0)
            buf262 = buf258; del buf258  # reuse
            # Topologically Sorted Source Nodes: [mul_123, matmul_70], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf261, buf260, out=buf262)
            buf263 = buf260; del buf260  # reuse
            # Topologically Sorted Source Nodes: [mul_122, B_23], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf263, buf262, 1179648, stream=stream0)
            buf265 = buf262; del buf262  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, mul_121, X_32, mul_122, B_23, matmul_71], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf263, buf264, out=buf265)
            buf266 = buf233; del buf233  # reuse
            buf267 = buf264; del buf264  # reuse
            buf268 = reinterpret_tensor(buf263, (32, 192, 192), (36864, 1, 192), 0); del buf263  # reuse
            buf273 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [mul_115, X_30, mul_118, X_31, mul_121, X_32, mul_124, X_33, A_24, transpose_34, matmul_74], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_37.run(buf266, buf241, buf249, buf257, buf265, buf267, buf268, buf273, 1179648, stream=stream0)
            del buf241
            buf269 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [A_24, transpose_34], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf267, buf268, out=buf269)
            buf270 = reinterpret_tensor(buf268, (32, 192, 192), (36864, 192, 1), 0); del buf268  # reuse
            # Topologically Sorted Source Nodes: [mul_126], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf269, buf270, 1179648, stream=stream0)
            buf271 = buf267; del buf267  # reuse
            # Topologically Sorted Source Nodes: [mul_126, matmul_73], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf270, buf269, out=buf271)
            buf272 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [mul_125, B_24], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf272, buf271, 1179648, stream=stream0)
            buf274 = buf271; del buf271  # reuse
            # Topologically Sorted Source Nodes: [mul_125, B_24, matmul_74], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf272, buf273, out=buf274)
            buf277 = buf273; del buf273  # reuse
            buf334 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [w0_norm, mul_127, X_34, w0_main_1, norm_12, add_89, truediv_9, w0_1, bmm_19, gate_before_act_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_43.run(buf111, buf266, buf274, buf113, buf277, buf334, 6144, 192, stream=stream0)
            buf278 = reinterpret_tensor(buf280, (32, 192, 1024), (196608, 1024, 1), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [q, bmm_19, qi_2], Original ATen: [aten.transpose, aten._to_copy, aten.slice, aten.bmm]
            extern_kernels.bmm(buf277, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf278)
            buf287 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [A_25, transpose_35], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf285, buf286, out=buf287)
            buf288 = reinterpret_tensor(buf286, (32, 192, 192), (36864, 192, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [mul_129], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf287, buf288, 1179648, stream=stream0)
            buf289 = buf285; del buf285  # reuse
            # Topologically Sorted Source Nodes: [mul_129, matmul_76], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf288, buf287, out=buf289)
            buf290 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [mul_128, B_25], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf290, buf289, 1179648, stream=stream0)
            buf292 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [mul_128, B_25, matmul_77], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf290, buf291, out=buf292)
            buf293 = buf291; del buf291  # reuse
            buf294 = reinterpret_tensor(buf290, (32, 192, 192), (36864, 1, 192), 0); del buf290  # reuse
            buf299 = buf288; del buf288  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, A_26, transpose_36, matmul_80], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_34.run(buf284, buf292, buf293, buf294, buf299, 1179648, stream=stream0)
            buf295 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, A_26, transpose_36], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf293, buf294, out=buf295)
            buf296 = reinterpret_tensor(buf294, (32, 192, 192), (36864, 192, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [mul_132], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf295, buf296, 1179648, stream=stream0)
            buf297 = buf293; del buf293  # reuse
            # Topologically Sorted Source Nodes: [mul_132, matmul_79], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf296, buf295, out=buf297)
            buf298 = buf295; del buf295  # reuse
            # Topologically Sorted Source Nodes: [mul_131, B_26], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf298, buf297, 1179648, stream=stream0)
            buf300 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_131, B_26, matmul_80], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf298, buf299, out=buf300)
            buf301 = buf299; del buf299  # reuse
            buf302 = reinterpret_tensor(buf298, (32, 192, 192), (36864, 1, 192), 0); del buf298  # reuse
            buf307 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, A_27, transpose_37, matmul_83], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_35.run(buf284, buf292, buf300, buf301, buf302, buf307, 1179648, stream=stream0)
            buf303 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, A_27, transpose_37], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf301, buf302, out=buf303)
            buf304 = reinterpret_tensor(buf302, (32, 192, 192), (36864, 192, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [mul_135], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf303, buf304, 1179648, stream=stream0)
            buf305 = buf301; del buf301  # reuse
            # Topologically Sorted Source Nodes: [mul_135, matmul_82], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf304, buf303, out=buf305)
            buf306 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [mul_134, B_27], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf306, buf305, 1179648, stream=stream0)
            buf308 = buf305; del buf305  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, mul_134, B_27, matmul_83], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf306, buf307, out=buf308)
            buf309 = buf307; del buf307  # reuse
            buf310 = reinterpret_tensor(buf306, (32, 192, 192), (36864, 1, 192), 0); del buf306  # reuse
            buf315 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, mul_136, X_39, A_28, transpose_38, matmul_86], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_36.run(buf284, buf292, buf300, buf308, buf309, buf310, buf315, 1179648, stream=stream0)
            buf311 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, mul_136, X_39, A_28, transpose_38], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf309, buf310, out=buf311)
            buf312 = reinterpret_tensor(buf310, (32, 192, 192), (36864, 192, 1), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [mul_138], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf311, buf312, 1179648, stream=stream0)
            buf313 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [mul_138, matmul_85], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf312, buf311, out=buf313)
            buf314 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [mul_137, B_28], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf314, buf313, 1179648, stream=stream0)
            buf316 = buf313; del buf313  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, mul_136, X_39, mul_137, B_28, matmul_86], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf314, buf315, out=buf316)
            buf317 = buf284; del buf284  # reuse
            buf318 = buf315; del buf315  # reuse
            buf319 = reinterpret_tensor(buf314, (32, 192, 192), (36864, 1, 192), 0); del buf314  # reuse
            buf324 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [mul_130, X_37, mul_133, X_38, mul_136, X_39, mul_139, X_40, A_29, transpose_39, matmul_89], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_37.run(buf317, buf292, buf300, buf308, buf316, buf318, buf319, buf324, 1179648, stream=stream0)
            del buf292
            del buf300
            del buf308
            buf320 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [A_29, transpose_39], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf318, buf319, out=buf320)
            buf321 = reinterpret_tensor(buf319, (32, 192, 192), (36864, 192, 1), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [mul_141], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf320, buf321, 1179648, stream=stream0)
            buf322 = buf318; del buf318  # reuse
            # Topologically Sorted Source Nodes: [mul_141, matmul_88], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf321, buf320, out=buf322)
            del buf321
            buf323 = buf320; del buf320  # reuse
            # Topologically Sorted Source Nodes: [mul_140, B_29], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf323, buf322, 1179648, stream=stream0)
            buf325 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [mul_140, B_29, matmul_89], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf323, buf324, out=buf325)
            buf328 = buf324; del buf324  # reuse
            buf336 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [w2_norm, mul_142, X_41, w2_main_1, norm_14, add_91, truediv_11, w2_1, h_2, hidden_before_mul_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_43.run(buf162, buf317, buf325, buf164, buf328, buf336, 6144, 192, stream=stream0)
            buf329 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [q, qi_2, h_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf328, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf329)
            del buf328
            buf335 = reinterpret_tensor(buf229, (32, 192, 1024), (196608, 1024, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [gate_before_act_2, ki_2, transpose_40], Original ATen: [aten._to_copy, aten.slice, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf334, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf335)
            del buf334
            buf337 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [ki_2, hidden_before_mul_2, transpose_41], Original ATen: [aten.slice, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf336, reinterpret_tensor(arg6_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf337)
            del buf336
            buf389 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [v, vi_2, transpose_42, dhidden_rot_2], Original ATen: [aten.transpose, aten.slice, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf388, reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 393216), out=buf389)
            del buf388
            buf330 = buf227; del buf227  # reuse
            buf338 = buf175; del buf175  # reuse
            buf390 = empty_strided_cuda((32, 48, 2048), (98304, 2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [gate_2, mul_146, getitem_62, float_7, x_rot_6, x1_6, hci_2, c_6, mul_147, x2_6, hsi_2, s__6, mul_148, sub_8, mul_149, mul_150, add_92, y_12, silu_7, hidden_2, getitem_68, float_8, x_rot_7, x1_7, c_7, mul_152, x2_7, s__7, mul_153, sub_9, mul_154, mul_155, add_93, y_14, getitem_74, float_9, x_rot_8, x1_8, c_8, mul_156, x2_8, neg_2, s__8, mul_157, sub_10, mul_158, mul_159, add_94, y_16], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack, aten.neg]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_neg_select_silu_slice_stack_sub_unsqueeze_view_44.run(buf278, buf329, arg10_1, arg11_1, buf335, buf337, buf389, buf330, buf338, buf390, 3145728, stream=stream0)
            buf332 = empty_strided_cuda((32, 192, 1024), (196608, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_2, mul_146, y_12, reshape_13, y_13, getitem_67, hq_rot_2], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22.run(buf330, buf278, buf329, buf332, 6291456, stream=stream0)
            del buf278
            del buf330
            buf333 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [bmm_20, gate_2, mul_146, y_12, reshape_13, y_13, getitem_67, hq_rot_2], Original ATen: [aten._to_copy, aten.silu, aten.mul, aten.stack, aten.view, aten.slice, aten.cat, aten.bmm]
            extern_kernels.bmm(buf331, buf332, out=buf333)
            buf339 = reinterpret_tensor(buf332, (32, 1024, 192), (196608, 192, 1), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [silu_7, hidden_2, y_14, reshape_15, y_15, getitem_73, hidden_rot_2, transpose_43, lr1i_2, mul_165, type_as_15], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_transpose_view_45.run(buf338, buf335, buf337, arg7_1, buf339, 32768, 192, stream=stream0)
            del arg7_1
            del buf338
            buf340 = buf331; del buf331  # reuse
            # Topologically Sorted Source Nodes: [v, vi_2, silu_7, hidden_2, y_14, reshape_15, y_15, getitem_73, hidden_rot_2, transpose_43, lr1i_2, mul_165, type_as_15, dw1_4], Original ATen: [aten.transpose, aten.slice, aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.cat, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(arg5_1, (32, 192, 1024), (786432, 1, 192), 393216), buf339, out=buf340)
            del arg5_1
            buf341 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [m_i_4, m_i_5], Original ATen: [aten.slice, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_slice_46.run(arg3_1, buf341, 32, 1024, stream=stream0)
            del arg3_1
            buf342 = buf340; del buf340  # reuse
            # Topologically Sorted Source Nodes: [dw1_momentum, m_i, m_i_1, mul_23, dw1_1, m_i_2, m_i_3, mul_96, dw1_3, m_i_4, m_i_5, mul_169, dw1_5, X_42], Original ATen: [aten.zeros_like, aten.slice, aten.mean, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47.run(buf342, buf177, buf14, buf15, buf178, buf341, 1179648, stream=stream0)
            buf343 = empty_strided_cuda((32, 1, 1, 5), (5, 160, 160, 1), torch.float32)
            # Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_48.run(buf342, buf343, 160, 7373, stream=stream0)
            buf344 = empty_strided_cuda((32, 1, 1), (1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [norm_15], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf343, buf344, 32, 5, stream=stream0)
            buf345 = buf177; del buf177  # reuse
            buf346 = reinterpret_tensor(buf14, (32, 192, 192), (36864, 1, 192), 0); del buf14  # reuse
            buf351 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, A_30, transpose_44, matmul_92], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49.run(buf342, buf344, buf345, buf346, buf351, 1179648, stream=stream0)
            buf347 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, A_30, transpose_44], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf345, buf346, out=buf347)
            buf348 = reinterpret_tensor(buf346, (32, 192, 192), (36864, 192, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [mul_172], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf347, buf348, 1179648, stream=stream0)
            buf349 = buf345; del buf345  # reuse
            # Topologically Sorted Source Nodes: [mul_172, matmul_91], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf348, buf347, out=buf349)
            buf350 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [mul_171, B_30], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf350, buf349, 1179648, stream=stream0)
            buf352 = buf349; del buf349  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_171, B_30, matmul_92], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf350, buf351, out=buf352)
            buf353 = buf351; del buf351  # reuse
            buf354 = reinterpret_tensor(buf350, (32, 192, 192), (36864, 1, 192), 0); del buf350  # reuse
            buf359 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, A_31, transpose_45, matmul_95], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50.run(buf342, buf344, buf352, buf353, buf354, buf359, 1179648, stream=stream0)
            buf355 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, A_31, transpose_45], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf353, buf354, out=buf355)
            buf356 = reinterpret_tensor(buf354, (32, 192, 192), (36864, 192, 1), 0); del buf354  # reuse
            # Topologically Sorted Source Nodes: [mul_175], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf355, buf356, 1179648, stream=stream0)
            buf357 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [mul_175, matmul_94], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf356, buf355, out=buf357)
            buf358 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [mul_174, B_31], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf358, buf357, 1179648, stream=stream0)
            buf360 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_174, B_31, matmul_95], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf358, buf359, out=buf360)
            buf361 = buf359; del buf359  # reuse
            buf362 = reinterpret_tensor(buf358, (32, 192, 192), (36864, 1, 192), 0); del buf358  # reuse
            buf367 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_176, X_45, A_32, transpose_46, matmul_98], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51.run(buf342, buf344, buf352, buf360, buf361, buf362, buf367, 1179648, stream=stream0)
            buf363 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_176, X_45, A_32, transpose_46], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf361, buf362, out=buf363)
            buf364 = reinterpret_tensor(buf362, (32, 192, 192), (36864, 192, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [mul_178], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf363, buf364, 1179648, stream=stream0)
            buf365 = buf361; del buf361  # reuse
            # Topologically Sorted Source Nodes: [mul_178, matmul_97], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf364, buf363, out=buf365)
            buf366 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [mul_177, B_32], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf366, buf365, 1179648, stream=stream0)
            buf368 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_176, X_45, mul_177, B_32, matmul_98], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf366, buf367, out=buf368)
            buf369 = empty_strided_cuda((32, 192, 192), (36864, 192, 1), torch.float32)
            buf370 = buf367; del buf367  # reuse
            buf371 = reinterpret_tensor(buf366, (32, 192, 192), (36864, 1, 192), 0); del buf366  # reuse
            buf376 = buf364; del buf364  # reuse
            # Topologically Sorted Source Nodes: [norm_15, add_99, X_43, mul_173, X_44, mul_176, X_45, mul_179, X_46, A_33, transpose_47, matmul_101], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52.run(buf342, buf344, buf352, buf360, buf368, buf369, buf370, buf371, buf376, 1179648, stream=stream0)
            del buf342
            del buf352
            buf372 = buf368; del buf368  # reuse
            # Topologically Sorted Source Nodes: [A_33, transpose_47], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf370, buf371, out=buf372)
            buf373 = reinterpret_tensor(buf371, (32, 192, 192), (36864, 192, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [mul_181], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf372, buf373, 1179648, stream=stream0)
            buf374 = buf370; del buf370  # reuse
            # Topologically Sorted Source Nodes: [mul_181, matmul_100], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf373, buf372, out=buf374)
            buf375 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [mul_180, B_33], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf375, buf374, 1179648, stream=stream0)
            buf377 = buf374; del buf374  # reuse
            # Topologically Sorted Source Nodes: [mul_180, B_33, matmul_101], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf375, buf376, out=buf377)
            buf378 = buf376; del buf376  # reuse
            buf379 = reinterpret_tensor(buf375, (32, 192, 192), (36864, 1, 192), 0); del buf375  # reuse
            buf384 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [mul_182, X_47, A_34, transpose_48, matmul_104], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_53.run(buf369, buf377, buf378, buf379, buf384, 1179648, stream=stream0)
            buf380 = buf360; del buf360  # reuse
            # Topologically Sorted Source Nodes: [mul_182, X_47, A_34, transpose_48], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf378, buf379, out=buf380)
            buf381 = reinterpret_tensor(buf379, (32, 192, 192), (36864, 192, 1), 0); del buf379  # reuse
            # Topologically Sorted Source Nodes: [mul_184], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf380, buf381, 1179648, stream=stream0)
            buf382 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [mul_184, matmul_103], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf381, buf380, out=buf382)
            buf383 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [mul_183, B_34], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf383, buf382, 1179648, stream=stream0)
            buf385 = buf382; del buf382  # reuse
            # Topologically Sorted Source Nodes: [mul_182, X_47, mul_183, B_34, matmul_104], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf383, buf384, out=buf385)
            buf386 = buf59; del buf59  # reuse
            buf494 = buf384; del buf384  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, mul_112, X_27, w1_main_1, mul_182, X_47, mul_185, X_48, w1_main_2, norm_19, add_136, truediv_16, w1_2, bmm_29], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54.run(buf386, buf214, buf222, buf369, buf377, buf385, buf61, buf494, 6144, 192, stream=stream0)
            del buf214
            del buf369
            del buf61
            buf391 = buf337; del buf337  # reuse
            buf442 = reinterpret_tensor(buf339, (32, 192, 1024), (196608, 1024, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [y_16, reshape_17, y_17, getitem_79, dhidden_2, dgate_2, sigma_2, mul_162, sub_11, mul_163, add_95, dx_2, silu_8, dhidden_before_mul_2], Original ATen: [aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_mul_rsub_sigmoid_silu_slice_stack_view_24.run(buf391, buf390, buf389, buf335, buf442, 6291456, stream=stream0)
            buf392 = reinterpret_tensor(buf389, (32, 1024, 192), (196608, 192, 1), 0); del buf389  # reuse
            buf443 = reinterpret_tensor(buf335, (32, 1024, 192), (196608, 192, 1), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [ki_2, lr0i_2, mul_166, type_as_16, lr2i_2, mul_167, type_as_17], Original ATen: [aten.slice, aten.mul, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mul_slice_55.run(arg6_1, arg9_1, arg8_1, buf392, buf443, 6291456, stream=stream0)
            del arg6_1
            del arg8_1
            del arg9_1
            buf393 = buf385; del buf385  # reuse
            # Topologically Sorted Source Nodes: [ki_2, y_16, reshape_17, y_17, getitem_79, dhidden_2, dgate_2, sigma_2, mul_162, sub_11, mul_163, add_95, dx_2, lr0i_2, mul_166, type_as_16, dw0_4], Original ATen: [aten.slice, aten.stack, aten.view, aten._to_copy, aten.cat, aten.mul, aten.sigmoid, aten.rsub, aten.add, aten.bmm]
            extern_kernels.bmm(buf391, buf392, out=buf393)
            del buf391
            buf394 = buf393; del buf393  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw0_momentum, mul_22, dw0_1, m_i_2, m_i_3, mul_95, dw0_3, m_i_4, m_i_5, mul_168, dw0_5, X_49], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47.run(buf394, buf230, buf67, buf15, buf178, buf341, 1179648, stream=stream0)
            buf395 = buf343; del buf343  # reuse
            # Topologically Sorted Source Nodes: [norm_16], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_48.run(buf394, buf395, 160, 7373, stream=stream0)
            buf396 = buf344; del buf344  # reuse
            # Topologically Sorted Source Nodes: [norm_16], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf395, buf396, 32, 5, stream=stream0)
            buf397 = buf67; del buf67  # reuse
            buf398 = reinterpret_tensor(buf230, (32, 192, 192), (36864, 1, 192), 0); del buf230  # reuse
            buf403 = buf377; del buf377  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, A_35, transpose_49, matmul_107], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49.run(buf394, buf396, buf397, buf398, buf403, 1179648, stream=stream0)
            buf399 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, A_35, transpose_49], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf397, buf398, out=buf399)
            buf400 = reinterpret_tensor(buf398, (32, 192, 192), (36864, 192, 1), 0); del buf398  # reuse
            # Topologically Sorted Source Nodes: [mul_187], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf399, buf400, 1179648, stream=stream0)
            buf401 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [mul_187, matmul_106], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf400, buf399, out=buf401)
            buf402 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [mul_186, B_35], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf402, buf401, 1179648, stream=stream0)
            buf404 = buf401; del buf401  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_186, B_35, matmul_107], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf402, buf403, out=buf404)
            buf405 = buf403; del buf403  # reuse
            buf406 = reinterpret_tensor(buf402, (32, 192, 192), (36864, 1, 192), 0); del buf402  # reuse
            buf411 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, A_36, transpose_50, matmul_110], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50.run(buf394, buf396, buf404, buf405, buf406, buf411, 1179648, stream=stream0)
            buf407 = buf383; del buf383  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, A_36, transpose_50], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf405, buf406, out=buf407)
            buf408 = reinterpret_tensor(buf406, (32, 192, 192), (36864, 192, 1), 0); del buf406  # reuse
            # Topologically Sorted Source Nodes: [mul_190], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf407, buf408, 1179648, stream=stream0)
            buf409 = buf405; del buf405  # reuse
            # Topologically Sorted Source Nodes: [mul_190, matmul_109], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf408, buf407, out=buf409)
            buf410 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [mul_189, B_36], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf410, buf409, 1179648, stream=stream0)
            buf412 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, mul_189, B_36, matmul_110], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf410, buf411, out=buf412)
            buf413 = buf411; del buf411  # reuse
            buf414 = reinterpret_tensor(buf410, (32, 192, 192), (36864, 1, 192), 0); del buf410  # reuse
            buf419 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, mul_191, X_52, A_37, transpose_51, matmul_113], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51.run(buf394, buf396, buf404, buf412, buf413, buf414, buf419, 1179648, stream=stream0)
            buf415 = buf381; del buf381  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, mul_191, X_52, A_37, transpose_51], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf413, buf414, out=buf415)
            buf416 = reinterpret_tensor(buf414, (32, 192, 192), (36864, 192, 1), 0); del buf414  # reuse
            # Topologically Sorted Source Nodes: [mul_193], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf415, buf416, 1179648, stream=stream0)
            buf417 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [mul_193, matmul_112], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf416, buf415, out=buf417)
            buf418 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [mul_192, B_37], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf418, buf417, 1179648, stream=stream0)
            buf420 = buf417; del buf417  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, mul_191, X_52, mul_192, B_37, matmul_113], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf418, buf419, out=buf420)
            buf421 = buf386; del buf386  # reuse
            buf422 = buf419; del buf419  # reuse
            buf423 = reinterpret_tensor(buf418, (32, 192, 192), (36864, 1, 192), 0); del buf418  # reuse
            buf428 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [norm_16, add_110, X_50, mul_188, X_51, mul_191, X_52, mul_194, X_53, A_38, transpose_52, matmul_116], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52.run(buf394, buf396, buf404, buf412, buf420, buf421, buf422, buf423, buf428, 1179648, stream=stream0)
            del buf394
            del buf396
            del buf404
            buf424 = buf420; del buf420  # reuse
            # Topologically Sorted Source Nodes: [A_38, transpose_52], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf422, buf423, out=buf424)
            buf425 = reinterpret_tensor(buf423, (32, 192, 192), (36864, 192, 1), 0); del buf423  # reuse
            # Topologically Sorted Source Nodes: [mul_196], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf424, buf425, 1179648, stream=stream0)
            buf426 = buf422; del buf422  # reuse
            # Topologically Sorted Source Nodes: [mul_196, matmul_115], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf425, buf424, out=buf426)
            buf427 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [mul_195, B_38], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf427, buf426, 1179648, stream=stream0)
            buf429 = buf426; del buf426  # reuse
            # Topologically Sorted Source Nodes: [mul_195, B_38, matmul_116], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf427, buf428, out=buf429)
            buf430 = buf428; del buf428  # reuse
            buf431 = reinterpret_tensor(buf427, (32, 192, 192), (36864, 1, 192), 0); del buf427  # reuse
            buf436 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [mul_197, X_54, A_39, transpose_53, matmul_119], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_53.run(buf421, buf429, buf430, buf431, buf436, 1179648, stream=stream0)
            buf432 = buf412; del buf412  # reuse
            # Topologically Sorted Source Nodes: [mul_197, X_54, A_39, transpose_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf430, buf431, out=buf432)
            buf433 = reinterpret_tensor(buf431, (32, 192, 192), (36864, 192, 1), 0); del buf431  # reuse
            # Topologically Sorted Source Nodes: [mul_199], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf432, buf433, 1179648, stream=stream0)
            buf434 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [mul_199, matmul_118], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf433, buf432, out=buf434)
            del buf433
            buf435 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [mul_198, B_39], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf435, buf434, 1179648, stream=stream0)
            buf437 = buf434; del buf434  # reuse
            # Topologically Sorted Source Nodes: [mul_197, X_54, mul_198, B_39, matmul_119], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf435, buf436, out=buf437)
            buf438 = buf111; del buf111  # reuse
            buf440 = buf436; del buf436  # reuse
            # Topologically Sorted Source Nodes: [w0_norm, mul_127, X_34, w0_main_1, mul_197, X_54, mul_200, X_55, w0_main_2, norm_18, add_135, truediv_15, w0_2, bmm_28], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54.run(buf438, buf266, buf274, buf421, buf429, buf437, buf113, buf440, 6144, 192, stream=stream0)
            del buf113
            del buf266
            del buf421
            buf441 = reinterpret_tensor(buf392, (32, 192, 1024), (196608, 1024, 1), 0); del buf392  # reuse
            # Topologically Sorted Source Nodes: [q, w0_norm, norm_18, add_135, truediv_15, w0_2, bmm_28, qi_3], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.slice, aten.bmm]
            extern_kernels.bmm(buf440, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 589824), out=buf441)
            buf444 = buf440; del buf440  # reuse
            # Topologically Sorted Source Nodes: [ki_2, y_16, reshape_17, y_17, getitem_79, dhidden_2, silu_8, dhidden_before_mul_2, lr2i_2, mul_167, type_as_17, dw2_4], Original ATen: [aten.slice, aten.stack, aten.view, aten._to_copy, aten.cat, aten.silu, aten.mul, aten.bmm]
            extern_kernels.bmm(buf442, buf443, out=buf444)
            buf445 = buf444; del buf444  # reuse
            # Topologically Sorted Source Nodes: [m_i, m_i_1, dw2_momentum, mul_24, dw2_1, m_i_2, m_i_3, mul_97, dw2_3, m_i_4, m_i_5, mul_170, dw2_5, X_56], Original ATen: [aten.slice, aten.mean, aten.zeros_like, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mean_mul_slice_zeros_like_47.run(buf445, buf281, buf118, buf15, buf178, buf341, 1179648, stream=stream0)
            del buf15
            del buf178
            buf446 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [norm_17], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_48.run(buf445, buf446, 160, 7373, stream=stream0)
            buf447 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [norm_17], Original ATen: [aten.linalg_vector_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mean_mul_slice_zeros_like_5.run(buf446, buf447, 32, 5, stream=stream0)
            del buf446
            buf448 = buf281; del buf281  # reuse
            buf449 = reinterpret_tensor(buf118, (32, 192, 192), (36864, 1, 192), 0); del buf118  # reuse
            buf454 = buf437; del buf437  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, A_40, transpose_54, matmul_122], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_transpose_49.run(buf445, buf447, buf448, buf449, buf454, 1179648, stream=stream0)
            buf450 = buf429; del buf429  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, A_40, transpose_54], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf448, buf449, out=buf450)
            buf451 = reinterpret_tensor(buf449, (32, 192, 192), (36864, 192, 1), 0); del buf449  # reuse
            # Topologically Sorted Source Nodes: [mul_202], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_7.run(buf450, buf451, 1179648, stream=stream0)
            buf452 = buf448; del buf448  # reuse
            # Topologically Sorted Source Nodes: [mul_202, matmul_121], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf451, buf450, out=buf452)
            buf453 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [mul_201, B_40], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf453, buf452, 1179648, stream=stream0)
            buf455 = buf452; del buf452  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_201, B_40, matmul_122], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf453, buf454, out=buf455)
            buf456 = buf454; del buf454  # reuse
            buf457 = reinterpret_tensor(buf453, (32, 192, 192), (36864, 1, 192), 0); del buf453  # reuse
            buf462 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, A_41, transpose_55, matmul_125], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_50.run(buf445, buf447, buf455, buf456, buf457, buf462, 1179648, stream=stream0)
            buf458 = buf274; del buf274  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, A_41, transpose_55], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf456, buf457, out=buf458)
            buf459 = reinterpret_tensor(buf457, (32, 192, 192), (36864, 192, 1), 0); del buf457  # reuse
            # Topologically Sorted Source Nodes: [mul_205], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_10.run(buf458, buf459, 1179648, stream=stream0)
            buf460 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [mul_205, matmul_124], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf459, buf458, out=buf460)
            buf461 = buf458; del buf458  # reuse
            # Topologically Sorted Source Nodes: [mul_204, B_41], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf461, buf460, 1179648, stream=stream0)
            buf463 = buf460; del buf460  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, mul_204, B_41, matmul_125], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf461, buf462, out=buf463)
            buf464 = buf462; del buf462  # reuse
            buf465 = reinterpret_tensor(buf461, (32, 192, 192), (36864, 1, 192), 0); del buf461  # reuse
            buf470 = buf459; del buf459  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, mul_206, X_59, A_42, transpose_56, matmul_128], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_51.run(buf445, buf447, buf455, buf463, buf464, buf465, buf470, 1179648, stream=stream0)
            buf466 = buf435; del buf435  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, mul_206, X_59, A_42, transpose_56], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf464, buf465, out=buf466)
            buf467 = reinterpret_tensor(buf465, (32, 192, 192), (36864, 192, 1), 0); del buf465  # reuse
            # Topologically Sorted Source Nodes: [mul_208], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_13.run(buf466, buf467, 1179648, stream=stream0)
            buf468 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [mul_208, matmul_127], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf467, buf466, out=buf468)
            buf469 = buf466; del buf466  # reuse
            # Topologically Sorted Source Nodes: [mul_207, B_42], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf469, buf468, 1179648, stream=stream0)
            buf471 = buf468; del buf468  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, mul_206, X_59, mul_207, B_42, matmul_128], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf469, buf470, out=buf471)
            buf472 = buf438; del buf438  # reuse
            buf473 = buf470; del buf470  # reuse
            buf474 = reinterpret_tensor(buf469, (32, 192, 192), (36864, 1, 192), 0); del buf469  # reuse
            buf479 = buf467; del buf467  # reuse
            # Topologically Sorted Source Nodes: [norm_17, add_121, X_57, mul_203, X_58, mul_206, X_59, mul_209, X_60, A_43, transpose_57, matmul_131], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_linalg_vector_norm_mul_transpose_52.run(buf445, buf447, buf455, buf463, buf471, buf472, buf473, buf474, buf479, 1179648, stream=stream0)
            del buf445
            del buf447
            del buf455
            buf475 = buf471; del buf471  # reuse
            # Topologically Sorted Source Nodes: [A_43, transpose_57], Original ATen: [aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf473, buf474, out=buf475)
            buf476 = reinterpret_tensor(buf474, (32, 192, 192), (36864, 192, 1), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [mul_211], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_16.run(buf475, buf476, 1179648, stream=stream0)
            buf477 = buf473; del buf473  # reuse
            # Topologically Sorted Source Nodes: [mul_211, matmul_130], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf476, buf475, out=buf477)
            buf478 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [mul_210, B_43], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_17.run(buf478, buf477, 1179648, stream=stream0)
            buf480 = buf477; del buf477  # reuse
            # Topologically Sorted Source Nodes: [mul_210, B_43, matmul_131], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf478, buf479, out=buf480)
            buf481 = buf479; del buf479  # reuse
            buf482 = reinterpret_tensor(buf478, (32, 192, 192), (36864, 1, 192), 0); del buf478  # reuse
            buf487 = buf476; del buf476  # reuse
            # Topologically Sorted Source Nodes: [mul_212, X_61, A_44, transpose_58, matmul_134], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_transpose_53.run(buf472, buf480, buf481, buf482, buf487, 1179648, stream=stream0)
            buf483 = buf463; del buf463  # reuse
            # Topologically Sorted Source Nodes: [mul_212, X_61, A_44, transpose_58], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf481, buf482, out=buf483)
            buf484 = reinterpret_tensor(buf482, (32, 192, 192), (36864, 192, 1), 0); del buf482  # reuse
            # Topologically Sorted Source Nodes: [mul_214], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_19.run(buf483, buf484, 1179648, stream=stream0)
            buf485 = buf481; del buf481  # reuse
            # Topologically Sorted Source Nodes: [mul_214, matmul_133], Original ATen: [aten.mul, aten.bmm]
            extern_kernels.bmm(buf484, buf483, out=buf485)
            del buf484
            buf486 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [mul_213, B_44], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_20.run(buf486, buf485, 1179648, stream=stream0)
            buf488 = buf485; del buf485  # reuse
            # Topologically Sorted Source Nodes: [mul_212, X_61, mul_213, B_44, matmul_134], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf486, buf487, out=buf488)
            del buf486
            buf489 = buf162; del buf162  # reuse
            buf491 = buf487; del buf487  # reuse
            # Topologically Sorted Source Nodes: [w2_norm, mul_142, X_41, w2_main_1, mul_212, X_61, mul_215, X_62, w2_main_2, norm_20, add_137, truediv_17, w2_2, h_3], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_linalg_vector_norm_mul_54.run(buf489, buf317, buf325, buf472, buf480, buf488, buf164, buf491, 6144, 192, stream=stream0)
            del buf164
            del buf317
            del buf325
            del buf472
            del buf480
            del buf488
            del buf489
            buf492 = reinterpret_tensor(buf443, (32, 192, 1024), (196608, 1024, 1), 0); del buf443  # reuse
            # Topologically Sorted Source Nodes: [q, w2_norm, qi_3, norm_20, add_137, truediv_17, w2_2, h_3], Original ATen: [aten.transpose, aten.linalg_vector_norm, aten.slice, aten.add, aten.div, aten.mul, aten._to_copy, aten.bmm]
            extern_kernels.bmm(buf491, reinterpret_tensor(arg4_1, (32, 192, 1024), (786432, 1, 192), 589824), out=buf492)
            del arg4_1
            del buf491
            buf493 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [gate_3, mul_219, getitem_84, float_10, x_rot_9, x1_9, getitem_82, c_9, mul_220, x2_9, getitem_83, s__9, mul_221, sub_12, mul_222, mul_223, add_138, y_18], Original ATen: [aten.silu, aten.mul, aten.slice, aten._to_copy, aten.view, aten.select, aten.unsqueeze, aten.sub, aten.add, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_mul_select_silu_slice_stack_sub_unsqueeze_view_56.run(buf441, buf492, arg10_1, arg11_1, buf493, 3145728, stream=stream0)
            del arg10_1
            del arg11_1
            buf495 = buf442; del buf442  # reuse
            # Topologically Sorted Source Nodes: [gate_3, mul_219, y_18, reshape_19, y_19, getitem_89, hq_rot_3], Original ATen: [aten.silu, aten.mul, aten.stack, aten.view, aten._to_copy, aten.slice, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_mul_silu_slice_stack_view_22.run(buf493, buf441, buf492, buf495, 6291456, stream=stream0)
            del buf441
            del buf493
            buf496 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [w1_norm, norm_19, add_136, truediv_16, w1_2, bmm_29, gate_3, mul_219, y_18, reshape_19, y_19, getitem_89, hq_rot_3], Original ATen: [aten.linalg_vector_norm, aten.add, aten.div, aten.mul, aten._to_copy, aten.silu, aten.stack, aten.view, aten.slice, aten.cat, aten.bmm]
            extern_kernels.bmm(buf494, buf495, out=buf496)
            del buf494
            del buf495
            buf497 = empty_strided_cuda((32, 192, 4096), (786432, 4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_zeros_like_57.run(buf496, buf333, buf170, buf7, buf497, 25165824, stream=stream0)
            del buf170
            del buf333
            del buf496
            del buf7
            buf498 = empty_strided_cuda((32, 4096, 192), (786432, 192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [output, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.zeros_like, aten.slice, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_zeros_like_58.run(buf497, buf498, 131072, 192, stream=stream0)
            del buf497
        return (buf498, )

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
    arg10_1 = rand_strided((48, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((48, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
