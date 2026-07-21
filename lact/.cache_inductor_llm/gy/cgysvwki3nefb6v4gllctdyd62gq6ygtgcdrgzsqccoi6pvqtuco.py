# AOT ID: ['0_backward']
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


# kernel path: /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/loop_TTT/lact/.cache_inductor_llm/27/c27j3ujrihvxxhnkjv56kbaebhmahe5qtlnvaj2cdhai5ofsa7gv.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.neg, aten.add, aten.div, aten.mul, aten.sum, aten.eq, aten.masked_fill]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %tangents_1 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_1 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %pow_2 : Tensor "f32[96, 4096, 1][4096, 1, 1]cuda:0" = PlaceHolder[target=pow_2]
#   %sum_2 : Tensor "f32[96, 4096, 1][4096, 1, 393216]cuda:0" = PlaceHolder[target=sum_2]
#   %neg : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%tangents_1,), kwargs = {})
#   %add : Tensor "f32[96, 4096, 1][4096, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1e-05), kwargs = {})
#   %div_1 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %add), kwargs = {})
#   %div_2 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, %add), kwargs = {})
#   %mul : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %div_2), kwargs = {})
#   %div_3 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %add), kwargs = {})
#   %sum_2 : Tensor "f32[96, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [2], True), kwargs = {dtype: torch.float32})
#   %div_4 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %pow_2), kwargs = {})
#   %eq : Tensor "b8[96, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_2, 0), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div_4), kwargs = {})
#   %mul_1 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %where), kwargs = {})
#   %add_1 : Tensor "f32[96, 4096, 192][786432, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %mul_1), kwargs = {})
#   return %sum_2,%add_1
triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_0 = async_compile.triton('triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '4E7A6778FA033C9540BD4516D4C3B4EF4F52F0426F716BA619676A203337279F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864, 'r0_': 1207959552}}
)
@triton.jit
def triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 393216
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 192*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp2 / tmp5)
    tmp7 = (tmp6 / tmp5)
    tmp8 = tmp1 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = (tmp0 / tmp5)
    tmp14 = 0.0
    tmp15 = tmp3 == tmp14
    tmp16 = (tmp2 / tmp3)
    tmp17 = tl.where(tmp15, tmp14, tmp16)
    tmp18 = tmp12 * tmp17
    tmp19 = tmp13 + tmp18
    tl.store(out_ptr1 + (r0_1 + 192*x0), tmp19, r0_mask)
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
        primals_1, pow_2, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (96, 4096, 192), (786432, 192, 1))
        assert_size_stride(pow_2, (96, 4096, 1), (4096, 1, 1))
        assert_size_stride(tangents_1, (96, 4096, 192), (786432, 192, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((96, 4096, 192), (786432, 192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.neg, aten.add, aten.div, aten.mul, aten.sum, aten.eq, aten.masked_fill]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_0.run(tangents_1, primals_1, pow_2, buf1, 393216, 192, stream=stream0)
            del pow_2
            del primals_1
            del tangents_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((96, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.float32)
    pow_2 = rand_strided((96, 4096, 1), (4096, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((96, 4096, 192), (786432, 192, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, pow_2, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
