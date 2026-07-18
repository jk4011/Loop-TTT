"""Looped-TTT research layer (lact_ttt.py is the untouched baseline).

LoopFastWeightGluMLPMultihead adds loop-aware mechanisms selected by a single
`loop_mode` string in the YAML: a "+"-joined set of flags, e.g. "lrs", "rho",
"lrs+rho", "delta". Implemented flags (see IDEAS.md):

  lrs   (I4) per-loop learnable lr bias: lr = softplus(lr_fc(x) + base_lr_inv
        + loop_lr_bias[loop_idx]). Zero-init -> exact baseline at start.
  rho   (I2) residual-gated lr: scale per-token lr by rho = 1 - cos(f_w(k), v),
        detached. Update magnitude becomes proportional to the current misfit,
        restoring the contraction property that Muon/NS normalization erases.
  delta (I3) delta writes: regress onto the innovation v - f_w(k) instead of v,
        so already-explained content is not re-written (fixed point at perfect fit).

All flags compose; kernel differences are confined to _loop_fast_weight_apply.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from lact_ttt import (
    FastWeightGluMLPMultihead,
    silu_backprop,
    zeropower_via_newtonschulz5,
)


@torch.compile
def _loop_fast_weight_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_ua_order,
    muon_update_steps: int = 0,
    rho_gate: bool = False,
    delta: bool = False,
):
    """Variant of fast_weight_swish_glu_weight_norm_mini_batch_apply with
    residual-gated lr (rho_gate) and/or delta-write targets (delta).
    Identical to the baseline kernel when both flags are False."""
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            if rho_gate or delta:
                # Current memory prediction for the update keys (one extra bmm).
                f_k = hidden @ w1_now  # [b, l, d]

            if rho_gate:
                rho = 1.0 - F.cosine_similarity(f_k, vi, dim=-1).unsqueeze(-1)
                rho = rho.detach().to(lr0i.dtype)
                lr0i = lr0i * rho
                lr1i = lr1i * rho
                lr2i = lr2i * rho

            vt = vi - f_k if delta else vi

            dhidden = vt @ w1_now.transpose(-1, -2)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (hidden * lr1i).transpose(-1, -2) @ vt, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)

    return output, w0, w1, w2


class LoopFastWeightGluMLPMultihead(FastWeightGluMLPMultihead):

    def __init__(self, *args, loop_mode="none", n_loops_max=8, **kwargs):
        super().__init__(*args, **kwargs)
        flags = set(loop_mode.split("+")) if loop_mode != "none" else set()
        known = {"lrs", "rho", "delta"}
        assert flags <= known, f"unknown loop_mode flags: {flags - known}"
        self.loop_flags = flags
        self.n_loops_max = n_loops_max

        if "lrs" in flags:
            # One bias per (loop, lr-slot); zero-init = baseline schedule.
            self.loop_lr_bias = nn.Parameter(torch.zeros(n_loops_max, 3))

    def forward(self, x: torch.Tensor, info={}, *args):
        qkv = F.silu(self.to_qkv(x), inplace=True)
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, 3 * lr_dim]

        lr = lr.float() + self.base_lr_inv
        if "lrs" in self.loop_flags:
            loop_idx = min(info.get("loop_idx", 0), self.n_loops_max - 1)
            bias = self.loop_lr_bias[loop_idx].float()          # [3]
            lr = lr + bias.repeat_interleave(self.lr_dim)[None, None, :]
        lr = torch.nn.functional.softplus(lr)
        lr0, lr1, lr2 = rearrange(
            lr, "b l (lrs h d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads
        )

        if "w0" in info:
            assert "w1" in info and "w2" in info
            w0, w1, w2 = info["w0"], info["w1"], info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        output, w0, w1, w2 = _loop_fast_weight_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
            muon_update_steps=self.muon_update_steps,
            rho_gate="rho" in self.loop_flags,
            delta="delta" in self.loop_flags,
        )

        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}

    def extra_repr(self) -> str:
        return super().extra_repr() + f"loop_mode: {sorted(self.loop_flags)}, "
