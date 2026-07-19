"""Looped-TTT research layer (lact_ttt.py is the untouched baseline).

LoopFastWeightGluMLPMultihead adds loop-aware mechanisms selected by a single
`loop_mode` string in the YAML: a "+"-joined set of flags, e.g. "lrs", "rho",
"lrs+rho", "delta". Implemented flags (see IDEAS.md):

  lrs   (I4) per-loop learnable lr bias: lr = softplus(lr_fc(x) + base_lr_inv
        + loop_lr_bias[loop_idx]). Zero-init -> exact baseline at start.
  rho   (I2) residual-gated lr: scale per-token lr by rho = 1 - cos(f_w(k), v),
        detached. NOTE (probe finding, r2): lr multiplies INSIDE the NS argument,
        and NS re-normalizes the result -> magnitude info erased downstream.
        rho therefore only reweights token directions; PSNR-neutral in r2.
  rho2  (I2') post-NS residual scaling: multiply each NS-orthogonalized gradient
        by the chunk-level misfit rho_bar = mean(1 - cos(f_w(k), v)) in [0, 2],
        AFTER orthogonalization. This is the corrected fix for the measured
        pathology (dw1_rel does not decay across carry loops): steps now shrink
        as the fit closes -> contraction instead of constant-angle orbit.
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
    rho_post: bool = False,
    delta: bool = False,
    update_epochs: int = 1,
    wp0=None, wp1=None, wp2=None,
    read_refine: float = 0.0,
    precond_w1: int = 0,
    precond_lambda: float = 0.1,
    cumboost: bool = False,
    r_in=None,
    epavg: bool = False,
    gate_bias=None,
):
    """Variant of fast_weight_swish_glu_weight_norm_mini_batch_apply.

    Extra loop-aware knobs (all reduce to the baseline kernel when off):
      delta         regress onto v - f_w(k) instead of v.
      update_epochs E>1: apply the TTT update E times on the SAME (k, v) chunk,
                    each on the evolving weights (drift-free multi-step descent;
                    directly attacks the measured single-step underfit). Attention
                    /MLP/qkv are NOT recomputed, so cost ~ +1/3 TTT per extra epoch.
      wp0/1/2       cross-loop BOOSTING: previous loop's converged fast weights.
                    Their SwiGLU prediction on the current keys is subtracted from
                    the update target, so this fresh memory only stores the residual
                    the previous memory left (effective capacity x n_loops). Weights
                    are still init-fresh (reset), so no carry re-write pathology.
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)
    boost = wp0 is not None
    r_out = None

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]

            if cumboost and r_in is not None:
                # Cumulative-residual boost: regress onto the running residual the
                # ensemble-so-far left, carried in token space (proper stagewise
                # gradient boosting; fixes boost's last-only double-counting).
                vi = r_in
            elif boost:
                # Residual target: what the previous loop's memory could not explain.
                pred_prev = (F.silu(ki @ wp0, inplace=False) * (ki @ wp2)) @ wp1
                vi = vi - pred_prev

            w1_acc = None  # epavg: running sum of w1 iterates (Polyak averaging)
            for _ in range(update_epochs):
                gate_before_act = ki @ w0_now
                if gate_bias is not None:
                    gate_before_act = gate_before_act + gate_bias
                hidden_before_mul = ki @ w2_now
                hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

                if rho_gate or rho_post or delta:
                    # Current memory prediction for the update keys (one extra bmm).
                    f_k = hidden @ w1_now  # [b, l, d]

                lr0e, lr1e, lr2e = lr0i, lr1i, lr2i
                if rho_gate:
                    rho = 1.0 - F.cosine_similarity(f_k, vi, dim=-1).unsqueeze(-1)
                    rho = rho.detach().to(lr0i.dtype)
                    lr0e, lr1e, lr2e = lr0i * rho, lr1i * rho, lr2i * rho

                if rho_post:
                    rho_bar = (
                        1.0 - F.cosine_similarity(f_k, vi, dim=-1).mean(dim=1)
                    ).detach()[:, None, None]

                vt = vi - f_k if delta else vi

                dhidden = vt @ w1_now.transpose(-1, -2)
                dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                dgate = dhidden * hidden_before_mul
                dgate_before_act = silu_backprop(dgate, gate_before_act)

                w1_raw = (hidden * lr1e).transpose(-1, -2) @ vt  # [b, dh, d] raw grad
                if precond_w1 < 0:
                    # Diagonal (Jacobi) preconditioner: divide each row of the raw
                    # grad by that hidden unit's second moment -> guaranteed
                    # non-identity anisotropy (decisive test of whether conditioning
                    # the w1 DIRECTION helps PSNR at all). Elementwise, ~0 FLOPs.
                    Hf = hidden.float()
                    d = (Hf * Hf).mean(dim=1, keepdim=True).transpose(-1, -2)  # [b, dh, 1]
                    w1_raw = (w1_raw.float() / (d + precond_lambda * d.mean(dim=1, keepdim=True) + 1e-8)).to(w1_raw.dtype)
                elif precond_w1 > 0:
                    # Gauss-Newton / RLS readout: w1 is a LINEAR map from hidden H,
                    # so its optimal update preconditions the raw grad by (HᵀH+λI)⁻¹.
                    # Solve via Richardson iteration (operator form, no dh×dh matrix):
                    # M y = HᵀHy + λy computed as Hᵀ(Hy)+λy. NS is scale-invariant, so
                    # only the ANISOTROPY of the preconditioner survives -> escapes the
                    # dead lr-knob trap. precond_w1=0 -> exact baseline.
                    Hf = hidden.float()
                    hnorm2 = (Hf * Hf).sum(dim=(1, 2), keepdim=True)  # ||H||_F^2 >= max eig
                    lam = precond_lambda * hnorm2 / hidden.shape[-1]
                    c = hnorm2 + lam
                    y = w1_raw.float() / c
                    g = w1_raw.float()
                    for _ in range(precond_w1):
                        My = Hf.transpose(-1, -2) @ (Hf @ y) + lam * y
                        y = y + (g - My) / c
                    w1_raw = y.to(w1_raw.dtype)
                w1_grad = zeropower_via_newtonschulz5(w1_raw, muon_update_steps)
                w0_grad = zeropower_via_newtonschulz5(
                    (ki * lr0e).transpose(-1, -2) @ dgate_before_act, muon_update_steps
                )
                w2_grad = zeropower_via_newtonschulz5(
                    (ki * lr2e).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
                )
                if rho_post:
                    w1_grad = w1_grad * rho_bar
                    w0_grad = w0_grad * rho_bar
                    w2_grad = w2_grad * rho_bar
                w1_now = w1_now + w1_grad
                w0_now = w0_now + w0_grad
                w2_now = w2_now + w2_grad

                w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
                w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
                w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
                if epavg:
                    w1_acc = w1_now if w1_acc is None else w1_acc + w1_now

            if epavg and w1_acc is not None:
                # Polyak/Ruppert: apply from the MEAN of the w1 iterates (orbit
                # center) rather than the last (overshooting) iterate. w1 only —
                # nonlinear w0/w2 averaging is ill-defined.
                w1_now = w1_acc / update_epochs
                w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm

            if cumboost:
                # New running residual: what THIS memory left unexplained (detached
                # so it only sets the next loop's target, not a backprop path).
                h_now = (F.silu(ki @ w0_now, inplace=False) * (ki @ w2_now)) @ w1_now
                r_out = (vi - h_now).detach()

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            gate_q = qi @ w0_now
            if gate_bias is not None:
                gate_q = gate_q + gate_bias
            oi = (F.silu(gate_q, inplace=False) * (qi @ w2_now)) @ w1_now
            if read_refine != 0.0:
                # Read-side refinement: re-query the same memory with a query nudged
                # by its own first read (per-head, dims match), then renormalize.
                # The untested APPLY axis; one extra SwiGLU on the read tokens.
                qi2 = qi + read_refine * oi
                qi2 = qi2 / (qi2.norm(dim=2, keepdim=True) + 1e-5)
                oi = (F.silu(qi2 @ w0_now, inplace=False) * (qi2 @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)

    return output, w0, w1, w2, r_out


@torch.compile
def _loop_momentum_apply(
    w0i, w1i, w2i, q, k, v, lr0, lr1, lr2, ttt_ua_order,
    mp0, mp1, mp2, mu: float,
    muon_update_steps: int = 0, delta: bool = False,
):
    """Cross-loop MOMENTUM (heavy-ball on the fast-weight operator).

    The weights ALWAYS start from the fresh init w*i each loop (no orbiting W),
    but the pre-Newton-Schulz gradient momentum M carries across loops:
        M_l  = mu * M_{l-1} + raw_grad_l
        W_l  = weight_norm( W_init + NS(M_l) )
    Progress lives in the accumulating direction M, not in the (non-convergent)
    orbiting weights -- as M's direction stabilizes across loops, W_l converges
    to a better fit than any single crude step. This restores LaCT's own
    momentum recipe (App. A Eq. 20) along the loop axis. Returns the updated M.
    """
    w0_norm = w0i.detach().norm(dim=1, keepdim=True)
    w1_norm = w1i.detach().norm(dim=1, keepdim=True)
    w2_norm = w2i.detach().norm(dim=1, keepdim=True)
    m0, m1, m2 = mp0, mp1, mp2
    have_m = mp0 is not None
    # Current weights persist across segments within this call (updated in the
    # update segment, read in the apply segment). Start from the fresh init.
    w0_cur, w1_cur, w2_cur = w0i, w1i, w2i

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0_cur, w1_cur, w2_cur

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]

            gate_before_act = ki @ w0i
            hidden_before_mul = ki @ w2i
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            if delta:
                f_k = hidden @ w1i
                vt = vi - f_k
            else:
                vt = vi

            dhidden = vt @ w1i.transpose(-1, -2)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            g1 = (hidden * lr1i).transpose(-1, -2) @ vt
            g0 = (ki * lr0i).transpose(-1, -2) @ dgate_before_act
            g2 = (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul

            m0 = mu * m0 + g0 if have_m else g0
            m1 = mu * m1 + g1 if have_m else g1
            m2 = mu * m2 + g2 if have_m else g2
            have_m = True

            w1_now = w1i + zeropower_via_newtonschulz5(m1, muon_update_steps)
            w0_now = w0i + zeropower_via_newtonschulz5(m0, muon_update_steps)
            w2_now = w2i + zeropower_via_newtonschulz5(m2, muon_update_steps)

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
            w0_cur, w1_cur, w2_cur = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_cur, inplace=False) * (qi @ w2_cur)) @ w1_cur
            output.append(oi)

    output = torch.cat(output, dim=1)
    return output, m0, m1, m2


class LoopFastWeightGluMLPMultihead(FastWeightGluMLPMultihead):

    def __init__(self, *args, loop_mode="none", n_loops_max=8,
                 update_epochs=1, per_loop_init=False, read_refine=0.0,
                 momentum=0.0, precond_w1=0, precond_lambda=0.1, epavg=False,
                 muon_schedule=None, key_center=0.0, loop_temp=False,
                 geo_addr=False, qkv_route=0, rot_bag=False, nl_cond=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.epavg = epavg
        # NL-Cond: per-loop additive bias on the SwiGLU gate preactivation
        # (silu(k@w0 + b_l)). Changes each loop's NONLINEAR operating point ->
        # each pass computes a genuinely different SwiGLU. Additive (not scale),
        # so not erased by weight-norm/Muon; orthogonal to gates (linear channel
        # scaling). zero-init -> baseline. head_dim==dim so d_h = inter_multi*dim.
        self.nl_cond = nl_cond
        if nl_cond:
            d_h = self.w0.shape[2]
            self.loop_gate_bias = nn.Parameter(torch.zeros(n_loops_max, d_h))
        # RotBag: fixed per-loop random orthogonal rotation of q,k (same R for both,
        # v unrotated). Each loop fits+reads its fresh memory in a different frame;
        # silu is not rotation-equivariant so each pass computes a decorrelated
        # function -> ensemble/bagging diversity. 0 trainable params, norm-preserving.
        self.rot_bag = rot_bag
        if rot_bag:
            d = self.w0.shape[1]
            gcpu = torch.Generator().manual_seed(1234)
            R = torch.stack([torch.linalg.qr(torch.randn(d, d, generator=gcpu))[0]
                             for _ in range(n_loops_max)])
            self.register_buffer("loop_rot", R)
        # QKV-Route: per-loop low-rank (LoRA) adapter on to_qkv, so each loop pass
        # addresses the fast-weight memory through a slightly different q/k/v map.
        # Keeps the shared meta-learned to_qkv (unlike pli); B zero-init -> baseline.
        # New transform/addressing axis, orthogonal to gates (diagonal conditioning).
        self.qkv_route = qkv_route
        if qkv_route > 0:
            self.qkv_a = nn.Parameter(torch.zeros(n_loops_max, self.dim, qkv_route))
            self.qkv_b = nn.Parameter(torch.zeros(n_loops_max, qkv_route, 3 * self.dim))
            nn.init.normal_(self.qkv_a, std=0.02)  # A random, B zero -> product zero
        # geo_addr: condition the fast-weight q/k on per-patch Plücker geometry
        # (ray dir + moment, 6-d). Reciprocal-product structure: q gets [m; d],
        # k gets [d; m] (swapped) so q·k contains the epipolar term d_t·m_i + m_t·d_i.
        # Zero-init gains -> exact baseline; the memory can learn epipolar addressing.
        self.geo_addr = geo_addr
        if geo_addr:
            self.geo_q = nn.Linear(6, self.dim, bias=False)
            self.geo_k = nn.Linear(6, self.dim, bias=False)
            nn.init.zeros_(self.geo_q.weight)
            nn.init.zeros_(self.geo_k.weight)
        self.key_center = key_center  # >0 enables DC-decorrelation of keys
        if key_center > 0.0:
            self.key_center_gate = nn.Parameter(torch.zeros(1))  # zero-init = baseline
        # loop_temp: per-loop learned q/k temperature. silu(tau*x@w0) changes the
        # gate PATTERN (not just magnitude), so this reshapes the stored+read
        # function per loop -> bandwidth/scale-space axis that escapes the lr-trap.
        self.loop_temp = loop_temp
        if loop_temp:
            inv_sp1 = 1.0 + math.log(-math.expm1(-1.0))  # inv_softplus(1) -> tau=1
            self.loop_temp_raw = nn.Parameter(torch.full((n_loops_max,), inv_sp1))
        # Per-loop Newton-Schulz step schedule (coarse-to-fine spectral: fewer NS
        # steps = top-heavy/low-freq update, more = whitened/high-freq). Keep the
        # sum equal to baseline for iso-FLOPs. NS changes update SHAPE not magnitude,
        # so this escapes the dead lr-knob trap. None = fixed muon_update_steps.
        self.muon_schedule = list(muon_schedule) if muon_schedule is not None else None
        self.read_refine = read_refine  # apply-side re-query strength (0 = off)
        self.momentum = momentum        # cross-loop heavy-ball coefficient (0 = off)
        self.precond_w1 = precond_w1    # Gauss-Newton/RLS Richardson iters on w1 (0 = off)
        self.precond_lambda = precond_lambda
        flags = set(loop_mode.split("+")) if loop_mode != "none" else set()
        known = {"lrs", "rho", "rho2", "delta", "boost", "cumboost"}
        assert flags <= known, f"unknown loop_mode flags: {flags - known}"
        self.loop_flags = flags
        self.n_loops_max = n_loops_max
        self.update_epochs = update_epochs  # inner TTT steps per pass on same chunk

        if "lrs" in flags:
            # One bias per (loop, lr-slot); zero-init = baseline schedule.
            self.loop_lr_bias = nn.Parameter(torch.zeros(n_loops_max, 3))

        # per_loop_init: give each loop POSITION its own learned fast-weight init,
        # instead of one shared self.w0 that must compromise across all loop depths.
        # Adds parameters in the MEMORY space (the binding resource), 0 extra FLOPs.
        # Reset mode only (each pass inits from its own w0_l).
        # Loop 0 reuses the base self.w0/w1/w2; positions 1..n_loops_max-1 get
        # their own extra inits (so self.w0 stays used -> no DDP unused-param error).
        self.per_loop_init = per_loop_init
        if per_loop_init and n_loops_max > 1:
            g = math.sqrt(2)
            d_in = d_out = self.w0.shape[1]
            d_h = self.w0.shape[2]
            nh = self.num_heads
            extra = n_loops_max - 1
            self.w0_loop = nn.Parameter(torch.randn(extra, nh, d_in, d_h) * g / math.sqrt(d_in))
            self.w1_loop = nn.Parameter(torch.randn(extra, nh, d_h, d_out) * g / math.sqrt(d_h))
            self.w2_loop = nn.Parameter(torch.randn(extra, nh, d_in, d_h) * g / math.sqrt(d_in))

    def forward(self, x: torch.Tensor, info={}, *args):
        qkv_pre = self.to_qkv(x)
        if self.qkv_route > 0:
            li = min(info.get("loop_idx", 0), self.n_loops_max - 1)
            qkv_pre = qkv_pre + (x @ self.qkv_a[li]) @ self.qkv_b[li]
        qkv = F.silu(qkv_pre, inplace=True)
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        if self.rot_bag:
            Rl = self.loop_rot[min(info.get("loop_idx", 0), self.n_loops_max - 1)].to(q.dtype)
            q = q @ Rl
            k = k @ Rl

        if self.key_center > 0.0:
            # DC decorrelation: remove the common-mode of the keys so patches that
            # share a huge common component (sky/walls) stop colliding in the
            # associative memory -> raises the usable rank of a single instance.
            kc = self.key_center_gate * k.mean(dim=1, keepdim=True)
            k = k - kc
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5)

        if self.loop_temp:
            tau = F.softplus(self.loop_temp_raw[min(info.get("loop_idx", 0), self.n_loops_max - 1)])
            q = q * tau
            k = k * tau

        if self.geo_addr and info.get("pose_tokens", None) is not None:
            # pose_tokens [b, l, 6] = [ray_d(3), moment(3)]. Swap for reciprocal
            # product: q reads with [m; d], k writes with [d; m]. num_heads==1 here
            # so (b h) l d == b l d and the projection aligns directly.
            pose = info["pose_tokens"].to(q.dtype)
            pose_swap = torch.cat([pose[..., 3:], pose[..., :3]], dim=-1)
            q = q + self.geo_q(pose_swap)
            k = k + self.geo_k(pose)
            q = q / (q.norm(dim=2, keepdim=True) + 1e-5)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5)

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

        boost = "boost" in self.loop_flags
        cumboost = "cumboost" in self.loop_flags
        wp0 = wp1 = wp2 = None
        if boost or cumboost:
            # Boosting: always update from the fresh learned init. boost carries the
            # previous WEIGHTS (as residual-prediction source); cumboost carries the
            # running RESIDUAL vector r (info["r"]) — fresh init either way.
            if boost and "w0" in info:
                wp0, wp1, wp2 = info["w0"], info["w1"], info["w2"]
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)
        elif "w0" in info:
            assert "w1" in info and "w2" in info
            w0, w1, w2 = info["w0"], info["w1"], info["w2"]
        elif self.per_loop_init:
            li = min(info.get("loop_idx", 0), self.n_loops_max - 1)
            if li == 0:
                w0 = self.w0.repeat(x.shape[0], 1, 1)
                w1 = self.w1.repeat(x.shape[0], 1, 1)
                w2 = self.w2.repeat(x.shape[0], 1, 1)
            else:
                w0 = self.w0_loop[li - 1].repeat(x.shape[0], 1, 1)
                w1 = self.w1_loop[li - 1].repeat(x.shape[0], 1, 1)
                w2 = self.w2_loop[li - 1].repeat(x.shape[0], 1, 1)
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        if self.momentum > 0.0:
            # Cross-loop heavy-ball: weights reset to fresh init (w0/w1/w2 above),
            # momentum M carried via info["m0/1/2"].
            output, m0, m1, m2 = _loop_momentum_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
                info.get("m0", None), info.get("m1", None), info.get("m2", None),
                self.momentum, muon_update_steps=self.muon_update_steps,
                delta="delta" in self.loop_flags,
            )
            state = {"m0": m0, "m1": m1, "m2": m2}
        else:
            muon_steps = self.muon_update_steps
            if self.muon_schedule is not None:
                muon_steps = self.muon_schedule[min(info.get("loop_idx", 0), len(self.muon_schedule) - 1)]
            output, w0, w1, w2, r_out = _loop_fast_weight_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
                muon_update_steps=muon_steps,
                update_epochs=self.update_epochs,
                wp0=wp0, wp1=wp1, wp2=wp2,
                read_refine=self.read_refine,
                precond_w1=self.precond_w1,
                precond_lambda=self.precond_lambda,
                rho_gate="rho" in self.loop_flags,
                rho_post="rho2" in self.loop_flags,
                delta="delta" in self.loop_flags,
                cumboost=cumboost,
                r_in=info.get("r", None) if cumboost else None,
                epavg=self.epavg,
                gate_bias=(self.loop_gate_bias[min(info.get("loop_idx", 0), self.n_loops_max - 1)]
                           if self.nl_cond else None),
            )
            state = {"w0": w0, "w1": w1, "w2": w2}
            if cumboost and r_out is not None:
                state["r"] = r_out

        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        output = self.c_proj(output)
        return output, state

    def extra_repr(self) -> str:
        return super().extra_repr() + f"loop_mode: {sorted(self.loop_flags)}, "
