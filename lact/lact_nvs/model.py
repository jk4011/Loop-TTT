# LaCT

import math
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from lact_ttt import TTTOperator

def get_class_by_name(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
        out_gate=False,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.causal = causal
        # LT2 SDPA output gate: per-head data-dependent sigmoid gate on the
        # attention output (fixes attention-sink / residual-RMS blowup that
        # compounds when the tied attention runs every loop pass). zero-init ->
        # 2*sigmoid(0)=1 -> exact identity at start.
        self.out_gate = nn.Linear(dim, self.num_heads, bias=True) if out_gate else None
        if out_gate:
            nn.init.zeros_(self.out_gate.weight)
            nn.init.zeros_(self.out_gate.bias)

    def forward(self, x, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        if self.out_gate is not None:
            g = 2.0 * torch.sigmoid(self.out_gate(x))            # [b, l, nh]
            out = out * g.transpose(1, 2).unsqueeze(-1)          # per-head scale
        out = rearrange(out, "b nh l dh -> b l (nh dh)")

        out = self.c_proj(out)
        return out, {}


class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, dim, bias, block_config, n_loops_max=1, loop_film=False,
                 loop_gates=False):
        super().__init__()
        module_list = []
        self.length_dim_list = []

        for _, module_config in enumerate(block_config):
            CLASS = get_class_by_name(module_config["type"])
            module = nn.ModuleDict(
                {
                    "ln": LayerNorm(dim, bias=bias),
                    "f": CLASS(dim=dim, bias=bias, **module_config["params"]),
                }
            )

            module_list.append(module)
            self.length_dim_list.append(module_config.get("length_dim", "vl"))

        self.module_list = nn.ModuleList(module_list)

        # Per-loop channel-wise FiLM on each sub-module's LN output (Deja View's
        # per-step scales, found essential for tied-beating-untied there).
        # Zero-init -> exact identity at start. [n_loops, n_submodules, 2, dim]
        self.loop_film = None
        if loop_film:
            self.loop_film = nn.Parameter(
                torch.zeros(n_loops_max, len(module_list), 2, dim)
            )
        # Deja View-style per-loop gates (their Table-4 winners): branch-output
        # gates on each sub-module and a state gate on the accumulated stream at
        # block end. Zero-init -> exact identity at start. Zero FLOPs.
        self.branch_gate = None
        if loop_gates:
            self.branch_gate = nn.Parameter(torch.zeros(n_loops_max, len(block_config), dim))
            self.state_gate = nn.Parameter(torch.zeros(n_loops_max, dim))

    def forward(self, x, info):
        results = {}
        loop_idx = info.get("loop_idx", 0)
        for mod_idx, (module, length_dim) in enumerate(zip(self.module_list, self.length_dim_list)):
            residual = x
            x = module["ln"](x)
            if self.loop_film is not None:
                film = self.loop_film[loop_idx, mod_idx]
                x = x * (1 + film[0]) + film[1]

            if length_dim == "l":
                b, vl, d = x.shape
                l = info["num_img_tokens"]
                x = x.reshape(b * (vl // l), l, d)
                x, result = module["f"](x, info)
                x = x.reshape(b, vl, d)
            else:
                x, result = module["f"](x, info)

            if self.branch_gate is not None:
                x = x * (1 + self.branch_gate[loop_idx, mod_idx])
            x = residual + x
            results.update(result)
        if self.branch_gate is not None:
            x = x * (1 + self.state_gate[loop_idx])
        return x, results



def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d


class LaCTLVSM(nn.Module):
    def __init__(self, patch_size, dim, layers, block_config,
                 n_loops=1, ttt_state_mode="reset", input_injection="none",
                 loop_film=False, view_schedule="all", update_chunks=1,
                 render_feedback=False, target_loops=0, loop_gates=False,
                 input_loops=0, feat_mom=False, geo_addr=False,
                 stream_norm=False, fused_readout=False, drop_loop=0.0):
        """
        Looped-TTT extension of LaCT LVSM.

        layers:          number of UNIQUE Blocks (weight-tied loop unit).
        n_loops:         how many times the whole Block stack is repeated
                         (effective depth = layers * n_loops; n_loops=1 = vanilla LaCT).
        ttt_state_mode:  "reset" — each loop pass re-inits fast weights from the
                         learned init (attention-like stateless looping);
                         "carry" — fast weights from loop i seed loop i+1, so
                         looping performs more inner-loop TTT update steps.
        input_injection: "none" | "add" — re-add the embedded input tokens at the
                         start of every loop pass (looped-transformer input injection).
        """
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.n_loops = n_loops
        assert ttt_state_mode in ("reset", "carry")
        assert input_injection in ("none", "add")
        assert view_schedule in ("all", "incremental")
        self.ttt_state_mode = ttt_state_mode
        self.input_injection = input_injection
        # "incremental": loop pass i updates the fast-weight memory ONLY on the
        # i-th contiguous chunk of input views (incremental scene registration,
        # SfM-style); apply still covers all tokens. Update FLOPs drop n_loops x.
        # Pair with ttt_state_mode=carry so registrations accumulate.
        self.view_schedule = view_schedule
        # update_chunks=K: split each pass's update into K sequential view-chunk
        # steps (rotated by loop index) — K preconditioned steps per pass instead
        # of one large-chunk step. Token FLOPs unchanged; NS cost xK, so pair
        # with a lower muon_update_steps in the TTT params to stay iso-FLOPs.
        assert update_chunks == 1 or view_schedule == "all"
        self.update_chunks = update_chunks
        # render_feedback: decode the target tokens after each pass and re-embed
        # the RGB estimate into the target tokens of the next pass (RAFT-style
        # iterative refinement; zero-init gate+proj -> exact baseline at init).
        self.render_feedback = render_feedback
        # target_loops=T (late-join): target tokens participate only in the LAST
        # T loops; earlier loops run on input tokens only (input-only pass costs
        # ~0.66x of a full pass, and the memory trajectory is unchanged since
        # targets never influence inputs). 0 = targets in every loop (baseline).
        # [r5 result: target read depth matters a lot — late-join FAILED (-1.66).]
        self.target_loops = target_loops
        # input_loops=K (read-heavy / write->read split): input tokens participate
        # only in the FIRST K loops (memory writing); the remaining loops run on
        # target tokens only, re-applying the loop-K fast-weight states (deep
        # iterative read-out). Target-only pass costs ~0.5x of a full pass.
        # 0 = inputs in every loop (baseline). Mutually exclusive with target_loops.
        assert not (target_loops > 0 and input_loops > 0)
        self.input_loops = input_loops

        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]

        self.input_dim = len(self.posed_image_keys) * 3
        self.input_linear = nn.Linear(self.input_dim * (self.patch_size**2), self.dim, bias=False)
        self.input_layernorm = nn.LayerNorm(self.dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(dim=self.dim, bias=False, block_config=block_config,
                  n_loops_max=n_loops, loop_film=loop_film, loop_gates=loop_gates)
            for _ in range(layers)
        ])
        # LT2-style per-loop residual gate on the whole loop body:
        # x <- x + loop_rho[l] * x_loop_start (zero-init).
        self.loop_rho = nn.Parameter(torch.zeros(n_loops, dim)) if loop_gates else None
        # feat_mom: Anderson/Nesterov extrapolation of the feature fixed-point
        # iteration. At loop start, x <- x + beta[loop] * (x - x_prev): the loop's
        # own convergence direction is extrapolated -> fractional extra depth at ~0
        # FLOPs, in feature space where Muon cannot erase it. beta zero-init = baseline.
        self.feat_mom = nn.Parameter(torch.zeros(n_loops, dim)) if feat_mom else None
        # geo_addr: compute per-patch Plücker (ray dir + moment) and feed to the TTT
        # layer so its fast-weight addressing can be conditioned on camera geometry
        # (the epipolar constraint), the axis that gave +1.7 dB in the attention
        # camera-conditioning prior. Injected zero-init in the TTT layer -> baseline.
        self.geo_addr = geo_addr
        # StreamNorm: per-loop gated RMS re-normalization of the residual stream
        # (counters residual saturation so each pass regains leverage). zero-init.
        self.stream_norm = nn.Parameter(torch.zeros(n_loops, dim)) if stream_norm else None
        # Fused-Readout: decode the target from a learned combination of all per-loop
        # target features (output-composition axis). init so last loop dominates.
        self.fused_readout = None
        if fused_readout:
            w = torch.full((n_loops,), -4.0); w[-1] = 4.0
            self.fused_readout = nn.Parameter(w)
        self.drop_loop = drop_loop  # train-time stochastic skip of a loop's contribution

        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (self.patch_size**2) * 3, bias=False),
            nn.Sigmoid(),
        )

        if render_feedback:
            self.render_reembed = nn.Linear((self.patch_size**2) * 3, self.dim, bias=False)
            self.fb_gate = nn.Parameter(torch.zeros(()))

        # apply special scaled init to the residual projections, per GPT-2 paper
        self.apply(_init_weights)
        if render_feedback:
            torch.nn.init.zeros_(self.render_reembed.weight)
        # Residual-stream init scaled by EFFECTIVE depth (layers * n_loops), so a
        # looped model matches the activation statistics of its unrolled equivalent.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(len(block_config) * layers * n_loops))

    def _loop_ttt_op_order(self, loop_idx, num_input_views, num_img_tokens, apply_end):
        num_input_tokens = num_input_views * num_img_tokens
        if self.view_schedule == "incremental" and self.n_loops > 1:
            chunk = max(1, num_input_views // self.n_loops)
            start_v = min(loop_idx * chunk, num_input_views - chunk)
            update_ops = [TTTOperator(
                start=start_v * num_img_tokens,
                end=(start_v + chunk) * num_img_tokens,
                update=True, apply=False,
            )]
        elif self.update_chunks > 1:
            # K sequential view-chunk updates, rotated by loop index so each
            # pass sees a different recency order (views are trajectory-ordered).
            K = self.update_chunks
            chunk = max(1, num_input_views // K)
            starts = [(i * chunk) % num_input_views for i in range(K)]
            rot = loop_idx % K
            update_ops = [TTTOperator(
                start=s * num_img_tokens,
                end=min(s + chunk, num_input_views) * num_img_tokens,
                update=True, apply=False,
            ) for s in starts[rot:] + starts[:rot]]
        else:
            update_ops = [TTTOperator(start=0, end=num_input_tokens, update=True, apply=False)]
        return update_ops + [TTTOperator(start=0, end=apply_end, update=False, apply=True)]

    def forward(self, input_data_dict, target_data_dict, return_all_loops=False,
                n_loops_override=None):
            # Do not autocast during the data processing
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()
            num_target_views = target_data_dict["c2w"].size(1)

            for data_dict in [input_data_dict, target_data_dict]:
                fxfycxcy = data_dict["fxfycxcy"]
                c2w = data_dict["c2w"]

                data_dict["ray_o"], data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
                data_dict["o_cross_d"] = torch.cross(data_dict["ray_o"], data_dict["ray_d"], dim=2)
                data_dict["pose_only"] = torch.concat(
                    [data_dict[key] for key in self.pose_keys], dim=2
                )
                
                if "image" in data_dict:
                    data_dict["normalized_image"] = data_dict["image"] * 2.0 - 1.0

                    # Compile the information for posed-image input, and pose-only input.
                    data_dict["posed_image"] = torch.concat(
                        [data_dict[key] for key in self.posed_image_keys], dim=2
                    )
            
            transformer_input = input_data_dict["image"].new_zeros(
                batch_size, num_input_views + num_target_views, self.input_dim, h, w
            )  
            transformer_input[:, :num_input_views, :, :, :] = input_data_dict["posed_image"]
            pose_only_dim = target_data_dict["pose_only"].size(2)
            transformer_input[:, num_input_views:, :pose_only_dim, :, :] = target_data_dict["pose_only"]

            pose_tokens = None
            if self.geo_addr:
                pose_maps = []
                for dd, nv in [(input_data_dict, num_input_views), (target_data_dict, num_target_views)]:
                    rd = F.avg_pool2d(dd["ray_d"].flatten(0, 1), self.patch_size).unflatten(0, (batch_size, nv))
                    rd = rd / (rd.norm(dim=2, keepdim=True) + 1e-6)          # [b, v, 3, hh, ww]
                    center = dd["c2w"][:, :, :3, 3][..., None, None].expand_as(rd)
                    m = torch.cross(center, rd, dim=2)                      # Plücker moment
                    pm = torch.cat([rd, m], dim=2)                          # [b, v, 6, hh, ww]
                    pose_maps.append(rearrange(pm, "b v c hh ww -> b (v hh ww) c"))
                pose_tokens = torch.cat(pose_maps, dim=1)                   # [b, vl, 6]

        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        num_target_tokens = num_target_views * num_img_tokens
        info = {
            "num_img_tokens": num_img_tokens,
        }
        if self.geo_addr:
            info["pose_tokens"] = pose_tokens
        # n_eff = the ACTUAL number of loop passes this call runs. Defaults to the
        # configured n_loops; a deeper value (distill teacher) or a sampled value
        # (stochastic depth) can override it. Per-loop param lookups clamp to their
        # array size, so a plain loop runs at any depth.
        n_eff = n_loops_override if n_loops_override is not None else self.n_loops
        join_loop = n_eff - self.target_loops if self.target_loops > 0 else 0
        op_order_of_loop = []
        for li in range(n_eff):
            if self.input_loops > 0 and li >= self.input_loops:
                # read-heavy phase: target tokens only, apply-only (memory frozen)
                op_order_of_loop.append(
                    [TTTOperator(start=0, end=num_target_tokens, update=False, apply=True)])
            else:
                op_order_of_loop.append(self._loop_ttt_op_order(
                    li, num_input_views, num_img_tokens,
                    num_input_tokens if li < join_loop else num_input_tokens + num_target_tokens))
        if self.target_loops > 0 or self.input_loops > 0:
            assert self.input_injection == "none"
        if self.target_loops > 0:
            assert not self.render_feedback

        x = rearrange(
            transformer_input,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        x0 = x if self.input_injection == "add" else None
        block_states = [{} for _ in self.blocks]
        saved_states = [{} for _ in self.blocks]
        loop_renders = []
        target_feats = []
        frozen_targets = None
        x_prev_loop = None
        for loop_idx in range(n_eff):
            if self.target_loops > 0:
                if loop_idx == 0 and join_loop > 0:
                    frozen_targets = x[:, num_input_tokens:]
                    x = x[:, :num_input_tokens]
                elif loop_idx == join_loop and frozen_targets is not None:
                    x = torch.cat([x, frozen_targets], dim=1)
                    frozen_targets = None
            read_phase = self.input_loops > 0 and loop_idx >= self.input_loops
            if self.input_loops > 0 and loop_idx == self.input_loops:
                x = x[:, num_input_tokens:]  # drop input tokens; memory is written
            if loop_idx > 0 and x0 is not None:
                x = x + x0
            if self.stream_norm is not None and loop_idx > 0:
                x_hat = x * x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt().to(x.dtype)
                x = x + self.stream_norm[loop_idx] * (x_hat - x)
            if self.feat_mom is not None:
                # loop 0 references feat_mom[0] as a no-op (x_prev=x) so every row
                # enters the autograd graph (DDP requires all params used).
                delta = (x - x_prev_loop) if x_prev_loop is not None else torch.zeros_like(x)
                x = x + self.feat_mom[loop_idx] * delta
                x_prev_loop = x
            x_loop_start = x if self.loop_rho is not None else None
            x_pre_drop = x if (self.training and self.drop_loop > 0) else None
            for block_idx, block in enumerate(self.blocks):
                extra_state = saved_states[block_idx] if read_phase else block_states[block_idx]
                block_info = {**info, "loop_idx": loop_idx,
                              "ttt_op_order": op_order_of_loop[loop_idx],
                              **extra_state}
                x, result = block(x, block_info)
                if self.ttt_state_mode == "carry":
                    block_states[block_idx] = {
                        key: result[key] for key in ("w0", "w1", "w2", "m0", "m1", "m2", "r") if key in result
                    }
                if self.input_loops > 0 and loop_idx == self.input_loops - 1:
                    saved_states[block_idx] = {
                        key: result[key] for key in ("w0", "w1", "w2", "m0", "m1", "m2", "r") if key in result
                    }
            if x_loop_start is not None:
                x = x + self.loop_rho[loop_idx] * x_loop_start
            if x_pre_drop is not None and 0 < loop_idx < n_eff - 1:
                # inverted stochastic depth: skip this interior loop's net contribution
                keep = (torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_loop).to(x.dtype)
                x = x_pre_drop + keep / (1 - self.drop_loop) * (x - x_pre_drop)
            if self.fused_readout is not None:
                target_feats.append(x[:, -num_target_tokens:])
            if loop_idx < n_eff - 1 and frozen_targets is None and (return_all_loops or self.render_feedback):
                render_l = self._decode_targets(
                    x[:, -num_target_tokens:], num_target_views, h, w)
                if return_all_loops:
                    loop_renders.append(render_l)
                if self.render_feedback:
                    fb = rearrange(
                        render_l * 2.0 - 1.0,
                        "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                        ph=self.patch_size, pw=self.patch_size,
                    )
                    fb = self.render_reembed(fb)
                    x = torch.cat([
                        x[:, :-num_target_tokens],
                        x[:, -num_target_tokens:] + torch.tanh(self.fb_gate) * fb,
                    ], dim=1)

        if self.fused_readout is not None and len(target_feats) == n_eff:
            mix = torch.softmax(self.fused_readout[:n_eff], dim=0)
            fused = sum(ml * f for ml, f in zip(mix, target_feats))
            final = self._decode_targets(fused, num_target_views, h, w)
        else:
            final = self._decode_targets(x[:, -num_target_tokens:], num_target_views, h, w)
        if return_all_loops:
            return loop_renders + [final]
        return final

    def _decode_targets(self, target_x, num_target_views, h, w):
        target_x = self.image_token_decoder(target_x)
        return rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
    
    def reconstruct(self, input_data_dict):
        # render_feedback operates on target tokens inside forward(); the
        # split reconstruct/rendering path does not support it yet.
        assert not self.render_feedback, "render_feedback: use forward(), not reconstruct/rendering"
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()

            fxfycxcy = input_data_dict["fxfycxcy"]
            c2w = input_data_dict["c2w"]

            input_data_dict["ray_o"], input_data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            input_data_dict["o_cross_d"] = torch.cross(input_data_dict["ray_o"], input_data_dict["ray_d"], dim=2)
            input_data_dict["pose_only"] = torch.concat(
                [input_data_dict[key] for key in self.pose_keys], dim=2
            )
                
            input_data_dict["normalized_image"] = input_data_dict["image"] * 2.0 - 1.0

            # Compile the information for posed-image input, and pose-only input.
            posed_image = torch.concat(
                [input_data_dict[key] for key in self.posed_image_keys], dim=2
            )
            
        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        info = {
            "num_img_tokens": num_img_tokens,
        }
        op_order_of_loop = [
            self._loop_ttt_op_order(li, num_input_views, num_img_tokens, num_input_tokens)
            for li in range(self.n_loops)
        ]

        x = rearrange(
            posed_image,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        x0 = x if self.input_injection == "add" else None
        # states are saved in VISIT order (n_loops * layers entries); rendering
        # replays the same unrolled sequence.
        states = []
        block_states = [{} for _ in self.blocks]
        for loop_idx in range(self.n_loops):
            if loop_idx > 0 and x0 is not None:
                x = x + x0
            x_loop_start = x if self.loop_rho is not None else None
            for block_idx, block in enumerate(self.blocks):
                block_info = {**info, "loop_idx": loop_idx,
                              "ttt_op_order": op_order_of_loop[loop_idx],
                              **block_states[block_idx]}
                x, state = block(x, block_info)
                if self.ttt_state_mode == "carry":
                    block_states[block_idx] = {
                        key: state[key] for key in ("w0", "w1", "w2", "m0", "m1", "m2") if key in state
                    }
                states.append(state)
            if x_loop_start is not None:
                x = x + self.loop_rho[loop_idx] * x_loop_start
        return states
    
    def rendering(self, target_data_dict, states, h, w):
        with torch.autocast(device_type="cuda", enabled=False):
            batch_size, num_target_views, _, _ = target_data_dict["c2w"].size()

            fxfycxcy = target_data_dict["fxfycxcy"]
            c2w = target_data_dict["c2w"]

            target_data_dict["ray_o"], target_data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            target_data_dict["o_cross_d"] = torch.cross(target_data_dict["ray_o"], target_data_dict["ray_d"], dim=2)
            target_data_dict["pose_only"] = torch.concat(
                [target_data_dict[key] for key in self.pose_keys], dim=2
            )

            pose_only = target_data_dict["pose_only"].new_zeros(
                batch_size, num_target_views, self.input_dim, h, w
            )  
            pose_only_dim = target_data_dict["pose_only"].size(2)
            pose_only[:, :, :pose_only_dim, :, :] = target_data_dict["pose_only"]
            
        # Running the model for rendering
        num_img_tokens = h * w // (self.patch_size**2)
        num_target_tokens = num_target_views * num_img_tokens
        ttt_op_order = [
            TTTOperator(start=0, end=num_target_tokens, update=False, apply=True),
        ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }

        # Process each target view separately
        all_x = []
        for v in range(num_target_views):
            single_view_pose = pose_only[:, v:v+1]  # b, 1, c, h, w
            
            x = rearrange(
                single_view_pose,
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size,
            )
            x = self.input_linear(x)
            x = self.input_layernorm(x)
            x0 = x if self.input_injection == "add" else None

            # Apply the saved states from reconstruction, replaying the same
            # unrolled (loop, block) visit order used in reconstruct(). With
            # late-join (target_loops > 0), targets only run the last T loops,
            # matching forward().
            assert len(states) == self.n_loops * len(self.blocks)
            start_visit = 0
            if self.target_loops > 0:
                start_visit = (self.n_loops - self.target_loops) * len(self.blocks)
            x_loop_start = None
            for visit_idx, state in enumerate(states):
                if visit_idx < start_visit:
                    continue
                if visit_idx > start_visit and visit_idx % len(self.blocks) == 0 and x0 is not None:
                    x = x + x0
                if visit_idx % len(self.blocks) == 0 and self.loop_rho is not None:
                    x_loop_start = x
                block = self.blocks[visit_idx % len(self.blocks)]
                info["loop_idx"] = visit_idx // len(self.blocks)
                info.update(state)
                x, _ = block(x, info)
                if x_loop_start is not None and visit_idx % len(self.blocks) == len(self.blocks) - 1:
                    x = x + self.loop_rho[info["loop_idx"]] * x_loop_start

            all_x.append(x)
        
        # Concatenate all processed views
        x = torch.cat(all_x, dim=1)
            
        # Generate target images
        target_x = self.image_token_decoder(x)
        target_x = rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        
        return target_x

    