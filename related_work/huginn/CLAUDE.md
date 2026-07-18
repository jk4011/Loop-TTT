# Huginn — Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (arXiv:2502.05171, NeurIPS 2025, Geiping et al.)

Folder: paper PDF + official code `recurrent-pretraining/` (github.com/seal-rg/recurrent-pretraining;
full pretraining code, model in `recpre/`).

**Why it matters to us: the only scale-proven (3.5B params / 800B tokens) loop-training toolkit.
Their stabilizers are the difference between our runs training or collapsing.**

## Architecture
- Three groups, shape **(l_P, l_R, l_C) = (2, 4, 2)**, h=5280: prelude P embeds input -> e;
  recurrent core R iterated r times; coda C decodes. Effective depth 2+4r+2 = 132 at r=32.
- Forward: `e = P(x); s_0 ~ N(0, sigma_s^2 I); s_i = R(e, s_{i-1}); p = C(s_r)`.
- RoPE base 50000; QK biases (nowhere else); gated SiLU MLP; RMSNorm everywhere; tied embeddings.

## The training-stabilizer recipe (document precisely — all four were load-bearing)
1. **Sandwich norm, 4 norms per layer** (learned RMSNorm):
   `x^ = n2(x + Attn(n1(x)));  x = n4(x^ + MLP(n3(x^)))`, plus an RMSNorm n_c at core output.
   At small scale any norm scheme works; at scale sandwich was REQUIRED (see failure log below).
2. **Input injection every iteration** via learned adapter A: R^{2h} -> R^h on **concat(s_i, e)**
   (concat > add at scale; parameter-free `s+e` was in the failed run). Rationale: like gradient
   descent, the iterative operator must see the data every step, else the map can't be a
   data-dependent contraction. Because e is re-injected, the prelude gets gradients at every step
   even under truncated backprop.
3. **Random state init**: s_0 ~ truncated normal (3 sigma), **sigma_s^2 = 2/5** — promotes path
   independence (convergence to input-dependent fixed point regardless of s_0). General init:
   sigma_h^2 = 2/(5h); out-projections sigma_out^2 = 2/(5 h l) with l = l_P + r*l_R + l_C = 132;
   embedding output scaled by sqrt(h).
4. **Loop-count sampling, log-normal Poisson** targeting mean r~=32:
   `tau ~ N(log(r_bar) - sigma^2/2, sigma), sigma = 1/2; r ~ Poisson(e^tau) + 1`
   (mode ~24, median ~29, mean ~33; heavy tail of deep runs). **Locked-step sampling**: one r per
   micro-batch, synchronized across workers (else stragglers idle the cluster).
5. **Truncated BPTT through only the last k=8 iterations** — activation memory independent of the
   sampled r; fixed k beats sampled k on memory uniformity, works as well.
- Deliberate negatives: noise injection into s (diffusion-style) did not help; **step-index
  conditioning R_i broke path independence and killed extrapolation** (conflicts with
  Deja View/RAPTOR conditioning — choose based on whether K-extrapolation is a goal).

## Failure log (Sec. 4.3, Fig. 5 — memorize this)
- Bad Run 1: sandwich structure but parameter-free RMSNorm, no embedding scale, adapter = s+e,
  peak LR 4e-4 -> **token-correlation collapse** (hidden states identical across tokens; each
  recurrence iteration mixes the sequence toward collapse).
- Bad Run 2: pre-norm + learned adapter + embedding scale -> recovers from collapse but **learns to
  ignore s**: val PPL identical at r=1 and r=32 (dead recurrence local minimum).
- Main run: sandwich (learned norms) + learned concat adapter + embedding scale + **peak LR 4e-5**
  -> stable for 750B+ tokens, no loss spikes, recurrence actually used.

## Key numbers
- 3.5B params (1.5B in core), 795B tokens, AMD Frontier (4096 GPUs). Benchmarks improve
  monotonically with test-time r, saturating near r=32; emergent per-token adaptive compute and
  latent-space orbits/sliders; zero-shot continuous CoT via state reuse.

## What we borrow / compare against
1. Port stabilizers to the looped LaCT block: sandwich-norm it, inject the block input each loop
   via concat-adapter, random-init the loop state, sample loop counts (their log-normal Poisson or
   Deja View's Beta — beware conditioning conflict above), truncated BPTT k~=8 if memory binds.
2. Their two failure modes are our diagnostics: track token correlation (collapse) and
   PPL(r=1) vs PPL(r=K) (recurrence ignored) from day one.
3. TTT twist they can't ask: in Huginn only s carries across iterations; for us the fast-weight W
   is a second, slower cross-loop state. Their path-independence framing (fixed point in s) extends
   to "fixed point in (s, W)" — our stability analysis section.
