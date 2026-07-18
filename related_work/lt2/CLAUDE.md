# LT2 — Linear-Time Looped Transformers (arXiv:2605.20670, Rice/Apple/UCSC/CMU)

**Closest prior work.** Loops subquadratic mixers (the linear end of the fast-weight family) in LMs.
Folder: paper PDF + official code `LT2/` (github.com/chili-lab/LT2; model in
`LT2/apps/LT2/transformer.py`, built on lingua + flash-linear-attention).

## What the paper does
Replaces softmax attention inside a looped Transformer with linear attention (RetNet, Mamba2, GLA,
HGRN2, DeltaNet, GDN, KDA), sparse attention (Window, NSA, DSA), and hybrids. Claims: looping turns
compute into context — (a) DPLR linear attention: T loops upgrade the rank-1 state transition to an
effective rank-T memory-erasure operator, `A_t^eff = prod_tau (Diag(alpha_t^tau) (I - beta_t^tau k_t^tau k_t^tau^T))`,
provided the per-loop keys are non-collinear; (b) sliding window w: receptive field grows to O(T*w).

## Loop mechanics + the residual-gate recipe (verified in code)
- **Fixed T=4 loops** for all main results. ACT/halting explicitly rejected (App. A): optimization
  instability, pondering-weight sensitivity, ragged-batch throughput loss.
- **Learned per-loop residual gate**: `h^(tau) = h~^(tau) + rho_tau ⊙ h^(tau-1)` where h~ is the
  looped stack's output. In code: `residual_weight = nn.Parameter(zeros(loop_count, dim))` —
  **zero-initialized, per-iteration AND per-channel**; applied as
  `h = h + residual_weight[loop_idx] * h_input` (transformer.py ~L2488, L2689). Two residual
  levels total: per-block identity + per-loop learned gate.
- **IMPORTANT — no explicit state carry.** Each loop iteration re-runs the linear-attention scan
  from scratch on the *updated* hidden states (`layer(h)` per iteration); the matrix state S is
  recomputed within each loop, and cross-loop information flows only through h (+ rho gate). The
  rank-T theory is about the composition of per-loop transitions, not a literally persisted S.
  Their Limitations say it outright: "we do not design explicit cross-loop recurrent state carry
  mechanisms; principled state-sharing across loops may further improve long-context modeling,
  memory reuse, and compute efficiency." **That unexplored axis is exactly our project** (and our
  W is a nonlinear SwiGLU MLP updated by Muon, not a linear matrix state).
- Supervision: final-loop CE only (no deep supervision; matches CoTFormer reference signal).

## Stability findings per mixer (Sec. 3.5, Fig. 6-8) — the pre-written story for our analysis
- **Looped RetNet** (no data-dependent gate, no delta rule): persistently large grad norms, spikes,
  **diverges** at 0.6B.
- **Looped DeltaNet** (delta rule, weak gating) and **Looped Mamba2** (gating, no delta rule):
  stable but noisy — occasional grad spikes propagating into loss.
- **Looped GDN** (gate + delta rule): smoothest loss, grad norm **below the looped softmax
  Transformer** for essentially the whole run.
- Sparse loops (Window/NSA/DSA): never destabilize, slightly higher final loss.
- Hybrids (Full+GDN, GDN+DSA): match/beat looped Transformer loss with smaller, smoother grads.
- Takeaway: looping needs **forgetting (gate) + bounded updates (delta rule)**. LaCT has both
  analogs (per-channel forget gate/weight decay + Muon-normalized updates) -> predict stable
  looping; naive TTT-SGD lacks them -> predict instability. Reproduce their grad-norm figure.
- Attention-sink pathology compounds across loops (sawtooth first-token mass, residual RMS ~20x by
  loop 4); a zero-init head-specific sigmoid **SDPA output gate** inside the loop fixes it
  (PPL 9.87->9.39, avg +1.43 at 1.3B). Our window-attention sub-block may need the same gate.

## Exact configs (App. C + paper)
- Scales: 0.6B (d=1024, 25 layers, 16 heads) and 1.3B (d=2048, 16-25 layers, 16 heads); RoPE
  theta=10000; Llama tiktoken vocab 128,256; seqlen 4096.
- Data: FineWeb-Edu 100BT, 255K steps, ~4e5 tokens/step (~100B tokens, 1 epoch).
- Optim: AdamW (0.9, 0.95), clip 1.0, bf16; peak LR 3e-4 (1.3B) / 1.5e-4 (0.6B, lowered for looped
  sliding-window stability); linear warmup 5-10K steps + cosine to 1e-6*peak; wd 0.1 (0.033 for
  one unstable window baseline). FSDP full_shard + torch.compile (off for DSA). Seeds 777/42.
- Hybrid: **1 full-attn : 4 linear per loop body is optimal** (inverse-U over ratio); pattern:
  spreading full-attn along depth beats concentrating (bookend slightly best); loop-level mixer
  schedules (coarse->fine etc.) do NOT beat depth-level interleave.
- Window sizes: w=256 (0.6B), w=2048 (1.3B).

## Key numbers
- 1.3B/100B: Looped Transformer PPL 9.87 / 59.27 avg; Looped GDN 9.75 / 59.92 (beats it);
  Hybrid Full+GDN 9.12 / **62.89 avg (+2.1 pts over looped TF, ~5x decode speedup)**;
  Hybrid GDN+DSA 9.50 / 60.73 = full-attn-loop quality, ~5.7x decode throughput (125 vs 22 tok/s
  @8k, bs=8), no quadratic component.
- Synthetic recall+state-tracking (swap curriculum): looping helps subquadratic mixers more; Looped
  GDN+Window jumps stage 3 (T<=4) -> stage 5 (T=8), doubling n_max over the looped Transformer
  plateau (64 -> 128). Looped models gain 2-4 pts on realistic recall suites at matched params.
- Also: distills Ouro-1.4B into Ouro-Hybrid-1.4B (RADLADS-style, per-loop KL warmup then
  final-loop-only) retaining ~93% avg teacher capability with ~1B tokens.

## What we borrow / compare against
1. Borrow verbatim: zero-init per-loop-per-channel rho gate; fixed T=4 first (stability-first);
   final-step-only supervision; grad-norm instrumentation; SDPA output gate for our window attn.
2. Our delta over LT2 (for the paper's positioning): nonlinear MLP fast weights + Muon large-chunk
   updates vs their linear matrix states; **explicit fast-weight state policy across loops
   (reset/carry/gated)** — the mechanism they name as future work; vision/NVS domain.
3. Their theory hook: delta rule = projection => non-expansive transitions. Our analog: Muon =>
   bounded spectral-norm updates. Write the parallel argument for LaCT.
