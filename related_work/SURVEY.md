# Related-Work Survey: Looping × TTT Layers (for LaCT/LVSM NVS)

*Compiled 2026-07-18. All arXiv IDs verified via the arXiv API. Novelty check included a keyword sweep
over all Semantic Scholar-indexed citations of LaCT (104 papers) and of the original TTT paper (310 papers).*

---

## (A) Novelty Verdict

**Verdict: No existing work loops (weight-tied depth recurrence) a nonlinear TTT layer (TTT-MLP / LaCT-style
SwiGLU fast weights), and no existing work applies looping of ANY kind to TTT in vision / novel view
synthesis. The specific interaction we target — what happens to the fast-weight state across loop
iterations — is essentially unstudied.**

However, the *linear* end of the fast-weight family has just been looped (all in 2026, all language/time-series,
none in vision/3D, none with nonlinear/large-chunk TTT):

### Closest prior work table

| Paper | arXiv / venue | What is looped | Fast-weight state across loops | Domain | Distance from us |
|---|---|---|---|---|---|
| **LT2: Linear-Time Looped Transformers** | 2605.20670 (preprint, May 2026; CMU/Rice) | Gated DeltaNet, DeltaNet, Mamba2, RetNet, KDA, sparse attn, hybrids | **Carried across loops without reset**, plus learned per-channel residual gate ρ blending the previous iteration's hidden state | LM (FineWeb-Edu 0.6B/1.3B) + synthetic recall/state-tracking | **Closest.** Loops linear fast-weight mixers (delta rule = 1 linear TTT step). No nonlinear/MLP fast weights, no large-chunk TTT, no vision/3D. |
| **Looped SSMs** | 2605.16048 (preprint, May 2026; Farsang et al.) | Linear SSM blocks: LRU, S5, LinOSS, LrcSSM | Sequence recurrence and depth recurrence treated as orthogonal axes | Time-series classification (6 benchmarks) | Linear, non-input-dependent state; small scale; the paper the user flagged (still preprint, low-tier) |
| **Looped State-Space LMs w/ Adaptive Exit-State Selection** | 2607.10110 (preprint, Jul 2026; U. Tokyo) | Shared Mamba / hybrid Mamba-Transformer block | Not the focus; adds Ouro-style exit gates | LM (Mano modular arithmetic, p-hop, pretraining) | Looped Mamba works vs param-matched baselines, but deeper non-looped keeps a PPL edge |
| **Tiny Recursive Reasoning w/ Mamba-2 Attention Hybrid** | 2602.12078 (preprint + OpenReview page, venue unconfirmed; Intercom) | TRM recursion scaffold with Mamba-2 hybrid operator (6.83M params) | Recursion is over the TRM latent (z, y), not over the SSM state | ARC-AGI-1 (45.88% pass@2, +2.0 over TRM) | Validates SSM ops inside a recursion scaffold; tiny scale, puzzle domain |
| **Closed-Loop / Equilibrium Transformers** | 2511.21882 (preprint, Nov 2025) | DEQ-style iterative refinement to equilibrium of a learned energy | Claims to "unify DEQ, diffusion LMs, and TTT as special cases" (conceptual only) | Binary parity toy task | Conceptual overlap only; no TTT layer implementation, toy scale |
| **Déjà View** | 2605.30215 (preprint, May 2026; NVIDIA) | Full softmax-attention ViT block (frame + global attn) | N/A — no fast weights | Multi-view 3D reconstruction | Our vision-side twin: looping in 3D, but quadratic attention, no TTT |
| **RAPTOR / Block-Recurrent Dynamics in ViTs** | 2512.19941, **ICLR 2026** (Kempner) | 2–4 tied ViT blocks distilled from pretrained DINOv2 | N/A | ImageNet/ADE20k/NYUv2 | Evidence ViT depth ≈ few recurrent phases; distillation, not training-from-scratch, no TTT |

### Why we can be confident

- Direct searches on many formulations ("looped test-time training", "recursive TTT", "recurrent depth linear
  attention", "weight-tied fast weight", "DEQ state space", "depth recurrence Mamba", etc.) surface nothing
  combining looping with **nonlinear** TTT layers.
- Keyword scan (`loop|recurs|weight-tied|weight-shar|universal transformer|deep equilibrium`) over **all 104
  papers citing LaCT**: zero hits. Over **all 310 papers citing TTT (2407.04620)**: only 2511.21882 (toy DEQ,
  above) and Comba (2506.02475 — "closed-loop" in the *control-theory* sense inside a bilinear RNN update, not
  depth recurrence; unrelated).
- The community-maintained [Awesome-Loop-Models](https://github.com/huskydoge/Awesome-Loop-Models) list
  (~116 papers) contains no TTT/Titans/LaCT/DeltaNet + looping entry beyond LT2/Looped-SSM items above, and
  **no vision/3D looping applications at all** (Déjà View/RAPTOR not yet indexed there).

### The one caveat to manage in positioning

LT2 will be read as "looping linear attention already done." Our defense is real: a delta-rule linear mixer is a
*single* linear-regression gradient step with a matrix state; LaCT is a *nonlinear* SwiGLU-MLP fast network
updated with Muon on large chunks. LT2 itself reports the interesting phenomenon we can build on — looping is
*unstable* unless the mixer both **forgets (gating)** and **bounds its updates (delta rule)**; RetNet (neither)
diverges, Mamba2/DeltaNet (one each) are noisy, GDN (both) is smoothest. Whether LaCT's Muon +
normalized fast weights play the same stabilizing role under looping is an open question nobody has touched —
and neither has anyone touched the interpretation *loop-over-same-chunk = extra inner-loop optimization steps*.

---

## (B) Key Recipes from the Loop-Transformer Papers (concrete)

### B1. Reasoning with Latent Thoughts (arXiv:2502.17416, **ICLR 2025**, Saunshi et al., Google)

- Notation: **(k, L) = k-layer block looped L times**, ~1B-scale LMs.
- Findings: (k, L) nearly matches a kL-layer non-looped model on reasoning (addition, p-hop induction, GSM-style
  math) and clearly beats the plain k-layer model; on *memorization*-heavy metrics looping does NOT substitute
  for parameters — the loop benefit is specific to reasoning.
- Scaling behaves as a function of **effective depth** (k·L), mirroring inference-time CoT scaling; theory: L loops
  can simulate L steps of latent CoT.
- Also derives a looping-inspired *regularization* for non-looped models (cross-layer weight cosine similarity).

### B2. Déjà View (arXiv:2605.30215, NVIDIA, preprint May 2026) — our closest vision recipe

- **Backbone**: DINOv2 ViT-B encoder (P=14, d=768) → per-view tokens z₀; **one shared looped block** =
  frame-attention sub-block (per-view, 2D RoPE) + global-attention sub-block (all views), pre-norm, LayerScale.
  Separate shallow ray/depth decoder heads (untied).
- **Loop count**: training samples **K ~ Beta(2,1) scaled into [8, 16] per batch**; inference default K=16; K is an
  inference-time compute knob within the trained range (single checkpoint).
- **Tied vs untied**: everything in the block is tied. Untied per-step components are cheap: **three channel-wise
  scale vectors (s_attn, s_mlp, s_out) generated by an MLP from sinusoidal embeddings of the step interval
  (t_k, t_{k+1})** ("time-conditioned gates"), plus a state gate. No re-injection of the original features.
- **Supervision**: only at the final step z_K (no deep supervision). Two-stage: 200K iters end-to-end + 40K depth-head
  finetune; 29-dataset mixture with p_i ∝ √N_i.
- **Headline**: 117M params competitive with VGGT (1257M), Pi3 (959M), MASt3R-SfM (690M) across DTU/ETH3D/
  7-Scenes/ScanNet++/nuScenes.
- **The key ablation for us (Tab. 4)**: tied 16-step block **beats fully-untied 16 independent blocks**: inlier ratio
  66.4 vs 61.1 (69.2 with time-conditioned + state gates). Claim: explicit iteration is a *stronger inductive bias*,
  not just parameter saving — depth in recon transformers is implicit iterative refinement with redundant per-layer
  params. Mechanism observed: state norm grows monotonically while direction cos(z_k, z_K) → 1
  ("directional refinement"; decoder LayerNorm absorbs norm growth).

### B3. RAPTOR / Block-Recurrent Dynamics in ViTs (arXiv:2512.19941, **ICLR 2026**, Kempner Inst.)

- **Block Recurrent Hypothesis**: L-layer pretrained ViT ≈ unrolling of k ≪ L weight-tied blocks.
- **Phase discovery**: weighted **max-cut on the layer-representational-similarity matrix, solved by DP** →
  contiguous depth segments; repetition counts n₁..n_k sum to L. Layer-swap test: within-phase layers are
  interchangeable, cross-phase swaps collapse the model.
- **Distillation recipe**: match all intermediate activations of frozen DINOv2 ViT-B; hybrid loss
  λ·L_teacher-forced + (1−λ)·L_autoregressive + Ω(θ) with **λ annealed to 0** (open-loop → closed-loop);
  stage 2 = pure autoregressive finetune. **Learnable layer-index embeddings** make the tied block
  *non-autonomous* (per-iteration conditioning — same trick class as Déjà View's time gates). Reuses DINOv2 patch
  embed + final LN.
- **Numbers**: k=2 blocks → 81.2% ImageNet linear probe (96% of DINOv2's 84.5%); k=3 → 83.0% (98%); k=4 → 83.2%
  (saturated). Dense tasks keep a gap (ADE20k 43.0 vs 47.5 mIoU; NYUv2 0.618 vs 0.578 RMSE).
- Note: RAPTOR is *post-hoc compression/analysis* of a pretrained model, not from-scratch looped training — as a
  baseline for us it argues "LVSM-style stacks are loopable," but Déjà View is the closer methodological baseline.

### B4. Huginn (arXiv:2502.05171, NeurIPS 2025, Geiping et al.) — the scale-proven loop-training toolkit

- Structure **(prelude 2, recurrent core 4, coda 2)** layers; 3.5B params, 800B tokens; unrolls to effective depth
  132 at r=32.
- **Input injection**: every iteration, adapter A: ℝ^{2h}→ℝ^h maps concat(state sᵢ, embedded input e); concat > add
  at scale. **State init**: random truncated normal, σ² = 2/5.
- **Loop-count sampling**: log-normal Poisson, mean r̄=32 (τ ~ N(log r̄ − σ²/2, σ), σ=1/2; r ~ Poisson(e^τ)+1).
- **Truncated BPTT through last k=8 iterations only** (activation memory independent of sampled depth).
- **Sandwich RMSNorm (4 norms/layer) was *required*** to prevent representation collapse in the recurrence.

### B5. LT2 (arXiv:2605.20670) — the loop × fast-weight state findings

- Loops with **fixed T=4** (they explicitly reject ACT/halting: optimization instability + ragged-batch throughput
  loss, App. A.4). Mixer state **Sₜ carried across loop iterations without reset**; learned per-channel residual
  gate: h⁽ᵘ⁾ = h̃⁽ᵘ⁾ + ρᵤ ⊙ h⁽ᵘ⁻¹⁾.
- **Stability ranking under looping**: GDN (gate + delta rule) smoothest, below even the looped softmax
  Transformer in grad-norm; DeltaNet / Mamba2 (one ingredient each) noisy-but-stable; RetNet (neither) diverges.
- Hybrid ratio 1 full-attention : 4 linear per loop body is optimal. Results (1.3B, 100B tokens): LT2-Hybrid
  Full+GDN 62.89% avg downstream vs looped Transformer 59.27%; GDN+DSA ≈ full-attention quality at ~5.7×
  decode throughput; looped GDN+window doubles the state-tracking length plateau of the looped Transformer.

### B6. Quick reference: other loop recipes

- **Ouro / LoopLM** (2510.25741): 1.4B/2.6B, 7.7T tokens, up to 4 loop steps, **entropy-regularized learned
  depth/exit allocation**; matches ~12B SOTA LLMs (2–3× param efficiency). (Its exit gate is what 2607.10110
  ports to Mamba.)
- **Mixture-of-Recursions** (2507.10524, NeurIPS 2025): per-token routers choose recursion depth over a shared
  stack + recursion-wise KV caching; ~2× inference throughput at equal accuracy.
- **TRM** (2510.04871): 7M-param 2-layer net, deep recursion on latent (z, y), ARC-AGI-1 ~45%; HRM (2506.21734)
  is its two-timescale predecessor.
- **LoopViT** (2602.02156): 18M weight-tied conv+attention hybrid block, parameter-free entropy-based exit
  ("crystallization"), 65.8% visual ARC-AGI-1.

---

## (C) Annotated Bibliography (everything else relevant)

### C1. TTT / fast-weight line (our substrate)

- **TTT** — Learning to (Learn at Test Time): RNNs with Expressive Hidden States, arXiv:2407.04620. TTT-Linear /
  TTT-MLP; hidden state = fast network updated by self-supervised reconstruction; 1 inner GD step per (mini-)batch
  of tokens.
- **LaCT** — Test-Time Training Done Right, arXiv:2505.23884, **ICLR 2026**. Our baseline. Large-chunk updates
  (2K–1M tokens), SwiGLU-MLP fast weights, Muon-style normalized updates, window attention for locality. NVS:
  matches full attention (37.9 dB PSNR @48 views) with prefill 16s → 1.4s. Code: github.com/a1600012888/LaCT.
- **Titans** — Learning to Memorize at Test Time, arXiv:2501.00663 (Google). Deep neural memory updated with
  momentum + weight decay ("surprise" metric); persistent + contextual memory branches.
- **Atlas** (Behrouz et al., 2025) — "learning to optimally memorize context at test time"; Muon-optimized deep
  memory; successor to Titans. (Same group: "It's All Connected" / Miras framework, 2025.)
- **MesaNet** — arXiv:2506.05233. Locally *optimal* test-time training: solves the inner ridge-regression to
  optimality per step (conjugate gradient) instead of one GD step — the "many inner steps" extreme of the axis we
  want to exploit via looping.
- **TNT** — Improving Chunkwise Training for Test-Time Memorization, arXiv:2511.07343. Hierarchical
  global/local chunking to reconcile large-chunk parallelism with update freshness; directly about the
  chunk-size/quality tension LaCT created.
- **TTT-E2E** — End-to-End Test-Time Training for Long Context, arXiv:2512.23675 (Stanford/TTT team; the
  "TTT-E" the user mentioned). Meta-learns the inner loop against the true next-token loss; 3B/164B tokens scales
  with context like full attention, 2.7× faster @128K. Confirms TTT quality hinges on inner-loop optimization
  quality — support for our loop-as-extra-inner-steps view.
- **Test-time regression** — arXiv:2501.12352. Unifying theory: linear attention, SSMs, FWPs, online learners,
  softmax attention = associative-memory regression with 3 design choices. Useful theory scaffolding.
- **TTT with KV Binding Is Secretly Linear Attention** — arXiv:2602.21204. Recent theory tying practical TTT
  variants back to linear attention; useful for relating our method to LT2's linear mixers.
- **Gated DeltaNet** — arXiv:2412.06464, ICLR 2025. Gate + delta rule; the mixer LT2 found most loopable.
- **Going Beyond Linear Transformers with Recurrent FWPs** — arXiv:2106.06295 (NeurIPS'21). Classic: adds
  recurrence to fast-weight programmers (sequence-dim recurrence, not depth loops — good for a "not the same
  thing" citation).
- **One-Minute Video Generation with TTT** — arXiv:2504.05298. TTT-MLP layers in video diffusion.

### C2. TTT in vision / 3D (our task's competitive landscape)

- **ViT³** — Unlocking Test-Time Training in Vision, arXiv:2512.01643, **CVPR 2026 oral** (Tsinghua). Systematic
  TTT design study for visual sequence modeling: classification, generation, detection, segmentation; matches or
  beats Mamba/linear-attention baselines. No looping.
- **tttLRM** — arXiv:2602.20160, **CVPR 2026** (Adobe: Hao Tan, Sai Bi et al. — the LVSM lineage). TTT layer
  compresses posed views into fast weights = implicit 3D representation; NVS pretraining → feed-forward 3DGS
  reconstruction, autoregressive/streaming. The closest *task* competitor; no looping.
- **Fast Spatial Memory with Elastic TTT (LaCET)** — arXiv:2604.07350. Direct LaCT follow-up: adds Elastic
  Weight Consolidation regularizer to the LaCT update for long/dynamic 4D scenes (fixes cumulative-plasticity
  ghosting). Shows people are already modifying the LaCT inner update — but not with loops.
- **Spatial-TTT** — arXiv:2603.12255. Streaming visual-spatial intelligence with TTT.
- **ReCal3R** — arXiv:2607.05356. Reliability-calibrated inner learning rates for streaming 3D reconstruction —
  another "inner-update quality" knob in 3D TTT.
- **LVSM** — arXiv:2410.17242 (ICLR 2025 oral) [user-known; listed for completeness as the target task].

### C3. Looped / recurrent-depth transformers (verified user list + additions)

User's list — **all arXiv IDs verified correct**:
- Universal Transformers (ICLR'19); ALBERT (ICLR'20); Looped Transformers as Programmable Computers (ICML'23). ✓
- **2311.12424** Looped Transformers are Better at Learning Learning Algorithms ✓ (ICLR'24) — looped TF matches
  unlooped for *learning* iterative learning algorithms in-context; conceptually cute for us (loop ≈ optimizer
  iterations — same flavor as TTT inner steps).
- **2502.17416** Reasoning with Latent Thoughts ✓ (ICLR'25) — see B1.
- **2412.06769** Coconut = "Training LLMs to Reason in a Continuous Latent Space" ✓ (latent CoT, not weight-tied
  depth; adjacent).
- **2409.15647** Looped Transformers for Length Generalization ✓ (ICLR'25) — input-dependent #loops (n-RASP-L),
  big length-generalization gains.
- **2502.05171** Huginn ✓ (NeurIPS'25) — see B4.
- **2510.25741** Ouro ✓ — see B6. **2510.24824** Parallel Loop Transformer ✓ (full title "...for Efficient
  Test-Time Computation Scaling" — runs loop steps in parallel across the batch/pipeline to hide loop latency).
- **2506.21734** HRM ✓; **2510.04871** TRM ("Less is More: Recursive Reasoning with Tiny Networks") ✓.
- **RAPTOR** = "Block-Recurrent Dynamics in Vision Transformers", arXiv:2512.19941, ICLR 2026 ✓ — see B3.
- **2605.30215** Déjà View ✓ — see B2.

Additional 2025–2026 loop papers worth knowing:
- **bViT** — arXiv:2605.10661: single-block recurrence in ViTs for ImageNet; the from-scratch small-scale vision
  loop study.
- **Loop, Think & Generalize** — arXiv:2604.07822: implicit reasoning dynamics in recurrent-depth TFs.
- **Retrofitted Recurrence** — arXiv:2511.07384: continued-pretraining converts pretrained LLM blocks to looped
  mode.
- **CoLa / Skip-a-Layer-or-Loop-It** — arXiv:2507.07996: test-time per-input layer skipping/looping of pretrained
  LLMs (note: "test-time" here = depth adaptation, NOT TTT layers).
- **Encode-Think-Decode** — arXiv:2510.07358: recursion on a reasoning-relevant subset of layers.
- **Two-Scale Latent Dynamics** — arXiv:2509.23314; **Training-Free Looped Transformers** — arXiv:2605.23872;
  **DeepLoop (depth scaling for looped TFs)** — arXiv:2607.13491; **Looped World Models** — arXiv:2606.18208;
  **LARM (depth-conditioned looped TF for ASR)** — arXiv:2606.04678; **LoopCoder-v2** — arXiv:2606.18023.
- Curated index: github.com/huskydoge/Awesome-Loop-Models (~116 papers, 3 categories).

---

## (D) Implications & Opportunities for Our Project

1. **The novel axis nobody owns: fast-weight state policy across loop iterations.** With a looped LaCT layer over
   the same chunk, three regimes exist: (a) **reset** state each loop → loop only deepens feature refinement;
   (b) **carry** state → each loop applies *additional inner-loop gradient steps on the same chunk* with
   re-mixed features — i.e., looping buys better inner optimization (the MesaNet/TTT-E2E axis) for free with tied
   slow weights; (c) **gated blend** (LT2's ρ-gate). LT2 only tried (b)+gate for linear mixers in LMs. A clean
   study + a method exploiting (b) — e.g., loop-dependent inner LR / Muon schedule, or re-using the loop as a
   *second epoch* over the chunk — is a publishable core.
2. **Stability story is pre-written and testable.** LT2: looping needs forgetting + bounded updates. LaCT already
   has Muon-normalized updates and (optionally) weight decay/normalized fast weights — we can predict LaCT loops
   stably where naive TTT-SGD does not, and verify with grad-norm curves à la LT2 Fig. 6. This gives the paper its
   "why TTT × looping interacts specially" analysis section, with math intuition (delta rule = projection ⇒
   non-expansive; Muon ⇒ bounded spectral-norm updates).
3. **Recipe to start from (synthesis of B):** loop the LaCT block (window-attn + TTT + MLP) K∈[4,16] times with
   K ~ Beta(2,1)-style sampling (Déjà View) or fixed T=4 first (LT2's stability-first choice); per-loop
   *cheap untied* conditioning = 3 channel-wise scale vectors from step embeddings (Déjà View) or layer-index
   embeddings (RAPTOR); sandwich-norm the looped block (Huginn — collapse prevention); truncated BPTT (k≈8) if
   memory binds; supervision at final step only; expose K as an inference-time quality knob for NVS (render better
   with more loops — a demo full attention cannot do at constant param count).
4. **Headline target mirrors Déjà View:** show tied-K LaCT ≥ untied-L LaCT (not merely ≈) at ~1/L params — their
   IR 66.4-vs-61.1 result says this is achievable in 3D vision; if the TTT-state-carry mechanism (b) is why, that
   is exactly our novelty.
5. **Baselines/citations to pre-empt reviewers:** LT2 (closest prior, LM-only, linear mixers), Looped SSMs +
   Looped Mamba LMs (preprints, linear state), Déjà View + RAPTOR (vision loops, quadratic attention),
   tttLRM/ViT³/LaCET (TTT-in-vision concurrent work, no loops), MoR/Ouro (adaptive depth — future extension:
   per-view or per-chunk loop counts).
6. **Risks:** (i) 2607.10110 found deeper *non-looped* models keep a perplexity edge under strict compute
   matching — our claim should emphasize params + the K-knob + any *quality gain*, not FLOPs parity;
   (ii) the field is moving fast (5+ loop papers in May–Jul 2026 alone) — someone could loop TTT in LMs any month;
   the vision/NVS + large-chunk angle and the state-policy analysis are our moat, so move fast on the state-carry
   experiments.
