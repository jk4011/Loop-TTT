# Reasoning with Latent Thoughts / Looped Transformers (arXiv:2502.17416, ICLR 2025, Saunshi et al., Google)

Folder: paper PDF only. No official code released (Google internal; experiments are standard
Pythia-style LMs + synthetic tasks).

## What the paper does
The clean empirical + theoretical case that **looping buys reasoning depth, not knowledge**.
Notation: `(k ⊗ L)` = k-layer block looped L times; iso-param baseline `(k ⊗ 1)`; iso-FLOP
baseline `(kL ⊗ 1)`.

## Key findings
- On reasoning-heavy synthetic tasks (n-ary addition, p-hop induction, GSM-style math via i-GSM):
  `(k ⊗ L)` **nearly matches `(kL ⊗ 1)`** and crushes `(k ⊗ 1)` — loops substitute for depth.
- Language modeling at ~1B scale: looped models have visibly worse perplexity than iso-FLOP
  (perplexity tracks unique params) yet **downstream reasoning accuracy approaches / sometimes
  exceeds iso-FLOP**; memorization-heavy metrics (TriviaQA-style closed-book) do NOT improve —
  the loop benefit is specific to reasoning-like computation.
- Scaling behaves as a function of **effective depth k·L**: more loops at inference = latent CoT
  scaling; theory shows L loops can simulate L steps of CoT reasoning.
- Bonus: a looping-inspired regularizer (push cross-layer weight cosine similarity up) transfers
  some of the benefit to non-looped models.

## What we borrow / compare against
1. **Experimental grammar**: adopt the (k ⊗ L) notation and always report the two baselines —
   iso-param (k ⊗ 1) and iso-FLOP (kL ⊗ 1). Our headline claim mirrors theirs but in NVS:
   looped-LaCT (k ⊗ L) vs deep LaCT (kL ⊗ 1) at 1/L params.
2. **Expectation setting**: their perplexity-vs-downstream split predicts our looped model may not
   win raw pixel loss against iso-FLOP but can win on the "reasoning-like" part of NVS —
   multi-view consistency / occlusion & geometry inference. Worth building metrics that separate
   "memorized appearance" from "geometric reasoning" (their memorization-vs-reasoning dichotomy).
3. **Latent-CoT interpretation**: for TTT, each loop that re-optimizes the fast weights on the same
   chunk is a *concrete, interpretable* latent thought = one more inner optimization step. Their
   CoT-simulation theory is a citable frame for "loop count = test-time compute knob".
4. Distance from us: full softmax attention, LM domain, no fast-weight state — nothing about what
   loops do to a stateful layer. That interaction is exactly our contribution.
