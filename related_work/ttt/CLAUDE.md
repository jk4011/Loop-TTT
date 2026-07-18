# TTT — Learning to (Learn at Test Time): RNNs with Expressive Hidden States (arXiv:2407.04620)

Folder: paper PDF + LaTeX (`latex/`) + official code `ttt-lm-pytorch/` (github.com/test-time-training/
ttt-lm-pytorch; single-file reference implementation in `ttt.py`, HuggingFace-style).

## What the paper does
Introduces TTT layers: a sequence-modeling layer whose **hidden state is itself a model** (fast
network) f_W, updated at test time by gradient descent on a self-supervised loss over the incoming
tokens. Two instantiations: **TTT-Linear** (W is a linear map) and **TTT-MLP** (2-layer MLP).
Framing: "inner loop" = fast-weight SGD over the sequence; "outer loop" = normal training of
everything else (incl. the inner objective's projections and per-token inner LR).

## Key method (the equations our loop multiplies)
- Inner objective: reconstruction with learned views, `l(W; x_t) = || f_W(theta_K x_t) - theta_V x_t ||^2`;
  output uses a third view `z_t = f_W(theta_Q x_t)`.
- Update: `W_t = W_{t-1} - eta * grad l(W_{t-1}; x_t)` — **one SGD step per token** (online GD),
  made hardware-feasible via **mini-batch TTT** (batch size b=16 tokens: parallel gradients at the
  same base point) + a dual form that avoids materializing per-token W.
- Stabilizers they needed for TTT-MLP: learnable inner LR (sigmoid-gated, per token), LayerNorm +
  residual inside f_W. TTT-MLP is less stable than TTT-Linear in long context — relevant warning
  for us since LaCT's fast net is also an MLP.
- Theorem: TTT-Linear with batch GD == linear attention; softmax attention == nonparametric TTT.
  (Useful when relating our looped nonlinear TTT to LT2's looped linear mixers.)

## Key numbers
- 125M–1.3B params, Pile/Books: TTT-Linear and TTT-MLP match or beat Transformer and Mamba in
  perplexity at 2k/8k context; TTT layers keep improving past 16k where Mamba plateaus.
- Mini-batch size sweep: quality improves monotonically as inner batch shrinks (more sequential
  update steps) — **more/finer inner optimization = better**, the axis our loop exploits from the
  other direction (loop over the same chunk = extra inner steps at fixed chunk size).

## What we borrow / compare against
1. The inner/outer-loop vocabulary and the "hidden state = learner" view: a depth loop that carries
   W is literally **multiple inner-loop epochs over the same chunk** — cite their mini-batch-size
   ablation as evidence that more inner optimization helps quality.
2. `ttt.py` shows the clean dual-form implementation and the learnable inner-LR gating — the
   knob we may want to make **loop-dependent** (eta as a function of loop index k).
3. Naive TTT (SGD, no update normalization) is the control in our stability study: prediction is
   it loops poorly vs LaCT's Muon-normalized update (LT2's bounded-update argument).
4. Compare-against: looped TTT-Linear is close to LT2 territory; our claim must center on the
   nonlinear MLP fast weights + large-chunk + vision/NVS setting.
