"""Minimal looped-TTT causal LM — task-agnostic transfer test of the NVS method.

Reuses the exact NVS components (model.Block with per-loop film/gates/affine
conditioning, lact_ttt_loop.LoopFastWeightGluMLPMultihead fast-weight memory).
LM adaptations only: token+pos embeddings, causal attention, chunk-causal TTT
op order (apply chunk c with the memory of chunks < c, then update on c),
tied LM head.
"""
import math
import torch
import torch.nn as nn

from model import Block
from lact_ttt import TTTOperator


class LoopLM(nn.Module):
    def __init__(self, vocab_size, dim, layers, block_config, n_loops=1,
                 seq_len=1024, ttt_chunk=256, loop_film=False, loop_gates=False,
                 loop_affine=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([
            Block(dim=dim, bias=False, block_config=block_config,
                  n_loops_max=n_loops, loop_film=loop_film,
                  loop_gates=loop_gates, loop_affine=loop_affine)
            for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(dim, bias=False)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.n_loops = n_loops
        self.ttt_chunk = ttt_chunk

        self.apply(self._init_weights)
        # residual-branch init scaled by effective depth (same recipe as NVS)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(len(block_config) * layers * n_loops))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, l = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :l]
        # chunk-causal TTT: apply chunk c using memory of chunks < c, then write c.
        ops = []
        for s in range(0, l, self.ttt_chunk):
            e = min(s + self.ttt_chunk, l)
            ops.append(TTTOperator(start=s, end=e, update=False, apply=True))
            ops.append(TTTOperator(start=s, end=e, update=True, apply=False))
        info_base = {"ttt_op_order": ops, "num_img_tokens": l}
        for loop_idx in range(self.n_loops):
            for block in self.blocks:
                x, _ = block(x, {**info_base, "loop_idx": loop_idx})
        return self.head(self.ln_f(x))
