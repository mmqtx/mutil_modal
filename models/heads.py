"""
Diagnostic chain: 6 classification heads with inter-head Self-Attention,
GRU-gated hierarchical transfer.

All 6 diagnostic steps (including ischemia) are handled uniformly through
MultiTaskHead, producing per-subtask logits via shared GRU hidden states.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHAIN_ORDER, HEAD_HIDDEN_RATIO


# ---------------------------------------------------------------------------
# Inter-head Self-Attention
# ---------------------------------------------------------------------------

class ChainSelfAttention(nn.Module):
    """6 diagnostic steps attend to each other to model dependencies."""

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer["ln1"](embeds)
            h, _ = layer["attn"](h, h, h)
            embeds = embeds + h
            h = layer["ffn"](layer["ln2"](embeds))
            embeds = embeds + h
        return embeds


# ---------------------------------------------------------------------------
# Multi-output classification head
# ---------------------------------------------------------------------------

class MultiTaskHead(nn.Module):
    """Shared embedding → multiple sub-task logits."""

    def __init__(self, in_dim: int, subtasks: List[Tuple[str, int, str]], dropout: float = 0.1):
        super().__init__()
        self.subtask_names = [s[0] for s in subtasks]
        # 使用配置的hidden ratio，增加分类头capacity
        hidden_dim = int(in_dim * HEAD_HIDDEN_RATIO)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_cls),
            )
            for name, n_cls, _ in subtasks
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: self.projections[name](x) for name in self.subtask_names}


# ---------------------------------------------------------------------------
# Gated Reasoning Chain
# ---------------------------------------------------------------------------

class DiagnosticChain(nn.Module):
    """
    6-step clinical reasoning chain with:
      1. Per-step projection
      2. Inter-head Self-Attention
      3. GRU-gated hierarchical transfer
      4. Multi-task classification heads

    All 6 steps (including ischemia_infarct) are handled uniformly.
    """

    def __init__(
        self,
        fused_dim: int = 512,
        head_subtasks: Dict[str, List[Tuple[str, int, str]]] = None,
        chain_order: List[str] = None,
        num_chain_attn_heads: int = 4,
        num_chain_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chain_order = chain_order or CHAIN_ORDER
        self.num_steps = len(self.chain_order)
        self.head_subtasks = head_subtasks or {}

        # Per-step projections
        self.step_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(self.num_steps)
        ])

        # Inter-head self-attention
        self.chain_attn = ChainSelfAttention(
            dim=fused_dim, num_heads=num_chain_attn_heads,
            num_layers=num_chain_attn_layers, dropout=dropout,
        )

        # GRU for gated hierarchical transfer
        self.gru = nn.GRUCell(fused_dim, fused_dim)

        # Multi-task classification heads (all 6 steps, including ischemia)
        self.class_heads = nn.ModuleDict()
        for name in self.chain_order:
            subs = self.head_subtasks.get(name, [])
            self.class_heads[name] = MultiTaskHead(fused_dim, subs, dropout)

        self._init_weights()

    def _init_weights(self):
        for proj in self.step_projections:
            nn.init.xavier_uniform_(proj[0].weight)

    def forward(self, fused_feat: torch.Tensor, return_per_head_features: bool = False) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            fused_feat: (B, D)
            return_per_head_features: 是否返回每个头的独立特征（用于对比学习）
        Returns:
            logits: flat dict of "head.subtask" → logits
            head_embeds: (B, num_steps, D) 如果 return_per_head_features=True，则为 {head_name: (B, D)}
        """
        B, D = fused_feat.size(0), fused_feat.size(1)

        # Per-step projections → (B, num_steps, D)
        embeds = torch.stack([proj(fused_feat) for proj in self.step_projections], dim=1)

        # Inter-head self-attention
        embeds = self.chain_attn(embeds)

        # Gated hierarchical transfer + classification
        logits = {}
        h = torch.zeros(B, D, device=fused_feat.device)

        # 存储每个头的特征（用于对比学习）
        per_head_features = {}

        for i, name in enumerate(self.chain_order):
            hi = embeds[:, i]                          # (B, D)
            h = self.gru(hi, h)                         # nonlinear gated accumulation
            sub_logits = self.class_heads[name](h)      # dict of subtask → logits
            for sub_name, sub_logits_tensor in sub_logits.items():
                logits[f"{name}.{sub_name}"] = sub_logits_tensor

            # 存储该头的特征（经过 GRU 后的 h）
            if return_per_head_features:
                per_head_features[name] = h.clone()

        if return_per_head_features:
            return logits, per_head_features
        return logits, embeds

        return logits, embeds
