"""
Multi-layer cross-attention fusion between signal and image features.

signal_feat  (B, D) ──┐
                      ├─→ N × Cross-Attention layers ─→ concat ─→ MLP ─→ fused (B, D)
image_feat   (B, D) ──┘

Each layer performs bidirectional cross-attention with residual + LN + FFN,
allowing iterative refinement between modalities.
"""

import torch
from torch import nn


class _CrossAttnLayer(nn.Module):
    """One layer of bidirectional cross-attention with FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        # Signal attends to image
        self.sig_to_img_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.sig_norm1 = nn.LayerNorm(dim)
        self.sig_ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.sig_norm2 = nn.LayerNorm(dim)

        # Image attends to signal
        self.img_to_sig_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.img_norm1 = nn.LayerNorm(dim)
        self.img_ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.img_norm2 = nn.LayerNorm(dim)

    def forward(self, sig: torch.Tensor, img: torch.Tensor):
        """
        Args:
            sig: (B, 1, D)
            img: (B, 1, D)
        Returns:
            sig, img: refined (B, 1, D)
        """
        # Signal cross-attends to image
        sig_ctx, _ = self.sig_to_img_attn(sig, img, img)
        sig = self.sig_norm1(sig + sig_ctx)
        sig = self.sig_norm2(sig + self.sig_ffn(sig))

        # Image cross-attends to signal
        img_ctx, _ = self.img_to_sig_attn(img, sig, sig)
        img = self.img_norm1(img + img_ctx)
        img = self.img_norm2(img + self.img_ffn(img))

        return sig, img


class CrossAttentionFusion(nn.Module):
    """Multi-layer bidirectional cross-attention fusion.

    Stacks N cross-attention layers for iterative multi-modal refinement,
    then merges via concatenation + MLP.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _CrossAttnLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final merge: concat(sig, img) → MLP → fused
        mlp_hidden = int(dim * mlp_ratio)
        self.merge = nn.Sequential(
            nn.Linear(dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            for m in [layer.sig_to_img_attn, layer.img_to_sig_attn]:
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                    nn.init.zeros_(m.out_proj.bias)

    def forward(self, sig_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sig_feat: (B, D) signal CLS feature
            img_feat: (B, D) image CLS feature
        Returns:
            fused: (B, D)
        """
        # Expand to (B, 1, D) for attention
        sig = sig_feat.unsqueeze(1)
        img = img_feat.unsqueeze(1)

        # Iterative cross-attention refinement
        for layer in self.layers:
            sig, img = layer(sig, img)

        # Concatenate and project
        combined = torch.cat([sig.squeeze(1), img.squeeze(1)], dim=-1)  # (B, 2D)
        fused = self.out_norm(self.merge(combined))                      # (B, D)
        return fused
