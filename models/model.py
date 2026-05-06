"""
Top-level ECG Diagnostic Model.

Pipeline:
  signal (B,12,5000) -> EcgTransformer -> sig_feat (B,512) -> proj_head -> sig_proj (B,256) ->|
      |                                                                                             |-- InfoNCE loss
      +---------> signal_uplift (512->512) ----+                                                    |
                                              +---> CrossAttentionFusion (N layers) -> fused (B,512) -> chain -> logits
  image  (B,3,336,336) -> HF CLIP ViT   -> img_feat (B,1024) -> proj_head -> img_proj (B,256) ->|
      |
      +---------> image_uplift (1024->512) ---+

Signal backbone: GEM EcgTransformer, output_dim=512
Image backbone:  HuggingFace CLIPVisionModel (clip-vit-large-patch14-336), hidden_size=1024
Both uplifted to fusion_dim=512, fused via multi-layer cross-attention.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import EcgTransformer
from .fusion import CrossAttentionFusion
from .heads import DiagnosticChain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP Vision Encoder wrapper (aligned with GEM's llava/clip_encoder.py)
# ---------------------------------------------------------------------------

class CLIPVisionEncoder(nn.Module):
    """HuggingFace CLIPVisionModel wrapper — same as GEM's CLIPVisionTower.

    Uses transformers.CLIPVisionModel.from_pretrained() to load
    openai/clip-vit-large-patch14-336, which outputs hidden_size=1024.
    """

    def __init__(self, model_path: str = "openai/clip-vit-large-patch14-336"):
        super().__init__()
        from transformers import CLIPVisionModel

        if os.path.isdir(model_path):
            self.vision_model = CLIPVisionModel.from_pretrained(model_path)
            logger.info(f"Loaded CLIP vision model from local path: {model_path}")
        else:
            self.vision_model = CLIPVisionModel.from_pretrained(model_path)
            logger.info(f"Loaded CLIP vision model from HuggingFace: {model_path}")

        self.hidden_size = self.vision_model.config.hidden_size  # 1024 for ViT-L/14

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 336, 336) — CLIP-normalized
        Returns:
            pooled: (B, hidden_size) — CLS token from last hidden state
        """
        outputs = self.vision_model(images)
        return outputs.last_hidden_state[:, 0]  # (B, hidden_size)

    def lock(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    """Build a configurable MLP.

    Args:
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: intermediate hidden dimension
        num_layers: 1 = single Linear, >=2 = MLP with ReLU activations

    Architecture for num_layers >= 2:
        Linear(in_dim, hidden_dim) → ReLU → [Linear(hidden_dim, hidden_dim) → ReLU] × (n-2) → Linear(hidden_dim, out_dim)
    """
    if num_layers == 1:
        return nn.Linear(in_dim, out_dim)

    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ECGDiagModel(nn.Module):
    def __init__(
        self,
        # ECG signal encoder (GEM-aligned)
        seq_length: int = 5000,
        lead_num: int = 12,
        signal_patch_size: int = 50,
        signal_width: int = 768,
        signal_layers: int = 12,
        signal_heads: int = 12,
        signal_mlp_ratio: float = 4.0,
        embed_dim: int = 512,
        # Image encoder (HuggingFace CLIP)
        clip_model_path: str = "openai/clip-vit-large-patch14-336",
        freeze_signal_encoder: bool = False,
        freeze_image_encoder: bool = False,
        # Fusion
        fusion_dim: int = 512,
        fusion_heads: int = 8,
        fusion_num_layers: int = 2,
        fusion_dropout: float = 0.1,
        # Downstream
        head_subtasks: Optional[Dict[str, List[Tuple[str, int, str]]]] = None,
        chain_attn_heads: int = 8,
        chain_attn_layers: int = 2,
        head_dropout: float = 0.1,
        contrastive_weight: float = 0.1,
        # Uplift projection config
        uplift_hidden_dim: int = 512,
        uplift_num_layers: int = 2,
        # Contrastive projection config
        contrastive_hidden_dim: int = 512,
        contrastive_out_dim: int = 256,
        contrastive_num_layers: int = 2,
        # Head-level contrastive learning (新增)
        use_head_contrastive: bool = False,
        head_contrastive_weight: float = 0.1,
        head_contrastive_temp: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

        # ---- Signal backbone (GEM EcgTransformer) ----
        self.signal_backbone = EcgTransformer(
            seq_length=seq_length,
            lead_num=lead_num,
            patch_size=signal_patch_size,
            width=signal_width,
            layers=signal_layers,
            heads=signal_heads,
            mlp_ratio=signal_mlp_ratio,
            output_dim=embed_dim,
        )
        if freeze_signal_encoder:
            for param in self.signal_backbone.parameters():
                param.requires_grad = False

        # ---- Image backbone (HuggingFace CLIP ViT) ----
        self.image_backbone = CLIPVisionEncoder(model_path=clip_model_path)
        if freeze_image_encoder:
            self.image_backbone.lock()
        image_hidden = self.image_backbone.hidden_size  # 1024 for ViT-L/14

        # ---- Uplift projections: signal/image → fusion_dim ----
        self.signal_uplift = _build_mlp(
            embed_dim, fusion_dim, uplift_hidden_dim, uplift_num_layers,
        )
        self.image_uplift = _build_mlp(
            image_hidden, fusion_dim, uplift_hidden_dim, uplift_num_layers,
        )

        # ---- Multi-layer cross-attention fusion ----
        self.fusion = CrossAttentionFusion(
            dim=fusion_dim, num_heads=fusion_heads,
            num_layers=fusion_num_layers, dropout=fusion_dropout,
        )

        # ---- Diagnostic chain (all 6 steps, including ischemia) ----
        self.chain = DiagnosticChain(
            fused_dim=fusion_dim, head_subtasks=head_subtasks,
            num_chain_attn_heads=chain_attn_heads,
            num_chain_attn_layers=chain_attn_layers,
            dropout=head_dropout,
        )

        # ---- Contrastive projection heads → contrastive_out_dim ----
        self.sig_proj_head = _build_mlp(
            embed_dim, contrastive_out_dim,
            contrastive_hidden_dim, contrastive_num_layers,
        )
        self.img_proj_head = _build_mlp(
            image_hidden, contrastive_out_dim,
            contrastive_hidden_dim, contrastive_num_layers,
        )

        # ---- Contrastive alignment ----
        self.contrastive_weight = contrastive_weight
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

        # ---- Head-level contrastive learning (新增) ----
        self.use_head_contrastive = use_head_contrastive
        self.head_contrastive_weight = head_contrastive_weight
        self.head_contrastive_temp = head_contrastive_temp

    def forward(
        self,
        signal: torch.Tensor,
        image: torch.Tensor,
        return_head_features: bool = False,
        modality_dropout_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            signal: (B, 12, 5000) ECG signal
            image: (B, 3, 336, 336) ECG image
            return_head_features: 是否返回每个头的特征（用于对比学习）

        Returns:
            dict with keys:
                - "logits": flat dict of "head.subtask" → logit tensors
                - "aux_losses": dict with "contrastive" auxiliary loss
                - "log_vars": (12,) tensor for adaptive weighting
                - "head_features": (可选) {head_name: (B, D)} 每个头的特征
        """
        # Backbone features
        sig_feat = self.signal_backbone(signal)    # (B, 512)
        img_feat = self.image_backbone(image)      # (B, 1024)

        # Contrastive alignment
        sig_proj = self.sig_proj_head(sig_feat)    # (B, 256)
        img_proj = self.img_proj_head(img_feat)    # (B, 256)
        contrastive_loss = self._contrastive_loss(sig_proj, img_proj)

        if self.training and modality_dropout_prob > 0:
            sig_feat, img_feat = self._apply_modality_dropout(
                sig_feat, img_feat, modality_dropout_prob
            )

        # Uplift to fusion_dim
        sig_up = self.signal_uplift(sig_feat)  # (B, fusion_dim)
        img_up = self.image_uplift(img_feat)   # (B, fusion_dim)

        # Multi-layer cross-attention fusion
        fused = self.fusion(sig_up, img_up)    # (B, fusion_dim)

        # Diagnostic chain (all 6 steps including ischemia)
        if return_head_features:
            chain_logits, head_features = self.chain(fused, return_per_head_features=True)
            return {
                "logits": chain_logits,
                "aux_losses": {"contrastive": contrastive_loss},
                "head_features": head_features,
            }
        else:
            chain_logits, _ = self.chain(fused)
            return {
                "logits": chain_logits,
                "aux_losses": {"contrastive": contrastive_loss},
            }

    def _contrastive_loss(self, sig_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        B = sig_feat.size(0)
        if B < 2:
            return torch.tensor(0.0, device=sig_feat.device)

        sig_norm = F.normalize(sig_feat, dim=-1)
        img_norm = F.normalize(img_feat, dim=-1)
        sim = sig_norm @ img_norm.t() / self.temperature.clamp(min=0.01)
        labels = torch.arange(B, device=sig_feat.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0
        return loss

    def _apply_modality_dropout(
        self,
        sig_feat: torch.Tensor,
        img_feat: torch.Tensor,
        drop_prob: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop exactly one modality for a subset of samples during training."""
        if drop_prob <= 0:
            return sig_feat, img_feat

        B = sig_feat.size(0)
        device = sig_feat.device
        drop = torch.rand(B, device=device) < drop_prob
        drop_signal = drop & (torch.rand(B, device=device) < 0.5)
        drop_image = drop & ~drop_signal

        if drop_signal.any():
            sig_feat = sig_feat.clone()
            sig_feat[drop_signal] = 0
        if drop_image.any():
            img_feat = img_feat.clone()
            img_feat[drop_image] = 0
        return sig_feat, img_feat


# ---------------------------------------------------------------------------
# Pretrained weight loading (GEM ECG-CoCa)
# ---------------------------------------------------------------------------

def _load_state_dict(checkpoint_path: str, map_location='cpu'):
    """Load state dict from a GEM checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # Remove 'module.' prefix from DDP training
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_gem_ecg_pretrained(
    model: ECGDiagModel,
    checkpoint_path: str,
    map_location: str = 'cpu',
):
    """Load GEM ECG-CoCa pretrained weights into signal_backbone."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"GEM pretrained checkpoint not found: {checkpoint_path}\n"
            f"Please download 'cpt_wfep_epoch_20.pt' from:\n"
            f"  https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB\n"
            f"and place it at: {checkpoint_path}"
        )

    state_dict = _load_state_dict(checkpoint_path, map_location)

    # Extract ECG weights, stripping 'ecg.' or 'module.ecg.' prefix
    ecg_state = {}
    for key, value in state_dict.items():
        if key.startswith('ecg.'):
            ecg_state[key[4:]] = value
        elif key.startswith('module.ecg.'):
            ecg_state[key[11:]] = value

    if not ecg_state:
        raise ValueError(
            f"No ECG weights found in checkpoint. Available key prefixes: "
            f"{set(k.split('.')[0] for k in state_dict.keys())}"
        )

    # Load into signal_backbone
    incompatible = model.signal_backbone.load_state_dict(ecg_state, strict=False)

    if incompatible.missing_keys:
        logger.warning(f"Missing keys when loading ECG backbone: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")

    loaded = len(ecg_state) - len(incompatible.unexpected_keys)
    logger.info(f"Loaded {loaded} ECG backbone parameter tensors from {checkpoint_path}")
    print(f"GEM ECG pretrained weights loaded: {loaded} parameter tensors")
