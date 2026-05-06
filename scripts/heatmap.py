"""
GradCAM Attention Heatmap — ECG Diagnostic Visualization

生成ECG诊断模型的GradCAM注意力热力图，展示模型在做每个诊断步骤时
关注图像的哪些区域。

特性:
  - 6个诊断步骤的热力图（心律心率、传导电轴、电压肥厚、缺血梗死、QT电解质、总结诊断）
  - 可配置的网格大小（控制关注区域的粗细）
  - 高斯模糊平滑处理
  - 支持单独查看和组合查看

Usage:
  # 基本用法（默认8x8网格）
  python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

  # 使用更粗的网格（6x6）显示更大的关注区域
  python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --grid-size 6

  # 使用更细的网格（12x12）显示更精细的细节
  python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --grid-size 12

  # 指定评估的样本索引
  python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --sample-idx 100

  # 调整热力图透明度（0.7 = 更明显的热力图）
  python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --alpha 0.7

  # 调整模糊程度（值越大越平滑）
  python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --blur 7

Parameters:
  --checkpoint      模型检查点路径 (必需)
  --split           选择数据集: train/val/test (默认: test)
  --device          设备 (默认: cuda:0)
  --output-dir      输出目录 (默认: ./heatmaps)
  --sample-idx      指定样本索引 (默认: 自动查找正确分类的样本)
  --max-search      最大搜索样本数 (默认: 200)
  --alpha           热力图透明度 0-1 (默认: 0.5)
  --grid-size       热力图网格大小 (默认: 8, 越小=区域越大)
  --blur            高斯模糊核大小 (默认: 5, 越大=越平滑)

Output:
  在输出目录生成:
    - gradcam_*.png         每个诊断步骤的单独热力图
    - gradcam_combined.png   6个热力图的组合图
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import *
from data.dataset import ECGMultiModalDataset
from models.model import ECGDiagModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Coarser grid for better visualization (8x8 instead of 24x24)
# Each cell represents 42x42 pixels instead of 14x14
PATCH_GRID = 8  # 8x8 grid for heatmap (each cell = 42x42 pixels)

TARGET_SUBTASKS = [
    "rhythm_rate.rate_level",
    "conduction_axis.axis",
    "voltage.lvh",
    "ischemia_infarct.findings",
    "qt_electrolytes.qt_status",
    "summary.is_abnormal",
]

STEP_TITLES = [
    "Rhythm & Rate",
    "Conduction & Axis",
    "Voltage & Hypertrophy",
    "Ischemia & Infarct",
    "QT & Electrolytes",
    "Summary Diagnosis",
]

LABEL_MAP = {
    "rhythm_rate.rate_level":            ("rhythm_rate", 0, "mc"),
    "rhythm_rate.rhythm":                ("rhythm_rate", 1, "mc"),
    "conduction_axis.axis":              ("conduction_axis", 0, "mc"),
    "conduction_axis.pr_status":         ("conduction_axis", 1, "mc"),
    "conduction_axis.qrs_width":         ("conduction_axis", 2, "mc"),
    "conduction_axis.conduction_status": ("conduction_axis", 3, "mc"),
    "voltage.lvh":                       ("voltage", 0, "mc"),
    "voltage.rvh":                       ("voltage", 1, "mc"),
    "voltage.voltage":                   ("voltage", 2, "mc"),
    "ischemia_infarct.findings":         ("ischemia_infarct", 0, "ml"),
    "qt_electrolytes.qt_status":         ("qt_electrolytes", 0, "mc"),
    "summary.is_abnormal":               ("summary", 0, "mc"),
}


def build_head_subtasks(vocab):
    return {
        "rhythm_rate": [
            ("rate_level", len(vocab["rate_level"]), "mc"),
            ("rhythm", len(vocab["rhythm"]), "mc"),
        ],
        "conduction_axis": [
            ("axis", len(vocab["axis"]), "mc"),
            ("pr_status", len(vocab["pr_status"]), "mc"),
            ("qrs_width", len(vocab["qrs_width"]), "mc"),
            ("conduction_status", len(vocab["conduction_status"]), "mc"),
        ],
        "voltage": [
            ("lvh", 2, "mc"), ("rvh", 2, "mc"),
            ("voltage", len(vocab["voltage"]), "mc"),
        ],
        "ischemia_infarct": [("findings", 48, "ml")],
        "qt_electrolytes": [("qt_status", len(vocab["qt_status"]), "mc")],
        "summary": [("is_abnormal", 2, "mc")],
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path, head_subtasks, device="cuda:0"):
    model = ECGDiagModel(
        embed_dim=EMBED_DIM, clip_model_path=CLIP_VIT_PATH,
        fusion_dim=FUSION_DIM, fusion_heads=NUM_HEADS, fusion_num_layers=FUSION_NUM_LAYERS,
        head_subtasks=head_subtasks,
        chain_attn_heads=NUM_HEADS, chain_attn_layers=NUM_CHAIN_LAYERS,
        uplift_hidden_dim=UPLIFT_HIDDEN_DIM, uplift_num_layers=UPLIFT_NUM_LAYERS,
        contrastive_hidden_dim=CONTRASTIVE_HIDDEN_DIM,
        contrastive_out_dim=CONTRASTIVE_OUT_DIM, contrastive_num_layers=CONTRASTIVE_NUM_LAYERS,
        freeze_signal_encoder=FREEZE_SIGNAL_ENCODER,
        freeze_image_encoder=FREEZE_IMAGE_ENCODER,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


# ---------------------------------------------------------------------------
# Sample finding
# ---------------------------------------------------------------------------
def find_sample(model, dataset, device, max_search=200):
    """Find a correctly classified sample. Returns (sample_dict, original_pil)."""
    model.eval()
    n_correct_needed = len(TARGET_SUBTASKS)

    for i in range(min(max_search, len(dataset))):
        sample = dataset[i]
        signal = sample["signal"].unsqueeze(0).to(device)
        image = sample["image"].unsqueeze(0).to(device)
        labels = sample["labels"]

        with torch.no_grad():
            out = model(signal, image)
        logits = out["logits"]

        # Check correctness on all 6 target subtasks
        all_correct = True
        preds_info = {}
        for key in TARGET_SUBTASKS:
            if key not in logits:
                all_correct = False
                break
            (label_key, idx, task_type) = LABEL_MAP[key]
            pred = logits[key].argmax(dim=-1).item()
            target = labels[label_key]
            if target.numel() > 1:
                target = target[idx]
            target = target.item()
            correct = (pred == target)
            if not correct:
                all_correct = False
            preds_info[key] = {"pred": pred, "target": target, "correct": correct}

        if all_correct:
            # Load original PIL image
            rec = dataset.records[i]
            img_rel_path = rec["image_paths"][0]
            img_abs_path = os.path.join(dataset.image_root, img_rel_path)
            original_pil = Image.open(img_abs_path).convert("RGB")
            print(f"Found correctly classified sample: idx={i}, id={sample['id']}")
            return sample, original_pil, i

    # Fallback: just use first sample
    print(f"No fully correct sample found in {max_search}. Using first sample.")
    sample = dataset[0]
    rec = dataset.records[0]
    img_abs_path = os.path.join(dataset.image_root, rec["image_paths"][0])
    original_pil = Image.open(img_abs_path).convert("RGB")
    return sample, original_pil, 0


# ---------------------------------------------------------------------------
# GradCAM hooks
# ---------------------------------------------------------------------------
class GradCAMHooks:
    def __init__(self):
        self.activations = None
        self.gradients = None
        self._fwd_handle = None
        self._bwd_handle = None

    def register(self, model):
        target_layer = model.image_backbone.vision_model.vision_model.encoder.layers[-1]
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def remove(self):
        if self._fwd_handle:
            self._fwd_handle.remove()
        if self._bwd_handle:
            self._bwd_handle.remove()

    def _fwd_hook(self, module, input, output):
        # HuggingFace encoder layers return tuple (hidden_states, ...)
        if isinstance(output, tuple):
            self.activations = output[0]
        else:
            self.activations = output

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


# ---------------------------------------------------------------------------
# GradCAM forward (bypasses @torch.no_grad on CLIPVisionEncoder)
# ---------------------------------------------------------------------------
def gradcam_forward(model, signal, image):
    """
    Custom forward that allows gradient flow through the image path.
    Bypasses CLIPVisionEncoder.forward's @torch.no_grad() by calling
    vision_model directly.

    Uses a weighted sum: CLS token + mean of patch tokens, so gradients
    flow to all spatial positions while still leveraging CLS semantics.
    """
    # Signal path (no gradient needed)
    with torch.no_grad():
        sig_feat = model.signal_backbone(signal)

    # Image path — call vision_model directly to bypass @torch.no_grad wrapper
    img_out = model.image_backbone.vision_model(image)
    hidden = img_out.last_hidden_state  # (B, 577, 1024)
    # Use CLS token as main feature (same as normal forward)
    # But also keep patch contributions via attention-weighted sum
    # so gradients flow differentially to different spatial positions
    cls_token = hidden[:, 0]  # (B, 1024)
    patch_tokens = hidden[:, 1:]  # (B, 576, 1024)
    # Attention-weighted pooling: use CLS as query, patches as keys
    # This naturally creates spatial variation in gradient flow
    attn_weights = torch.bmm(
        cls_token.unsqueeze(1),  # (B, 1, 1024)
        patch_tokens.transpose(1, 2)  # (B, 1024, 576)
    ).squeeze(1)  # (B, 576)
    attn_weights = torch.softmax(attn_weights, dim=-1)  # (B, 576)
    weighted_patches = torch.bmm(
        attn_weights.unsqueeze(1),  # (B, 1, 576)
        patch_tokens  # (B, 576, 1024)
    ).squeeze(1)  # (B, 1024)
    img_feat = weighted_patches

    # Uplift both to fusion_dim
    sig_up = model.signal_uplift(sig_feat.detach())
    img_up = model.image_uplift(img_feat)

    # Fusion
    fused = model.fusion(sig_up, img_up)

    # Diagnostic chain
    chain_logits, _ = model.chain(fused)

    return chain_logits


# ---------------------------------------------------------------------------
# Compute single GradCAM
# ---------------------------------------------------------------------------
def compute_gradcam(model, signal, image, hooks, target_key, device, blur=5):
    """
    Compute GradCAM heatmap for one diagnostic step.

    Returns:
        cam: (336, 336) numpy array, values in [0, 1]
    """
    model.zero_grad()

    # Temporarily enable gradients on CLIP
    saved_requires_grad = {}
    for name, param in model.image_backbone.named_parameters():
        saved_requires_grad[name] = param.requires_grad
        param.requires_grad_(True)

    # Image needs gradient
    image = image.detach().requires_grad_(True)

    # Forward pass
    chain_logits = gradcam_forward(model, signal, image)

    # Get target score
    target_logits = chain_logits[target_key]  # (1, C) or (1, 48)

    if target_key == "ischemia_infarct.findings":
        # Multi-label: use mean of sigmoid as target
        target_score = torch.sigmoid(target_logits).mean()
    else:
        # Multi-class: use predicted class logit
        pred_class = target_logits.argmax(dim=-1).item()
        target_score = target_logits[0, pred_class]

    # Backward
    target_score.backward(retain_graph=False)

    # Get activations and gradients
    act = hooks.activations.detach()    # (1, 577, 1024)
    grad = hooks.gradients.detach()      # (1, 577, 1024)

    # Remove CLS token, keep patches only
    act = act[:, 1:, :]   # (1, 576, 1024)
    grad = grad[:, 1:, :]

    # Global average pool of gradients -> channel weights
    weights = grad.mean(dim=(0, 1))  # (1024,)

    # Weighted combination
    cam = (act * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (1, 576)
    cam = cam.squeeze(0).cpu().numpy()  # (576,)

    # Reshape to 24x24 first (original patch grid)
    cam = cam.reshape(24, 24)

    # ReLU (only positive)
    cam = np.maximum(cam, 0)

    # Downsample to coarser grid (PATCH_GRID x PATCH_GRID)
    # This creates larger, more visible regions
    cam_small = cv2.resize(cam, (PATCH_GRID, PATCH_GRID), interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur for smoother heatmaps
    if blur > 0:
        kernel_size = blur if blur % 2 == 1 else blur + 1
        cam_small = cv2.GaussianBlur(cam_small, (kernel_size, kernel_size), 0)

    # Normalize using percentile-based approach with wider range
    if cam_small.max() > 0:
        # Use 95th percentile as max (instead of max or 99th)
        # This allows top 5% regions to be clearly visible
        p95 = np.percentile(cam_small, 95)
        if p95 > 0:
            cam_small = np.clip(cam_small / p95, 0, 1)
        else:
            cam_small = cam_small / cam_small.max()

        # Mild power transformation - much gentler
        cam_small = np.power(cam_small, 0.8)  # Less aggressive than 0.5

    # Upsample to image size
    cam_tensor = torch.from_numpy(cam_small).float().unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(cam_tensor, size=(IMAGE_SIZE, IMAGE_SIZE),
                               mode='bilinear', align_corners=False)
    cam = cam_tensor.squeeze().numpy()

    # Optional: light smoothing only (skip for sharper heatmaps)
    # Skip final smoothing to preserve detail

    # Restore requires_grad
    for name, param in model.image_backbone.named_parameters():
        param.requires_grad_(saved_requires_grad[name])

    return cam


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def apply_heatmap(image_rgb, cam, alpha=0.5):
    """Overlay heatmap on image with balanced visibility."""
    # Apply colormap directly without additional enhancement
    # The input cam should already be normalized to [0, 1]
    cam_clipped = np.clip(cam, 0, 1)

    # Use JET colormap (blue->red, better for ECG)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_clipped), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = (image_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay


def save_individual(original_pil, cam, title, save_path, alpha=0.5):
    """Save individual heatmap: original + overlay side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img_np = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    overlay = apply_heatmap(img_np, cam, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original ECG", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(f"GradCAM: {title}", fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_combined(original_pil, cams, titles, save_path, alpha=0.5):
    """Save 2x3 grid of all heatmaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img_np = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # First column: original image spanning both rows
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original ECG", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(img_np)
    axes[1, 0].axis('off')

    for idx, (cam, title) in enumerate(zip(cams, titles)):
        row = idx // 3
        col = idx % 3 + 1
        overlay = apply_heatmap(img_np, cam, alpha)
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(title, fontsize=13, fontweight='bold')
        axes[row, col].axis('off')

    # Hide unused subplot
    axes[1, 0].set_visible(False)

    plt.suptitle("GradCAM Attention Heatmaps — Diagnostic Chain", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GradCAM Heatmap for ECG Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate heatmaps with default 8x8 grid
  python heatmap.py --checkpoint outputs/ecg_diag/<ts>/checkpoints/best.pt

  # Use coarser grid (6x6) for larger regions
  python heatmap.py --checkpoint outputs/ecg_diag/<ts>/checkpoints/best.pt --grid-size 6

  # Use finer grid (12x12) for more detail
  python heatmap.py --checkpoint outputs/ecg_diag/<ts>/checkpoints/best.pt --grid-size 12
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="./heatmaps")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="Specific sample index (auto-find if not set)")
    parser.add_argument("--max-search", type=int, default=200,
                        help="Max samples to search for correct classification")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Heatmap transparency (0=only image, 1=only heatmap)")
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Heatmap grid size (default: 8, lower=larger regions)")
    parser.add_argument("--blur", type=int, default=5,
                        help="Gaussian blur kernel size (default: 5, higher=smoother)")
    args = parser.parse_args()

    # Update global PATCH_GRID
    global PATCH_GRID
    PATCH_GRID = args.grid_size
    print(f"Using {PATCH_GRID}x{PATCH_GRID} heatmap grid (each cell = {336//PATCH_GRID}x{336//PATCH_GRID} pixels)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = ECGMultiModalDataset(
        jsonl_path=JSONL_PATH, image_root=IMAGE_ROOT, is_train=False,
    )
    head_subtasks = build_head_subtasks(dataset.vocab)

    # Load model
    model = load_model(args.checkpoint, head_subtasks, args.device)

    # Find sample
    if args.sample_idx is not None:
        idx = args.sample_idx
        sample = dataset[idx]
        rec = dataset.records[idx]
        img_abs_path = os.path.join(dataset.image_root, rec["image_paths"][0])
        original_pil = Image.open(img_abs_path).convert("RGB")
        print(f"Using sample idx={idx}, id={sample['id']}")
    else:
        sample, original_pil, idx = find_sample(
            model, dataset, args.device, args.max_search
        )

    # Register hooks
    hooks = GradCAMHooks()
    hooks.register(model)

    # Compute 6 heatmaps
    signal = sample["signal"].unsqueeze(0).to(args.device)
    image = sample["image"].unsqueeze(0).to(args.device)

    cams = []
    for key, title in zip(TARGET_SUBTASKS, STEP_TITLES):
        print(f"  Computing GradCAM for: {title} ({key})")
        cam = compute_gradcam(model, signal, image, hooks, key, args.device, blur=args.blur)
        cams.append(cam)

        fname = f"gradcam_{key.replace('.', '_')}.png"
        save_path = os.path.join(args.output_dir, fname)
        save_individual(original_pil, cam, title, save_path, alpha=args.alpha)
        print(f"    Saved: {save_path}")

    # Combined figure
    combined_path = os.path.join(args.output_dir, "gradcam_combined.png")
    save_combined(original_pil, cams, STEP_TITLES, combined_path, alpha=args.alpha)
    print(f"\n  Combined figure: {combined_path}")

    # Cleanup
    hooks.remove()
    print("Done.")


if __name__ == "__main__":
    main()
