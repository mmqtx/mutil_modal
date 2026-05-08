"""
DDP Training Script for ECG Diagnostic Model.

Features Three-Layer Dynamic Compensation Loss:
1. Sample-level: Focal Loss (gamma=2) - down-weights easy examples
2. Class-level: Learned Class Weights - up-weights difficult classes
3. Task-level: Uncertainty Weighting - balances task gradients

Usage (2x4090):
  torchrun --nproc_per_node=2 train.py
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import *
from config import DATA_VERSION, get_label_config
from data.dataset import ECGMultiModalDataset
from models.model import ECGDiagModel, load_gem_ecg_pretrained
from models.losses import DynamicMultiTaskLoss
from models.contrastive_heads import MultiHeadContrastive


# ---------------------------------------------------------------------------
# Build head_subtasks config from dataset vocab
# ---------------------------------------------------------------------------

def build_head_subtasks(vocab, data_version="v2"):
    """Convert dataset vocab into the subtask config expected by DiagnosticChain.

    Args:
        vocab: vocabulary dict from dataset
        data_version: "v2", "v3", or "v4"

    Returns:
        head_subtasks config dict
    """
    base_config = {
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
            ("lvh", 2, "mc"),
            ("rvh", 2, "mc"),
            ("voltage", len(vocab["voltage"]), "mc"),
        ],
        "qt_electrolytes": [
            ("qt_status", len(vocab["qt_status"]), "mc"),
        ],
        "summary": [
            ("is_abnormal", 2, "mc"),
        ],
    }

    # Version-specific ischemia_infarct handling
    if data_version == "v4":
        # v4: 4 binary classifications
        base_config["ischemia_infarct"] = [
            ("st_elevation_present", 2, "mc"),
            ("st_depression_present", 2, "mc"),
            ("t_wave_abnormal", 2, "mc"),
            ("q_wave_present", 2, "mc"),
        ]
    elif data_version == "v3":
        # v3: 4 multi-class territory classifications
        territory_vocab = vocab.get("ischemia_territory", [])
        # Each territory type gets its own classification head
        base_config["ischemia_infarct"] = [
            ("st_elevation_territory", len(territory_vocab), "mc"),
            ("st_depression_territory", len(territory_vocab), "mc"),
            ("t_wave_abnormality", len(territory_vocab), "mc"),
            ("q_wave_territory", len(territory_vocab), "mc"),
        ]
    else:
        # v2: 48-dim multi-label (original)
        base_config["ischemia_infarct"] = [
            ("findings", 48, "ml"),
        ]

    return base_config


# ---------------------------------------------------------------------------
# Label mapping: dataset labels → flat "head.subtask" keys
# ---------------------------------------------------------------------------

def build_label_map(data_version="v2"):
    """Build LABEL_MAP based on data version.

    Args:
        data_version: "v2", "v3", or "v4"

    Returns:
        LABEL_MAP dict
    """
    base_map = {
        "rhythm_rate.rate_level":                ("rhythm_rate", 0, "mc"),
        "rhythm_rate.rhythm":                    ("rhythm_rate", 1, "mc"),
        "conduction_axis.axis":                  ("conduction_axis", 0, "mc"),
        "conduction_axis.pr_status":             ("conduction_axis", 1, "mc"),
        "conduction_axis.qrs_width":             ("conduction_axis", 2, "mc"),
        "conduction_axis.conduction_status":     ("conduction_axis", 3, "mc"),
        "voltage.lvh":                          ("voltage", 0, "mc"),
        "voltage.rvh":                          ("voltage", 1, "mc"),
        "voltage.voltage":                      ("voltage", 2, "mc"),
        "qt_electrolytes.qt_status":            ("qt_electrolytes", 0, "mc"),
        "summary.is_abnormal":                  ("summary", 0, "mc"),
    }

    # Version-specific ischemia_infarct handling
    if data_version == "v4":
        # v4: 4 binary classifications
        base_map.update({
            "ischemia_infarct.st_elevation_present": ("ischemia_infarct", 0, "mc"),
            "ischemia_infarct.st_depression_present": ("ischemia_infarct", 1, "mc"),
            "ischemia_infarct.t_wave_abnormal": ("ischemia_infarct", 2, "mc"),
            "ischemia_infarct.q_wave_present": ("ischemia_infarct", 3, "mc"),
        })
    elif data_version == "v3":
        # v3: 4 multi-class territory classifications
        base_map.update({
            "ischemia_infarct.st_elevation_territory": ("ischemia_infarct", 0, "mc"),
            "ischemia_infarct.st_depression_territory": ("ischemia_infarct", 1, "mc"),
            "ischemia_infarct.t_wave_abnormality": ("ischemia_infarct", 2, "mc"),
            "ischemia_infarct.q_wave_territory": ("ischemia_infarct", 3, "mc"),
        })
    else:
        # v2: 48-dim multi-label
        base_map["ischemia_infarct.findings"] = ("ischemia_infarct", 0, "ml")

    return base_map


# Global LABEL_MAP initialized based on DATA_VERSION
LABEL_MAP = build_label_map(DATA_VERSION)

# Ordered list of subtask keys (must match log_vars index)
SUBTASK_KEYS = list(LABEL_MAP.keys())


# ---------------------------------------------------------------------------
# Three-Layer Dynamic Compensation Loss Configuration
# ---------------------------------------------------------------------------

def build_task_config(vocab, data_version="v2"):
    """
    Build task configuration for DynamicMultiTaskLoss.

    Args:
        vocab: vocabulary dict from dataset
        data_version: "v2", "v3", or "v4"

    Returns:
        {task_name: (num_classes, task_type)}
        task_type: 'mc' for multi-class, 'ml' for multi-label
    """
    base_config = {
        "rhythm_rate.rate_level": (len(vocab["rate_level"]), "mc"),
        "rhythm_rate.rhythm": (len(vocab["rhythm"]), "mc"),
        "conduction_axis.axis": (len(vocab["axis"]), "mc"),
        "conduction_axis.pr_status": (len(vocab["pr_status"]), "mc"),
        "conduction_axis.qrs_width": (len(vocab["qrs_width"]), "mc"),
        "conduction_axis.conduction_status": (len(vocab["conduction_status"]), "mc"),
        "voltage.lvh": (2, "mc"),
        "voltage.rvh": (2, "mc"),
        "voltage.voltage": (len(vocab["voltage"]), "mc"),
        "qt_electrolytes.qt_status": (len(vocab["qt_status"]), "mc"),
        "summary.is_abnormal": (2, "mc"),
    }

    # Version-specific ischemia_infarct handling
    if data_version == "v4":
        # v4: 4 binary classifications
        base_config.update({
            "ischemia_infarct.st_elevation_present": (2, "mc"),
            "ischemia_infarct.st_depression_present": (2, "mc"),
            "ischemia_infarct.t_wave_abnormal": (2, "mc"),
            "ischemia_infarct.q_wave_present": (2, "mc"),
        })
    elif data_version == "v3":
        # v3: 4 multi-class territory classifications
        territory_vocab = vocab.get("ischemia_territory", [])
        base_config.update({
            "ischemia_infarct.st_elevation_territory": (len(territory_vocab), "mc"),
            "ischemia_infarct.st_depression_territory": (len(territory_vocab), "mc"),
            "ischemia_infarct.t_wave_abnormality": (len(territory_vocab), "mc"),
            "ischemia_infarct.q_wave_territory": (len(territory_vocab), "mc"),
        })
    else:
        # v2: 48-dim multi-label
        base_config["ischemia_infarct.findings"] = (48, "ml")

    return base_config


# ---------------------------------------------------------------------------
# Loss: Three-Layer Dynamic Compensation
# ---------------------------------------------------------------------------

class LossComputer(nn.Module):
    """
    Wrapper for DynamicMultiTaskLoss that integrates with existing training loop.
    """

    def __init__(self, task_config, device, static_class_weights=None, task_loss_weights=None):
        super().__init__()
        self.device = device
        self.dynamic_loss = DynamicMultiTaskLoss(
            task_config=task_config,
            focal_gamma=FOCAL_GAMMA,  # 从config读取
            device=device,
            static_class_weights=static_class_weights,
            task_loss_weights=task_loss_weights,
        )

    def compute(self, logits_dict, labels):
        """
        Compute three-layer dynamic compensation loss.

        Returns:
            losses dict with keys:
                - 'total': total weighted loss
                - 'reg': regularization term
                - 'details': full breakdown for logging
        """
        total_loss, details = self.dynamic_loss(
            logits_dict=logits_dict,
            labels_dict=labels,
            label_map=LABEL_MAP,
        )

        # Build losses dict for compatibility with existing code
        losses = {
            'total': total_loss,
            'reg': torch.tensor(details.get('regularization', 0.0), device=self.device),
            'details': details,
        }

        # Add per-task raw losses for monitoring
        for task_name, task_details in details.items():
            if isinstance(task_details, dict) and 'raw_loss' in task_details:
                losses[task_name] = torch.tensor(task_details['raw_loss'], device=self.device)

        return losses

    def get_log_stats(self):
        """Get statistics for TensorBoard logging."""
        return self.dynamic_loss.get_log_stats()


def create_loss_computer(vocab, device, data_version="v2", static_class_weights=None, task_loss_weights=None):
    """Factory function to create LossComputer."""
    task_config = build_task_config(vocab, data_version)
    return LossComputer(
        task_config,
        device,
        static_class_weights=static_class_weights,
        task_loss_weights=task_loss_weights,
    )


def build_optimizer_param_groups(model, loss_computer, base_lr, encoder_lr_scale):
    """为预训练编码器和新训练模块设置不同学习率。

    解冻 GEM/CLIP 编码器时，预训练权重通常需要更保守的学习率；分类头、融合层、
    投影层和动态损失参数继续使用主学习率。
    """
    model_for_names = model.module if isinstance(model, DDP) else model
    encoder_prefixes = ("signal_backbone.", "image_backbone.")
    encoder_params = []
    other_params = []

    for name, param in model_for_names.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(encoder_prefixes):
            encoder_params.append(param)
        else:
            other_params.append(param)

    other_params.extend([p for p in loss_computer.parameters() if p.requires_grad])

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "new_modules"})
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": base_lr * encoder_lr_scale,
            "name": "pretrained_encoders",
        })
    n_other_params = sum(p.numel() for p in other_params)
    n_encoder_params = sum(p.numel() for p in encoder_params)
    return param_groups, n_other_params, n_encoder_params


def build_subtask_loss_weights(task_config, enabled):
    """根据配置生成子任务损失权重；默认全部为 1.0。"""
    if not enabled:
        return None
    return {
        task_name: float(SUBTASK_LOSS_WEIGHTS.get(task_name, 1.0))
        for task_name in task_config
    }


def make_split_indices(total_size, seed, train_ratio, val_ratio):
    """Create deterministic train/val/test indices."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    n_train = int(total_size * train_ratio)
    n_val = int(total_size * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_split_indices(split_indices, run_dir):
    path = os.path.join(run_dir, "split_indices.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)
    return path


def build_static_class_weights(dataset, indices, task_config, label_map, beta=0.999, max_weight=20.0):
    """Compute effective-number class weights from the training split labels."""
    counts = {
        task_name: torch.zeros(num_classes, dtype=torch.float64)
        for task_name, (num_classes, task_type) in task_config.items()
        if task_type == "mc"
    }

    for idx in indices:
        labels = dataset._encode_labels(dataset.records[idx]["structured_data"])
        for task_name in counts:
            label_group, label_idx, _ = label_map[task_name]
            target = labels[label_group]
            if target.dim() > 0 and target.numel() > 1:
                target = target[label_idx]
            counts[task_name][int(target.item())] += 1

    weights = {}
    for task_name, task_counts in counts.items():
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        safe_counts = task_counts.clamp(min=1.0)
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), safe_counts)
        class_weights = (1.0 - beta) / effective_num.clamp(min=1e-12)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)
        class_weights = class_weights.clamp(max=max_weight).float()
        weights[task_name] = class_weights
    return weights


def create_head_contrastive(head_subtasks, temperature=0.1):
    """Factory function to create MultiHeadContrastive for head-level contrastive learning.

    Returns:
        (head_contrastive, head_modes): 对比学习模块和每个头的模式字典
    """
    head_names = list(head_subtasks.keys())

    # 检查哪些头是多标签任务
    head_modes = {}
    for head_name, subtasks in head_subtasks.items():
        # 检查是否有子任务是 multi-label
        has_ml = any(task_type == "ml" for _, _, task_type in subtasks)
        head_modes[head_name] = "multi_label" if has_ml else "multi_class"

    head_contrastive = MultiHeadContrastive(
        head_names=head_names,
        temperature=temperature,
        head_modes=head_modes,
    )
    return head_contrastive, head_modes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits_dict, labels):
    """Compute per-subtask accuracy and macro-F1."""
    metrics = {}
    for logit_key, (label_key, idx, task_type) in LABEL_MAP.items():
        if logit_key not in logits_dict:
            continue
        preds = logits_dict[logit_key].detach().cpu()
        target = labels[label_key].cpu()

        if task_type == "ml":
            t = target.float()
            pred_bin = (torch.sigmoid(preds) > 0.5).float()
            acc = (pred_bin == t).float().mean().item()
            tp = (pred_bin * t).sum(dim=0)
            fp = (pred_bin * (1 - t)).sum(dim=0)
            fn = ((1 - pred_bin) * t).sum(dim=0)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()
        else:
            t = target[:, idx] if target.dim() > 1 else target
            pred_cls = preds.argmax(dim=-1)
            acc = (pred_cls == t.long()).float().mean().item()
            nc = preds.size(-1)
            f1s = []
            for c in range(nc):
                tp = ((pred_cls == c) & (t.long() == c)).float().sum()
                fp = ((pred_cls == c) & (t.long() != c)).float().sum()
                fn = ((pred_cls != c) & (t.long() == c)).float().sum()
                p = tp / (tp + fp + 1e-8)
                r = tp / (tp + fn + 1e-8)
                f1s.append((2 * p * r / (p + r + 1e-8)).item())
            f1 = sum(f1s) / nc

        metrics[logit_key] = {"acc": acc, "f1": f1}
    return metrics


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, scheduler, epoch, rank, args, loss_computer=None, head_contrastive=None, head_modes=None):
    model.train()
    loader.sampler.set_epoch(epoch)

    # Initialize smoothed metrics
    smooth = {k: {"loss": 0.0, "acc": 0.0, "f1": 0.0} for k in SUBTASK_KEYS}
    smooth["total"] = {"loss": 0.0}
    smooth["contrastive"] = {"loss": 0.0}
    smooth["head_contrastive"] = {"loss": 0.0}  # 新增
    n = 0
    t0 = time.time()

    # 是否需要返回头特征
    return_head_features = head_contrastive is not None

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", disable=rank != 0)):
        signal = batch["signal"].cuda(rank, non_blocking=True)
        image = batch["image"].cuda(rank, non_blocking=True)
        labels = batch["labels"]

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=args["amp"]):
            out = model(
                signal,
                image,
                return_head_features=return_head_features,
                modality_dropout_prob=args.get("modality_dropout_prob", 0.0),
                input_mode=args.get("input_mode", INPUT_MODE),
            )
            logits = out["logits"]
            aux = out["aux_losses"]

            # Three-layer dynamic compensation loss
            if loss_computer is not None:
                losses = loss_computer.module.compute(logits, labels)
            else:
                raise RuntimeError("loss_computer is required for training")

            # Head-level contrastive loss (新增)
            head_cont_loss = torch.tensor(0.0, device=signal.device)
            if head_contrastive is not None and "head_features" in out:
                head_features = out["head_features"]  # {head_name: (B, D)}
                # 准备每个头的标签
                head_labels = {}
                for head_name in head_features.keys():
                    if head_name not in labels:
                        continue

                    label_tensor = labels[head_name]

                    # 确保标签在同一设备上
                    if hasattr(label_tensor, 'device') and label_tensor.device != signal.device:
                        label_tensor = label_tensor.to(signal.device)

                    # 根据头的模式提取标签
                    mode = head_modes.get(head_name, "multi_class")
                    if mode == "multi_label":
                        # 多标签任务：使用整个标签张量
                        # ischemia_infarct: (B, 48)
                        head_labels[head_name] = label_tensor.float()
                    else:
                        # 多分类任务：取第一个子任务的标签
                        # rhythm_rate: (B, 2) -> 取第一列 (B,)
                        # conduction_axis: (B, 4) -> 取第一列 (B,)
                        # voltage: (B, 3) -> 取第一列 (B,)
                        # qt_electrolytes: (B, 1) -> 取第一列 (B,)
                        # summary: (B, 1) -> 取第一列 (B,)
                        if label_tensor.dim() > 1:
                            head_labels[head_name] = label_tensor[:, 0]
                        else:
                            head_labels[head_name] = label_tensor

                # 计算对比学习损失 - 使用 detach 避免梯度重复计算
                with torch.amp.autocast('cuda', enabled=args["amp"]):
                    head_cont_losses = head_contrastive(head_features, head_labels)
                    head_cont_loss = head_contrastive.get_total_loss(head_cont_losses)

            # Add contrastive losses
            total = losses["total"] + args["contrastive_weight"] * aux["contrastive"]
            # 只有当对比学习损失不是零时才添加
            if head_cont_loss.item() > 0:
                total = total + args.get("head_contrastive_weight", 0.0) * head_cont_loss

        if args["amp"]:
            scaler.scale(total).backward()
            if args["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if args["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
            optimizer.step()

        scheduler.step()

        # Accumulate metrics
        bs = signal.size(0)
        smooth["total"]["loss"] += total.item() * bs
        smooth["contrastive"]["loss"] += aux["contrastive"].item() * bs
        smooth["head_contrastive"]["loss"] += head_cont_loss.item() * bs  # 新增

        metrics = compute_metrics(logits, labels)
        for k in SUBTASK_KEYS:
            if k in metrics:
                smooth[k]["loss"] += losses.get(k, torch.tensor(0.0)).item() * bs
                smooth[k]["acc"] += metrics[k]["acc"] * bs
                smooth[k]["f1"] += metrics[k]["f1"] * bs
        n += bs

        # Periodic logging
        if step % args["log_interval"] == 0 and rank == 0:
            writer = args.get("_writer")
            gs = epoch * len(loader) + step
            if writer:
                writer.add_scalar("train/loss_total", losses["total"].item(), gs)
                writer.add_scalar("train/loss_contrastive", aux["contrastive"].item(), gs)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], gs)

                # Log three-layer dynamic compensation stats
                if loss_computer is not None:
                    log_stats = loss_computer.module.get_log_stats()
                    for stat_name, stat_value in log_stats.items():
                        if isinstance(stat_value, list):
                            # Log class weights as histogram or line plot
                            for i, w in enumerate(stat_value):
                                writer.add_scalar(f"{stat_name}_w{i}", w, gs)
                        else:
                            writer.add_scalar(stat_name, stat_value, gs)

            logging.info(f"[E{epoch} S{step:>4d}] loss={total.item():.4f}  "
                  f"contrastive={aux['contrastive'].item():.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  [{time.time()-t0:.0f}s]")

    # Epoch averages
    for k in smooth:
        for m in smooth[k]:
            smooth[k][m] /= max(n, 1)

    # Return class weight stats for epoch-level logging
    weight_stats = None
    if loss_computer is not None:
        weight_stats = loss_computer.module.get_log_stats()

    return smooth, weight_stats


@torch.no_grad()
def validate(model, loader, rank, args, loss_computer=None):
    model.eval()
    smooth = {k: {"loss": 0.0, "acc": 0.0, "f1": 0.0} for k in SUBTASK_KEYS}
    smooth["total"] = {"loss": 0.0}
    smooth["contrastive"] = {"loss": 0.0}
    n = 0

    for batch in tqdm(loader, desc="Val", disable=rank != 0):
        signal = batch["signal"].cuda(rank, non_blocking=True)
        image = batch["image"].cuda(rank, non_blocking=True)
        labels = batch["labels"]

        out = model(signal, image, input_mode=args.get("input_mode", INPUT_MODE))
        logits = out["logits"]
        aux = out["aux_losses"]

        # Use new loss computer if available
        if loss_computer is not None:
            losses = loss_computer.module.compute(logits, labels)
        else:
            raise RuntimeError("loss_computer is required for validation")

        total = losses["total"] + args["contrastive_weight"] * aux["contrastive"]

        metrics = compute_metrics(logits, labels)
        bs = signal.size(0)
        smooth["total"]["loss"] += total.item() * bs
        smooth["contrastive"]["loss"] += aux["contrastive"].item() * bs
        for k in SUBTASK_KEYS:
            if k in metrics:
                smooth[k]["loss"] += losses.get(k, torch.tensor(0.0)).item() * bs
                smooth[k]["acc"] += metrics[k]["acc"] * bs
                smooth[k]["f1"] += metrics[k]["f1"] * bs
        n += bs

    for k in smooth:
        for m in smooth[k]:
            smooth[k][m] /= max(n, 1)
    return smooth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--encoder-lr-scale", type=float, default=ENCODER_LR_SCALE,
                        help="解冻预训练信号/图像编码器时的学习率倍率，实际 encoder_lr = lr * scale")
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", action="store_true", default=AMP_ENABLED)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--save-every-epoch", action="store_true", default=SAVE_EVERY_EPOCH)
    parser.add_argument("--save-best-loss", action="store_true", default=SAVE_BEST_LOSS,
                        help="额外保存 best_loss.pt；默认关闭以减少磁盘占用")
    parser.add_argument("--contrastive-weight", type=float, default=CONTRASTIVE_WEIGHT)
    parser.add_argument("--freeze-signal-encoder", action="store_true", default=FREEZE_SIGNAL_ENCODER,
                        help="Freeze GEM ECG signal encoder")
    parser.add_argument("--unfreeze-signal-encoder", action="store_true", default=False,
                        help="Override config and train GEM ECG signal encoder")
    parser.add_argument("--freeze-image-encoder", action="store_true", default=FREEZE_IMAGE_ENCODER,
                        help="Freeze CLIP image encoder")
    parser.add_argument("--unfreeze-image-encoder", action="store_true", default=False,
                        help="Override config and train CLIP image encoder")
    parser.add_argument("--modality-dropout-prob", type=float, default=MODALITY_DROPOUT_PROB,
                        help="Probability of dropping exactly one modality per sample during training")
    parser.add_argument("--input-mode", type=str, default=INPUT_MODE,
                        choices=["dual", "signal", "image"],
                        help="多模态输入模式：dual=信号+图像，signal=只用信号，image=只用图像")
    parser.add_argument("--fusion-type", type=str, default=FUSION_TYPE,
                        choices=["cross_attention", "gated_cross_attention", "late_concat"],
                        help="融合方式：cross_attention=交叉注意力，gated_cross_attention=门控残差交叉注意力，late_concat=简单后融合")
    parser.add_argument("--no-signal-augmentation", action="store_true", default=False,
                        help="Disable all training-time ECG signal augmentations")
    parser.add_argument("--use-cutmix", action="store_true", default=USE_CUTMIX,
                        help="Enable signal CutMix augmentation (off by default)")
    parser.add_argument("--no-static-class-weights", action="store_true", default=False,
                        help="Disable static effective-number class weights")
    parser.add_argument("--use-subtask-loss-weights", action="store_true", default=USE_SUBTASK_LOSS_WEIGHTS,
                        help="启用 config.SUBTASK_LOSS_WEIGHTS 中的弱任务损失权重")
    # Head-level contrastive learning (新增)
    parser.add_argument("--use-head-contrastive", action="store_true", default=USE_HEAD_CONTRASTIVE,
                        help="使用分类头级别的对比学习 / Enable head-level contrastive learning")
    parser.add_argument("--no-head-contrastive", action="store_true", default=False,
                        help="禁用分类头对比学习 / Disable head-level contrastive learning")
    parser.add_argument("--head-contrastive-weight", type=float, default=HEAD_CONTRASTIVE_WEIGHT,
                        help="分类头对比学习损失权重 / Weight for head contrastive loss")
    parser.add_argument("--head-contrastive-temp", type=float, default=HEAD_CONTRASTIVE_TEMP,
                        help="分类头对比学习温度参数 / Temperature for head contrastive loss")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-model-only", action="store_true", default=False,
                        help="只从 checkpoint 加载模型/loss 状态，重新初始化 optimizer/scheduler 做短程微调")
    parser.add_argument("--pretrained-gem", type=str, default=GEM_PRETRAINED_PATH,
                        help="Path to GEM pretrained checkpoint (cpt_wfep_epoch_20.pt)")
    parser.add_argument("--pretrained-clip", type=str, default=CLIP_VIT_PATH,
                        help="Path to CLIP ViT model directory")
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT,
                        help="Root directory for all experiment outputs")
    parser.add_argument("--name", type=str, default=EXPERIMENT_NAME,
                        help="Experiment name")
    cli = parser.parse_args()
    args = vars(cli)

    # 处理分类头对比学习参数
    # 如果指定了 --no-head-contrastive，则禁用
    if args.get("no_head_contrastive", False):
        args["use_head_contrastive"] = False
    if args.get("unfreeze_signal_encoder", False):
        args["freeze_signal_encoder"] = False
    if args.get("unfreeze_image_encoder", False):
        args["freeze_image_encoder"] = False

    # ---- Seed for reproducibility ----
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- DDP setup ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    args["world_size"] = world_size

    if rank == 0:
        # Create structured output directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args["output_root"], args["name"], ts)
        ckpt_dir = os.path.join(run_dir, DIR_CHECKPOINTS)
        tb_dir = os.path.join(run_dir, DIR_TENSORBOARD)
        log_dir = os.path.join(run_dir, DIR_LOGS)
        for d in [ckpt_dir, tb_dir, log_dir]:
            os.makedirs(d, exist_ok=True)

        writer = SummaryWriter(tb_dir)
        # Text log file
        log_file = os.path.join(log_dir, "train.log")
        # Setup logging to write to both console and file
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        args["_writer"] = writer
        args["_run_dir"] = run_dir
        args["_ckpt_dir"] = ckpt_dir

        logging.info(f"Run dir: {run_dir}")
        logging.info(f"  checkpoints: {ckpt_dir}")
        logging.info(f"  tensorboard: {tb_dir}")
        logging.info(f"  logs:        {log_dir}")
        logging.info(f"  GPUs: {world_size}")

    # ---- Dataset ----
    load_signal = args["input_mode"] in {"dual", "signal"}
    load_image = args["input_mode"] in {"dual", "image"}
    train_dataset = ECGMultiModalDataset(
        JSONL_PATH,
        IMAGE_ROOT,
        is_train=True,
        load_signal=load_signal,
        load_image=load_image,
        use_signal_augmentation=not args.get("no_signal_augmentation", False),
        use_baseline_wander=USE_BASELINE_WANDER,
        use_cutmix=args.get("use_cutmix", False),
        use_random_masking=USE_RANDOM_MASKING,
    )
    eval_dataset = ECGMultiModalDataset(
        JSONL_PATH,
        IMAGE_ROOT,
        is_train=False,
        load_signal=load_signal,
        load_image=load_image,
        use_signal_augmentation=False,
    )
    head_subtasks = build_head_subtasks(train_dataset.vocab, DATA_VERSION)

    # Split into train (70%), val (10%), test (20%)
    total_size = len(train_dataset)
    split_indices = make_split_indices(total_size, SEED, TRAIN_RATIO, VAL_RATIO)
    n_train = len(split_indices["train"])
    n_val = len(split_indices["val"])
    n_test = len(split_indices["test"])

    train_ds = Subset(train_dataset, split_indices["train"])
    val_ds = Subset(eval_dataset, split_indices["val"])
    test_ds = Subset(eval_dataset, split_indices["test"])

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"],
                              sampler=DistributedSampler(train_ds),
                              num_workers=args["workers"], pin_memory=True, drop_last=True)
    # Rank 0 evaluates the full validation/test split while other ranks wait.
    val_loader = DataLoader(val_ds, batch_size=args["batch_size"],
                            shuffle=False, num_workers=args["workers"],
                            pin_memory=True, drop_last=False)
    # Test loader - only used for final evaluation
    test_loader = DataLoader(test_ds, batch_size=args["batch_size"],
                             shuffle=False, num_workers=args["workers"],
                             pin_memory=True, drop_last=False)

    if rank == 0:
        split_path = save_split_indices(split_indices, args["_run_dir"])
        logging.info(f"Train: {n_train}  Val: {n_val}  Test: {n_test} (held out until training ends)")
        logging.info(f"  split_indices: {split_path}")
        for head, subs in head_subtasks.items():
            logging.info(f"  {head}: {[(n, c) for n, c, _ in subs]}")
        logging.info(f"  contrastive_weight: {args['contrastive_weight']}")
        logging.info(f"  signal_augmentation: {not args.get('no_signal_augmentation', False)}")
        logging.info(f"  use_cutmix: {args.get('use_cutmix', False)}")
        logging.info(f"  modality_dropout_prob: {args.get('modality_dropout_prob', 0.0)}")
        logging.info(f"  input_mode: {args.get('input_mode', INPUT_MODE)}")
        logging.info(f"  fusion_type: {args.get('fusion_type', FUSION_TYPE)}")

    # ---- Model ----
    model = ECGDiagModel(
        # ECG signal encoder (GEM-aligned)
        seq_length=ECG_SEQ_LENGTH,
        lead_num=ECG_LEAD_NUM,
        signal_patch_size=ECG_PATCH_SIZE,
        signal_width=ECG_WIDTH,
        signal_layers=ECG_LAYERS,
        signal_heads=ECG_HEADS,
        signal_mlp_ratio=ECG_MLP_RATIO,
        embed_dim=EMBED_DIM,
        # Image encoder (HuggingFace CLIP)
        clip_model_path=args.get("pretrained_clip", CLIP_VIT_PATH),
        freeze_signal_encoder=args["freeze_signal_encoder"],
        freeze_image_encoder=args["freeze_image_encoder"],
        # Downstream
        fusion_dim=FUSION_DIM,
        fusion_heads=NUM_HEADS,
        fusion_num_layers=FUSION_NUM_LAYERS,
        fusion_dropout=FUSION_DROPOUT,
        fusion_type=args["fusion_type"],
        input_mode=args["input_mode"],
        head_subtasks=head_subtasks,
        chain_attn_heads=NUM_HEADS,
        chain_attn_layers=NUM_CHAIN_LAYERS,
        head_dropout=HEAD_DROPOUT,
        contrastive_weight=args["contrastive_weight"],
        # Uplift projection config
        uplift_hidden_dim=UPLIFT_HIDDEN_DIM,
        uplift_num_layers=UPLIFT_NUM_LAYERS,
        # Contrastive projection config
        contrastive_hidden_dim=CONTRASTIVE_HIDDEN_DIM,
        contrastive_out_dim=CONTRASTIVE_OUT_DIM,
        contrastive_num_layers=CONTRASTIVE_NUM_LAYERS,
        # Head-level contrastive learning
        use_head_contrastive=args["use_head_contrastive"],
        head_contrastive_weight=args["head_contrastive_weight"],
        head_contrastive_temp=args["head_contrastive_temp"],
    ).cuda(rank)

    # Load GEM pretrained weights for signal backbone (if checkpoint exists)
    gem_path = args.get("pretrained_gem", GEM_PRETRAINED_PATH)
    if os.path.isfile(gem_path):
        if rank == 0:
            logging.info(f"Loading GEM pretrained weights from {gem_path}")
        load_gem_ecg_pretrained(model, gem_path, map_location=f"cuda:{rank}")
    elif rank == 0:
        logging.info(f"GEM pretrained checkpoint not found at {gem_path}")
        logging.info(f"  Download from: https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB")
        logging.info(f"  Place at: pretrained/cpt_wfep_epoch_20.pt")
        logging.info(f"  Training with random initialization.")

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if rank == 0:
        total_p = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Params: {total_p:,} (trainable: {trainable:,})")

    # ---- Three-Layer Dynamic Compensation Loss ----
    # Create loss computer AFTER dataset is loaded (needs vocab)
    task_config = build_task_config(train_dataset.vocab, DATA_VERSION)
    static_class_weights = None
    if USE_STATIC_CLASS_WEIGHTS and not args.get("no_static_class_weights", False):
        static_class_weights = build_static_class_weights(
            train_dataset,
            split_indices["train"],
            task_config,
            LABEL_MAP,
            beta=CLASS_WEIGHT_BETA,
            max_weight=CLASS_WEIGHT_MAX,
        )
    loss_computer = create_loss_computer(
        train_dataset.vocab,
        torch.device(f"cuda:{rank}"),
        DATA_VERSION,
        static_class_weights=static_class_weights,
        task_loss_weights=build_subtask_loss_weights(
            task_config,
            args.get("use_subtask_loss_weights", False),
        ),
    )
    # Wrap loss_computer in DDP as well since it has learnable parameters
    loss_computer = DDP(loss_computer, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # ---- Head-level Contrastive Learning (新增) ----
    head_contrastive = None
    head_modes = {}  # 存储每个头的模式（multi_class/multi_label）
    if args["use_head_contrastive"]:
        head_contrastive, head_modes = create_head_contrastive(
            head_subtasks,
            temperature=args["head_contrastive_temp"]
        )
        head_contrastive = head_contrastive.cuda(rank)
        if rank == 0:
            logging.info(f"Head-level Contrastive Learning: ENABLED")
            logging.info(f"  Temperature: {args['head_contrastive_temp']}")
            logging.info(f"  Weight: {args['head_contrastive_weight']}")

    if rank == 0:
        # Count learnable class weights
        total_weight_params = sum(p.numel() for p in loss_computer.parameters())
        logging.info(f"Dynamic Loss: {len(loss_computer.module.dynamic_loss.task_names)} tasks")
        logging.info(f"  Learnable class weight params: {total_weight_params:,}")
        logging.info(f"  Focal gamma: {loss_computer.module.dynamic_loss.focal_gamma}")
        logging.info(f"  Static class weights: {static_class_weights is not None}")
        if args.get("use_subtask_loss_weights", False):
            enabled_weights = {
                task_name: weight
                for task_name, weight in SUBTASK_LOSS_WEIGHTS.items()
                if task_name in task_config
            }
            logging.info(f"  Subtask loss weights: {enabled_weights}")

    # ---- Optimizer ----
    # 预训练编码器使用更小学习率，新训练模块和动态损失参数使用主学习率。
    param_groups, n_other_params, n_encoder_params = build_optimizer_param_groups(
        model,
        loss_computer,
        args["lr"],
        args["encoder_lr_scale"],
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args["lr"], weight_decay=args["wd"])
    if rank == 0:
        logging.info(
            f"Optimizer groups: new_modules={n_other_params:,} params lr={args['lr']:.2e}; "
            f"pretrained_encoders={n_encoder_params:,} params "
            f"lr={args['lr'] * args['encoder_lr_scale']:.2e}"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args["amp"])
    total_steps = args["epochs"] * len(train_loader)

    def lr_lambda(step):
        if step < args["warmup_steps"]:
            return step / max(args["warmup_steps"], 1)
        progress = (step - args["warmup_steps"]) / max(total_steps - args["warmup_steps"], 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume ----
    start_epoch = 0
    if args["resume"]:
        ckpt = torch.load(args["resume"], map_location=f"cuda:{rank}")
        model.module.load_state_dict(ckpt["model"])
        # Also restore loss_computer state if available
        if "loss_computer" in ckpt:
            loss_computer.module.load_state_dict(ckpt["loss_computer"], strict=False)
        if args.get("resume_model_only", False):
            if rank == 0:
                logging.info(
                    f"Loaded model/loss state from {args['resume']} "
                    "with fresh optimizer and scheduler"
                )
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1

    # ---- Loop ----
    best_metric = 0.0  # Track best Macro-F1 mean instead of loss
    best_val_loss = float("inf")  # Also track best loss for reference
    for epoch in range(start_epoch, args["epochs"]):
        train_m, weight_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler, epoch, rank, args,
            loss_computer=loss_computer, head_contrastive=head_contrastive, head_modes=head_modes)

        if rank == 0:
            val_m = validate(model.module, val_loader, 0, args, loss_computer=loss_computer)
            writer = args["_writer"]

            # Log per-head metrics
            for k in SUBTASK_KEYS:
                if k in val_m:
                    writer.add_scalar(f"val/acc_{k}", val_m[k]["acc"], epoch)
                    writer.add_scalar(f"val/f1_{k}", val_m[k]["f1"], epoch)
            writer.add_scalar("val/loss_total", val_m["total"]["loss"], epoch)
            writer.add_scalar("val/loss_contrastive", val_m["contrastive"]["loss"], epoch)

            # Log three-layer dynamic compensation stats (learned class weights)
            if weight_stats:
                for stat_name, stat_value in weight_stats.items():
                    if isinstance(stat_value, float):
                        writer.add_scalar(f"val/{stat_name}", stat_value, epoch)

            # Calculate Macro-F1 mean across all subtasks
            macro_f1s = [val_m[k]["f1"] for k in SUBTASK_KEYS if k in val_m]
            macro_f1_mean = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
            writer.add_scalar("val/macro_f1_mean", macro_f1_mean, epoch)

            # Print epoch summary
            parts = [f"[E{epoch:>2d}] train={train_m['total']['loss']:.4f}  "
                     f"val={val_m['total']['loss']:.4f}  "
                     f"macro_f1={macro_f1_mean:.4f}"]
            for chain_name in CHAIN_ORDER:
                sub_accs = [val_m.get(f"{chain_name}.{s}", {}).get("acc", 0)
                            for s, _, _ in head_subtasks.get(chain_name, [])]
                if sub_accs:
                    parts.append(f"{chain_name}:acc={sum(sub_accs)/len(sub_accs):.3f}")
            logging.info("  ".join(parts))

            # Save best based on Macro-F1 mean (better for imbalanced data)
            if macro_f1_mean > best_metric:
                best_metric = macro_f1_mean
                best_val_loss = val_m["total"]["loss"]
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss_computer": loss_computer.module.state_dict(),
                    "epoch": epoch,
                    "macro_f1_mean": macro_f1_mean,
                    "val_loss": val_m["total"]["loss"],
                }, os.path.join(args["_ckpt_dir"], "best.pt"))
                logging.info(f"  ** Best macro_f1={best_metric:.4f} (val_loss={best_val_loss:.4f})")

            # 可选保存 best_loss.pt。默认关闭，避免额外占用约 2GB 磁盘空间。
            if args.get("save_best_loss", False) and val_m["total"]["loss"] < best_val_loss:
                best_val_loss = val_m["total"]["loss"]
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss_computer": loss_computer.module.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                }, os.path.join(args["_ckpt_dir"], "best_loss.pt"))

            if args.get("save_every_epoch", False) and (epoch + 1) % args["save_interval"] == 0:
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss_computer": loss_computer.module.state_dict(),
                    "epoch": epoch,
                }, os.path.join(args["_ckpt_dir"], f"epoch_{epoch}.pt"))

        dist.barrier()

    if rank == 0:
        writer.close()

        # ---- Final Test Set Evaluation ----
        logging.info("\n" + "="*70)
        logging.info("TRAINING COMPLETE - EVALUATING ON HELD-OUT TEST SET")
        logging.info("="*70)

        # Load best model (based on Macro-F1)
        best_ckpt = torch.load(os.path.join(args["_ckpt_dir"], "best.pt"))
        model.module.load_state_dict(best_ckpt["model"])
        best_epoch = best_ckpt["epoch"]
        best_macro_f1 = best_ckpt.get("macro_f1_mean", "N/A")
        logging.info(f"Loaded best model from epoch {best_epoch} (macro_f1={best_macro_f1})")

        # Evaluate on test set
        test_m = validate(model.module, test_loader, 0, args, loss_computer=loss_computer)

        # Calculate test metrics
        test_macro_f1s = [test_m[k]["f1"] for k in SUBTASK_KEYS if k in test_m]
        test_macro_f1_mean = sum(test_macro_f1s) / len(test_macro_f1s) if test_macro_f1s else 0.0

        logging.info(f"\nTEST SET RESULTS ({n_test} samples):")
        logging.info(f"  Overall Macro-F1: {test_macro_f1_mean:.4f}")
        logging.info(f"  Overall Loss: {test_m['total']['loss']:.4f}")
        logging.info(f"\nPer-task Macro-F1:")

        # Group results by chain
        for chain_name in CHAIN_ORDER:
            sub_f1s = [test_m.get(f"{chain_name}.{s}", {}).get("f1", 0.0)
                       for s, _, _ in head_subtasks.get(chain_name, [])]
            if sub_f1s:
                avg_f1 = sum(sub_f1s) / len(sub_f1s)
                logging.info(f"  {chain_name:20s}: {avg_f1:.4f}")
                for s, _, _ in head_subtasks.get(chain_name, []):
                    f1 = test_m.get(f"{chain_name}.{s}", {}).get("f1", 0.0)
                    logging.info(f"    {s}={f1:.3f}")

        logging.info("="*70)
        logging.info("Done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
