"""
评估脚本 — 在指定数据集上生成完整评估报告，包含混淆矩阵和详细指标。

与train.py保持一致的数据划分 (7:1:2):
  - Train (70%): 训练集
  - Val (10%): 验证集，用于选择最佳模型
  - Test (20%): 测试集，用于最终评估

Usage:
  # 默认在test集上评估
  python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

  # 在val集上评估
  python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split val

  # 在train集上评估（检查过拟合）
  python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split train
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import *
from config import DATA_VERSION, get_label_config
from data.dataset import ECGMultiModalDataset
from models.model import ECGDiagModel, load_gem_ecg_pretrained

# Ischemia labels
_ISCHEMIA_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
_ISCHEMIA_SUBTYPES = ["st_elevation", "st_depression", "t_wave_alt", "q_wave"]

# Label map (same as train.py) - version-aware
def build_label_map(data_version="v2"):
    """Build LABEL_MAP based on data version."""
    base_map = {
        "rhythm_rate.rate_level":            ("rhythm_rate", 0, "mc"),
        "rhythm_rate.rhythm":                ("rhythm_rate", 1, "mc"),
        "conduction_axis.axis":              ("conduction_axis", 0, "mc"),
        "conduction_axis.pr_status":         ("conduction_axis", 1, "mc"),
        "conduction_axis.qrs_width":         ("conduction_axis", 2, "mc"),
        "conduction_axis.conduction_status": ("conduction_axis", 3, "mc"),
        "voltage.lvh":                       ("voltage", 0, "mc"),
        "voltage.rvh":                       ("voltage", 1, "mc"),
        "voltage.voltage":                   ("voltage", 2, "mc"),
        "qt_electrolytes.qt_status":         ("qt_electrolytes", 0, "mc"),
        "summary.is_abnormal":               ("summary", 0, "mc"),
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


def make_split_indices(total_size, seed, train_ratio, val_ratio):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    n_train = int(total_size * train_ratio)
    n_val = int(total_size * val_ratio)
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def load_split_indices(path, total_size):
    with open(path, "r", encoding="utf-8") as f:
        split_indices = json.load(f)
    seen = sorted(split_indices["train"] + split_indices["val"] + split_indices["test"])
    if seen != list(range(total_size)):
        raise ValueError(f"Invalid split indices in {path}: do not cover dataset exactly once")
    return split_indices


def load_model(
    checkpoint_path,
    head_subtasks,
    device="cuda:0",
    fusion_type=FUSION_TYPE,
    input_mode=INPUT_MODE,
):
    """Load trained model."""
    model = ECGDiagModel(
        embed_dim=EMBED_DIM, clip_model_path=CLIP_VIT_PATH,
        fusion_dim=FUSION_DIM, fusion_heads=NUM_HEADS, fusion_num_layers=FUSION_NUM_LAYERS,
        fusion_type=fusion_type, input_mode=input_mode,
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
    model.to(device).eval()
    return model


def confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix. Returns (num_classes, num_classes) numpy array."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def print_confusion_matrix(cm, class_names, title=""):
    """Pretty print confusion matrix."""
    n = len(class_names)
    # Header
    print(f"\n  {title}")
    max_name = max(len(str(c)) for c in class_names) + 2
    header = " " * max_name + "".join(f"{str(c):^{max_name}}" for c in class_names)
    print(f"  {header}")
    print(f"  {'─' * len(header)}")
    for i, name in enumerate(class_names):
        row = f"  {str(name):<{max_name}}" + "".join(f"{cm[i][j]:^{max_name}}" for j in range(n))
        print(row)
    print()


def classification_report(cm, class_names):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    report = {}
    total_support = sum(cm.sum(axis=1))
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }

    # Macro avg
    p_avg = np.mean([v["precision"] for v in report.values()])
    r_avg = np.mean([v["recall"] for v in report.values()])
    f1_avg = np.mean([v["f1-score"] for v in report.values()])
    report["macro_avg"] = {
        "precision": round(p_avg, 4),
        "recall": round(r_avg, 4),
        "f1-score": round(f1_avg, 4),
        "support": total_support,
    }

    # Weighted avg
    p_w = sum(report[n]["precision"] * report[n]["support"] for n in class_names) / total_support
    r_w = sum(report[n]["recall"] * report[n]["support"] for n in class_names) / total_support
    f1_w = sum(report[n]["f1-score"] * report[n]["support"] for n in class_names) / total_support
    report["weighted_avg"] = {
        "precision": round(p_w, 4),
        "recall": round(r_w, 4),
        "f1-score": round(f1_w, 4),
        "support": total_support,
    }
    return report


def load_thresholds(path):
    """读取验证集阈值校准结果，返回 task_name -> threshold。"""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    thresholds = payload.get("thresholds", payload)
    return {
        task_name: float(values["best"]["threshold"])
        for task_name, values in thresholds.items()
        if isinstance(values, dict) and "best" in values and "threshold" in values["best"]
    }


def load_logit_biases(path):
    """读取验证集 logit bias 校准结果，返回 task_name -> bias tensor。"""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    biases = payload.get("logit_biases", payload.get("biases", {}))
    return {
        task_name: torch.tensor(values["bias"], dtype=torch.float32)
        for task_name, values in biases.items()
        if isinstance(values, dict) and "bias" in values
    }


def print_report(report, class_names):
    """Pretty print classification report."""
    print(f"  {'':>20s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    print(f"  {'─' * 65}")
    for name in class_names:
        r = report[name]
        print(f"  {str(name):>20s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10d}")
    print(f"  {'─' * 65}")
    for avg in ["macro_avg", "weighted_avg"]:
        r = report[avg]
        print(f"  {avg:>20s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10d}")
    print()


def multilabel_report(y_true, y_pred, threshold=0.5):
    """Report for multi-label task (ischemia)."""
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    # Per-subtype-per-lead metrics
    results = {}
    all_f1 = []
    for si, subtype in enumerate(_ISCHEMIA_SUBTYPES):
        for li, lead in enumerate(_ISCHEMIA_LEADS):
            idx = si * 12 + li
            t = y_true[:, idx]
            p = y_pred_bin[:, idx]
            tp = ((p == 1) & (t == 1)).sum()
            fp = ((p == 1) & (t == 0)).sum()
            fn = ((p == 0) & (t == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            pos = int(t.sum())
            if pos > 0:  # only report if there are positive samples
                results[f"{subtype}.{lead}"] = {
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1-score": round(f1, 4), "support": pos,
                }
                all_f1.append(f1)

    # Per-subtype aggregate
    subtype_summary = {}
    for si, subtype in enumerate(_ISCHEMIA_SUBTYPES):
        sub_f1s = []
        for li, lead in enumerate(_ISCHEMIA_LEADS):
            idx = si * 12 + li
            t = y_true[:, idx]
            p = y_pred_bin[:, idx]
            tp = ((p == 1) & (t == 1)).sum()
            fp = ((p == 1) & (t == 0)).sum()
            fn = ((p == 0) & (t == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            sub_f1s.append(f1)
        subtype_summary[subtype] = round(np.mean(sub_f1s), 4)

    overall_f1 = round(np.mean(all_f1), 4) if all_f1 else 0.0
    return results, subtype_summary, overall_f1


@torch.no_grad()
def evaluate(model, dataloader, vocab, device="cuda:0", input_mode=INPUT_MODE):
    """Run full evaluation and collect predictions."""
    # Collect all predictions and targets
    all_logits = defaultdict(list)
    all_labels = defaultdict(list)

    for batch in dataloader:
        signal = batch["signal"].to(device)
        image = batch["image"].to(device)
        labels = batch["labels"]

        out = model(signal, image, input_mode=input_mode)
        logits = out["logits"]

        for logit_key, (label_key, idx, task_type) in LABEL_MAP.items():
            if logit_key not in logits:
                continue
            all_logits[logit_key].append(logits[logit_key].cpu())
            all_labels[logit_key].append(labels[label_key])

    # Concatenate
    for k in all_logits:
        all_logits[k] = torch.cat(all_logits[k], dim=0)
        all_labels[k] = torch.cat(all_labels[k], dim=0)

    return all_logits, all_labels


def build_head_subtasks(vocab, data_version="v2"):
    """Same as train.py - version-aware."""
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
        base_config["ischemia_infarct"] = [
            ("st_elevation_territory", len(territory_vocab), "mc"),
            ("st_depression_territory", len(territory_vocab), "mc"),
            ("t_wave_abnormality", len(territory_vocab), "mc"),
            ("q_wave_territory", len(territory_vocab), "mc"),
        ]
    else:
        # v2: 48-dim multi-label
        base_config["ischemia_infarct"] = [("findings", 48, "ml")]

    return base_config


def main():
    parser = argparse.ArgumentParser(
        description="ECG Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set (default, recommended for final reporting)
  python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

  # Evaluate on val set (useful for comparing with training logs)
  python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split val

  # Evaluate on train set (check for overfitting)
  python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split train
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (auto-detected from checkpoint path if not set)")
    parser.add_argument("--split-indices", type=str, default=None,
                        help="Optional split_indices.json saved by training")
    parser.add_argument("--input-mode", type=str, default=INPUT_MODE,
                        choices=["dual", "signal", "image"],
                        help="多模态输入模式：dual=信号+图像，signal=只用信号，image=只用图像")
    parser.add_argument("--fusion-type", type=str, default=FUSION_TYPE,
                        choices=["cross_attention", "gated_cross_attention", "late_concat"],
                        help="融合方式，需要和训练该 checkpoint 时的结构保持一致")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="验证集校准得到的 thresholds_val.json；仅作用于二分类任务")
    parser.add_argument("--logit-biases", type=str, default=None,
                        help="验证集校准得到的多分类 logit bias JSON；作用于 argmax/threshold 前")
    args = parser.parse_args()

    # Auto-detect output directory from checkpoint path
    if args.output_dir is None:
        # checkpoint: outputs/ecg_diag/<timestamp>/checkpoints/best.pt
        # output:     outputs/ecg_diag/<timestamp>/evaluation_<split>/
        ckpt_dir = os.path.dirname(args.checkpoint)  # .../checkpoints
        run_dir = os.path.dirname(ckpt_dir)           # .../<timestamp>
        args.output_dir = os.path.join(run_dir, f"evaluation_{args.split}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load full dataset
    print("Loading dataset...")
    full_ds = ECGMultiModalDataset(
        jsonl_path=JSONL_PATH,
        image_root=IMAGE_ROOT,
        is_train=False,
        load_signal=args.input_mode in {"dual", "signal"},
        load_image=args.input_mode in {"dual", "image"},
        use_signal_augmentation=False,
    )
    head_subtasks = build_head_subtasks(full_ds.vocab, DATA_VERSION)
    vocab = full_ds.vocab

    total_size = len(full_ds)
    if args.split_indices is None:
        run_split = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "split_indices.json")
        args.split_indices = run_split if os.path.isfile(run_split) else None

    if args.split_indices:
        split_indices = load_split_indices(args.split_indices, total_size)
        print(f"Using split indices: {args.split_indices}")
    else:
        split_indices = make_split_indices(total_size, SEED, TRAIN_RATIO, VAL_RATIO)
        print("Using deterministic split from config seed")

    n_train = len(split_indices["train"])
    n_val = len(split_indices["val"])
    n_test = len(split_indices["test"])

    train_ds = Subset(full_ds, split_indices["train"])
    val_ds = Subset(full_ds, split_indices["val"])
    test_ds = Subset(full_ds, split_indices["test"])

    # Select the requested split
    split_map = {
        "train": (train_ds, n_train),
        "val": (val_ds, n_val),
        "test": (test_ds, n_test),
    }
    eval_ds, n_samples = split_map[args.split]

    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"Evaluating on: {args.split} ({n_samples} samples)")

    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint,
        head_subtasks,
        args.device,
        fusion_type=args.fusion_type,
        input_mode=args.input_mode,
    )

    # Evaluate
    print("Running evaluation...")
    all_logits, all_labels = evaluate(model, eval_loader, vocab, args.device, input_mode=args.input_mode)
    calibrated_thresholds = load_thresholds(args.thresholds)
    calibrated_biases = load_logit_biases(args.logit_biases)
    if calibrated_thresholds:
        print(f"Using calibrated thresholds: {args.thresholds}")
    if calibrated_biases:
        print(f"Using calibrated logit biases: {args.logit_biases}")

    # Generate reports
    print(f"\n{'=' * 70}")
    print(f"  EVALUATION REPORT — {args.split.upper()} SET ({n_samples} samples)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'=' * 70}")

    results = {}
    all_macro_f1s = []

    # --- Multi-class tasks ---
    mc_tasks = [
        ("rhythm_rate.rate_level", "rate_level", "rhythm_rate"),
        ("rhythm_rate.rhythm", "rhythm", "rhythm_rate"),
        ("conduction_axis.axis", "axis", "conduction_axis"),
        ("conduction_axis.pr_status", "pr_status", "conduction_axis"),
        ("conduction_axis.qrs_width", "qrs_width", "conduction_axis"),
        ("conduction_axis.conduction_status", "conduction_status", "conduction_axis"),
        ("voltage.lvh", "lvh_binary", "voltage"),
        ("voltage.rvh", "rvh_binary", "voltage"),
        ("voltage.voltage", "voltage", "voltage"),
        ("qt_electrolytes.qt_status", "qt_status", "qt_electrolytes"),
        ("summary.is_abnormal", "is_abnormal", "summary"),
    ]
    if DATA_VERSION == "v4":
        mc_tasks.extend([
            ("ischemia_infarct.st_elevation_present", "binary", "ischemia_infarct"),
            ("ischemia_infarct.st_depression_present", "binary", "ischemia_infarct"),
            ("ischemia_infarct.t_wave_abnormal", "binary", "ischemia_infarct"),
            ("ischemia_infarct.q_wave_present", "binary", "ischemia_infarct"),
        ])

    for logit_key, vocab_key, group in mc_tasks:
        if logit_key not in all_logits:
            continue

        logits = all_logits[logit_key]
        labels = all_labels[logit_key]
        bias = calibrated_biases.get(logit_key)
        if bias is not None and bias.numel() == logits.size(-1):
            logits = logits + bias.view(1, -1)

        threshold = calibrated_thresholds.get(logit_key)
        if threshold is not None and logits.size(-1) == 2:
            prob_pos = torch.softmax(logits, dim=-1)[:, 1]
            pred = (prob_pos >= threshold).long().numpy()
        else:
            pred = logits.argmax(dim=-1).numpy()

        # Get class names
        if vocab_key in ("lvh_binary", "rvh_binary", "binary"):
            class_names = ["0", "1"]
        elif vocab_key == "is_abnormal":
            class_names = ["Normal", "Abnormal"]
        else:
            class_names = vocab.get(vocab_key, [str(i) for i in range(logits.size(-1))])

        # Get targets
        if labels.dim() > 1:
            idx_in_group = LABEL_MAP[logit_key][1]
            target = labels[:, idx_in_group].numpy()
        else:
            target = labels.numpy()

        num_classes = logits.size(-1)
        cm = confusion_matrix(target, pred, num_classes)
        report = classification_report(cm, class_names)

        print(f"\n{'─' * 70}")
        print(f"  [{group}] {logit_key}  (classes={num_classes})")
        print(f"{'─' * 70}")
        print_confusion_matrix(cm, class_names, title="Confusion Matrix:")
        print_report(report, class_names)

        acc = (pred == target).mean()
        macro_f1 = report["macro_avg"]["f1-score"]
        all_macro_f1s.append(macro_f1)

        results[logit_key] = {
            "accuracy": round(acc, 4),
            "macro_f1": macro_f1,
            "weighted_f1": report["weighted_avg"]["f1-score"],
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "threshold": threshold,
            "per_class": {n: report[n] for n in class_names},
        }

    # --- Multi-label task: ischemia ---
    isc_key = "ischemia_infarct.findings"
    if isc_key in all_logits:
        print(f"\n{'─' * 70}")
        print(f"  [ischemia_infarct] Multi-label (4 subtypes × 12 leads = 48 dims)")
        print(f"{'─' * 70}")

        probs = torch.sigmoid(all_logits[isc_key]).numpy()
        targets = all_labels[isc_key].numpy()
        det_report, subtype_summary, overall_f1 = multilabel_report(targets, probs)

        print(f"\n  Per-subtype average F1:")
        for subtype, f1 in subtype_summary.items():
            print(f"    {subtype:20s}  F1={f1:.4f}")
        print(f"\n  Overall micro-avg F1: {overall_f1:.4f}")

        # Count positive labels
        pos_count = targets.sum(axis=0)
        total_pos = int(pos_count.sum())
        total_cells = targets.size
        print(f"  Positive labels: {total_pos}/{total_cells} ({total_pos/total_cells*100:.1f}%)")

        all_macro_f1s.append(overall_f1)

        results[isc_key] = {
            "overall_f1": overall_f1,
            "subtype_f1": subtype_summary,
            "positive_ratio": round(total_pos / total_cells, 4),
        }

    # --- Overall summary ---
    overall_macro_f1 = sum(all_macro_f1s) / len(all_macro_f1s) if all_macro_f1s else 0.0
    print(f"\n{'=' * 70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Mean Macro-F1 (all tasks): {overall_macro_f1:.4f}")
    print(f"\n  {'Task':<40s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    print(f"  {'─' * 65}")
    for logit_key in LABEL_MAP:
        if logit_key not in results:
            continue
        r = results[logit_key]
        if "accuracy" in r:
            print(f"  {logit_key:<40s} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")
        elif "overall_f1" in r:
            print(f"  {logit_key:<40s} {'N/A':>10s} {r['overall_f1']:>10.4f}")
    print()

    # ---- Save all results ----
    save_dir = args.output_dir
    print(f"Saving results to {save_dir}")

    # Save confusion matrix images
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for logit_key in LABEL_MAP:
        if logit_key not in results or "confusion_matrix" not in results[logit_key]:
            continue
        r = results[logit_key]
        cm = np.array(r["confusion_matrix"])
        class_names = r["class_names"]

        fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names))))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label', xlabel='Predicted label',
               title=logit_key)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.tight_layout()
        fname = f"cm_{logit_key.replace('.', '_')}.png"
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Generate summary figure
    fig, ax = plt.subplots(figsize=(12, 6))
    task_names = []
    accs = []
    f1s = []
    for logit_key in LABEL_MAP:
        if logit_key not in results:
            continue
        r = results[logit_key]
        task_names.append(logit_key.replace('.', '\n.'))
        accs.append(r.get("accuracy", 0))
        f1s.append(r.get("macro_f1", r.get("overall_f1", 0)))

    x = np.arange(len(task_names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, accs, width, label='Accuracy', color='#4C72B0')
    bars2 = ax.bar(x + width / 2, f1s, width, label='Macro-F1', color='#DD8452')
    ax.set_ylabel('Score')
    ax.set_title(f'Evaluation Summary — {args.split.upper()} Set\n({os.path.basename(os.path.dirname(args.checkpoint))})')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Save JSON with metadata
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_results = {
        "split": args.split,
        "n_samples": n_samples,
        "overall_macro_f1": overall_macro_f1,
        "checkpoint": args.checkpoint,
        "input_mode": args.input_mode,
        "fusion_type": args.fusion_type,
        "thresholds": args.thresholds,
        "logit_biases": args.logit_biases,
        "results": results,
    }
    json_path = os.path.join(save_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, default=convert, indent=2, ensure_ascii=False)

    print(f"  Saved: {len([k for k in results if 'confusion_matrix' in results[k]])} confusion matrices")
    print(f"  Saved: summary.png")
    print(f"  Saved: eval_results.json")
    print(f"\nOverall Macro-F1 on {args.split} set: {overall_macro_f1:.4f}")


if __name__ == "__main__":
    main()
