"""
验证集阈值校准脚本。

用途：
  对 v4 中的二分类任务，在验证集上搜索 0.05~0.95 的正类概率阈值，
  以 macro-F1 或 positive-F1 为目标保存推荐阈值。这个脚本不改动训练数据，
  只生成轻量 JSON，供后续预测或报告生成阶段选择使用。

示例：
  python scripts/calibrate_thresholds.py \
      --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    BATCH_SIZE,
    CLIP_VIT_PATH,
    DATA_VERSION,
    FUSION_TYPE,
    IMAGE_ROOT,
    INPUT_MODE,
    JSONL_PATH,
    NUM_WORKERS,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)
from data.dataset import ECGMultiModalDataset  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    LABEL_MAP,
    build_head_subtasks,
    load_model,
    load_split_indices,
    make_split_indices,
)


def _f1_binary(target, pred):
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def _macro_f1_binary(target, pred):
    return (_f1_binary(target, pred) + _f1_binary(1 - target, 1 - pred)) / 2


@torch.no_grad()
def collect_binary_probs(model, dataloader, device, input_mode):
    probs = defaultdict(list)
    targets = defaultdict(list)

    for batch in dataloader:
        signal = batch["signal"].to(device)
        image = batch["image"].to(device)
        labels = batch["labels"]
        out = model(signal, image, input_mode=input_mode)

        for task_name, (label_group, label_idx, task_type) in LABEL_MAP.items():
            if task_type != "mc" or task_name not in out["logits"]:
                continue
            logits = out["logits"][task_name]
            if logits.size(-1) != 2:
                continue

            prob_pos = torch.softmax(logits, dim=-1)[:, 1].cpu()
            label_tensor = labels[label_group]
            if label_tensor.dim() > 1:
                target = label_tensor[:, label_idx]
            else:
                target = label_tensor

            probs[task_name].append(prob_pos)
            targets[task_name].append(target.cpu().long())

    return {
        task_name: {
            "prob": torch.cat(task_probs).numpy(),
            "target": torch.cat(targets[task_name]).numpy(),
        }
        for task_name, task_probs in probs.items()
    }


def calibrate(collected, objective):
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    results = {}

    for task_name, values in collected.items():
        prob = values["prob"]
        target = values["target"]
        positive = int(target.sum())
        total = int(target.shape[0])

        best = {
            "threshold": 0.5,
            "macro_f1": 0.0,
            "positive_f1": 0.0,
        }
        for threshold in thresholds:
            pred = (prob >= threshold).astype(np.int64)
            macro_f1 = float(_macro_f1_binary(target, pred))
            positive_f1 = float(_f1_binary(target, pred))
            score = macro_f1 if objective == "macro_f1" else positive_f1
            best_score = best[objective]
            if score > best_score:
                best = {
                    "threshold": float(threshold),
                    "macro_f1": round(macro_f1, 6),
                    "positive_f1": round(positive_f1, 6),
                }

        default_pred = (prob >= 0.5).astype(np.int64)
        results[task_name] = {
            "positive": positive,
            "negative": total - positive,
            "positive_ratio": round(positive / max(total, 1), 6),
            "default_0_5": {
                "macro_f1": round(float(_macro_f1_binary(target, default_pred)), 6),
                "positive_f1": round(float(_f1_binary(target, default_pred)), 6),
            },
            "best": best,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="在验证集上校准 v4 二分类任务阈值")
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--split-indices", type=str, default=None,
                        help="训练 run 保存的 split_indices.json；默认自动从 checkpoint 目录查找")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径；默认保存到 run 目录 thresholds_val.json")
    parser.add_argument("--objective", type=str, default="macro_f1",
                        choices=["macro_f1", "positive_f1"],
                        help="阈值搜索目标，macro_f1 更平衡，positive_f1 更偏向少数阳性类")
    parser.add_argument("--input-mode", type=str, default=INPUT_MODE,
                        choices=["dual", "signal", "image"])
    parser.add_argument("--fusion-type", type=str, default=FUSION_TYPE,
                        choices=["cross_attention", "late_concat"],
                        help="必须与训练 checkpoint 的模型结构一致")
    args = parser.parse_args()

    full_ds = ECGMultiModalDataset(
        JSONL_PATH,
        IMAGE_ROOT,
        is_train=False,
        load_signal=args.input_mode in {"dual", "signal"},
        load_image=args.input_mode in {"dual", "image"},
        use_signal_augmentation=False,
    )
    total_size = len(full_ds)
    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))

    split_path = args.split_indices
    if split_path is None:
        candidate = os.path.join(run_dir, "split_indices.json")
        split_path = candidate if os.path.isfile(candidate) else None

    if split_path:
        split_indices = load_split_indices(split_path, total_size)
        print(f"使用 split 索引: {split_path}")
    else:
        split_indices = make_split_indices(total_size, SEED, TRAIN_RATIO, VAL_RATIO)
        print("未找到 split_indices.json，使用 config seed 重新生成确定性划分")

    val_ds = Subset(full_ds, split_indices["val"])
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    head_subtasks = build_head_subtasks(full_ds.vocab, DATA_VERSION)
    model = load_model(
        args.checkpoint,
        head_subtasks,
        args.device,
        fusion_type=args.fusion_type,
        input_mode=args.input_mode,
    )

    collected = collect_binary_probs(model, val_loader, args.device, args.input_mode)
    results = calibrate(collected, args.objective)

    output_path = args.output or os.path.join(run_dir, "thresholds_val.json")
    payload = {
        "checkpoint": args.checkpoint,
        "split": "val",
        "n_samples": len(val_ds),
        "data_version": DATA_VERSION,
        "clip_path": CLIP_VIT_PATH,
        "input_mode": args.input_mode,
        "fusion_type": args.fusion_type,
        "objective": args.objective,
        "thresholds": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"已保存阈值校准结果: {output_path}")
    for task_name, result in results.items():
        best = result["best"]
        print(
            f"{task_name}: threshold={best['threshold']:.2f}, "
            f"macro_f1={best['macro_f1']:.4f}, positive_f1={best['positive_f1']:.4f}, "
            f"pos={result['positive']}/{result['positive'] + result['negative']}"
        )


if __name__ == "__main__":
    main()
