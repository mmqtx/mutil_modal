"""
验证集 logit bias 校准脚本。

用途：
  对多分类任务在验证集上搜索每个类别的加性 logit bias，用于改善少数类
  macro-F1。该脚本不训练模型、不修改数据，只生成轻量 JSON，供 evaluate.py
  的 --logit-biases 使用。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    BATCH_SIZE,
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
    evaluate,
    load_model,
    load_split_indices,
    make_split_indices,
)


def macro_f1(target, pred, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        scores.append(2 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(scores))


def target_for_task(labels, task_name):
    label_key, idx, _ = LABEL_MAP[task_name]
    task_labels = labels[task_name]
    if task_labels.dim() > 1:
        return task_labels[:, idx].numpy().astype(np.int64)
    return task_labels.numpy().astype(np.int64)


def search_bias(logits, target, grid, rounds):
    logits = logits.numpy()
    num_classes = logits.shape[1]
    bias = np.zeros(num_classes, dtype=np.float32)
    pred = logits.argmax(axis=1)
    best = macro_f1(target, pred, num_classes)

    for _ in range(rounds):
        improved = False
        for cls in range(num_classes):
            cls_best = best
            cls_value = bias[cls]
            for value in grid:
                candidate = bias.copy()
                candidate[cls] = value
                candidate -= candidate.mean()
                pred = (logits + candidate.reshape(1, -1)).argmax(axis=1)
                score = macro_f1(target, pred, num_classes)
                if score > cls_best:
                    cls_best = score
                    cls_value = candidate[cls]
                    best_bias = candidate
            if cls_best > best:
                best = cls_best
                bias = best_bias
                improved = True
            else:
                bias[cls] = cls_value
        if not improved:
            break
    return bias, best


def main():
    parser = argparse.ArgumentParser(description="在验证集上校准多分类 logit bias")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--split-indices", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--input-mode", type=str, default=INPUT_MODE,
                        choices=["dual", "signal", "image"])
    parser.add_argument("--fusion-type", type=str, default=FUSION_TYPE,
                        choices=["cross_attention", "gated_cross_attention", "late_concat"])
    parser.add_argument("--bias-min", type=float, default=-1.0)
    parser.add_argument("--bias-max", type=float, default=1.0)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    full_ds = ECGMultiModalDataset(
        JSONL_PATH,
        IMAGE_ROOT,
        is_train=False,
        load_signal=args.input_mode in {"dual", "signal"},
        load_image=args.input_mode in {"dual", "image"},
        use_signal_augmentation=False,
    )
    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    split_path = args.split_indices or os.path.join(run_dir, "split_indices.json")
    if os.path.isfile(split_path):
        split_indices = load_split_indices(split_path, len(full_ds))
        print(f"使用 split 索引: {split_path}")
    else:
        split_indices = make_split_indices(len(full_ds), SEED, TRAIN_RATIO, VAL_RATIO)
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
    all_logits, all_labels = evaluate(model, val_loader, full_ds.vocab, args.device, args.input_mode)
    grid = np.round(np.arange(args.bias_min, args.bias_max + 1e-8, args.bias_step), 4)

    results = {}
    for task_name, logits in all_logits.items():
        if logits.size(-1) <= 2:
            continue
        target = target_for_task(all_labels, task_name)
        default_pred = logits.numpy().argmax(axis=1)
        default_f1 = macro_f1(target, default_pred, logits.size(-1))
        bias, best_f1 = search_bias(logits, target, grid, args.rounds)
        results[task_name] = {
            "default_macro_f1": round(default_f1, 6),
            "best_macro_f1": round(best_f1, 6),
            "bias": [round(float(x), 4) for x in bias.tolist()],
        }

    output_path = args.output or os.path.join(run_dir, "logit_biases_val.json")
    payload = {
        "checkpoint": args.checkpoint,
        "split": "val",
        "n_samples": len(val_ds),
        "input_mode": args.input_mode,
        "fusion_type": args.fusion_type,
        "logit_biases": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"已保存 logit bias 校准结果: {output_path}")
    for task_name, item in results.items():
        print(f"{task_name}: {item['default_macro_f1']:.4f} -> {item['best_macro_f1']:.4f}, bias={item['bias']}")


if __name__ == "__main__":
    main()
