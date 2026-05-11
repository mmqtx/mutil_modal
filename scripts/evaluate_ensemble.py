"""
双模型概率集成评估脚本。

用途：
  固定数据和 checkpoint，不重新训练模型。在验证集上为每个任务搜索
  dual 模型与 signal-only 模型的概率融合权重；二分类任务同时搜索阈值。
  然后在测试集上应用该校准配置，验证是否能缓解类别分布不均造成的少数类问题。
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
    IMAGE_ROOT,
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
    classification_report,
    confusion_matrix,
    load_model,
    load_split_indices,
    make_split_indices,
)


TASKS = list(LABEL_MAP.keys())


def macro_f1(target, pred, num_classes):
    cm = confusion_matrix(target, pred, num_classes)
    class_names = [str(i) for i in range(num_classes)]
    return classification_report(cm, class_names)["macro_avg"]["f1-score"]


def collect_probs(models, loader, device):
    probs = {name: {task: [] for task in TASKS} for name in models}
    targets = {task: [] for task in TASKS}

    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            image = batch["image"].to(device)
            labels = batch["labels"]

            for name, spec in models.items():
                out = spec["model"](signal, image, input_mode=spec["input_mode"])
                for task in TASKS:
                    if task in out["logits"]:
                        probs[name][task].append(torch.softmax(out["logits"][task], dim=-1).cpu())

            for task, (group, idx, _) in LABEL_MAP.items():
                label_tensor = labels[group]
                if label_tensor.dim() > 1:
                    targets[task].append(label_tensor[:, idx].cpu())
                else:
                    targets[task].append(label_tensor.cpu())

    out_probs = {
        name: {task: torch.cat(chunks).numpy() for task, chunks in task_probs.items() if chunks}
        for name, task_probs in probs.items()
    }
    out_targets = {task: torch.cat(chunks).numpy().astype(np.int64) for task, chunks in targets.items() if chunks}
    return out_probs, out_targets


def calibrate(probs_a, probs_b, targets):
    weights = np.round(np.arange(0.0, 1.001, 0.05), 2)
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    config = {}

    for task, target in targets.items():
        if task not in probs_a or task not in probs_b:
            continue
        num_classes = probs_a[task].shape[1]
        best = {"weight_a": 1.0, "macro_f1": -1.0, "threshold": None}

        for weight in weights:
            prob = weight * probs_a[task] + (1.0 - weight) * probs_b[task]
            if num_classes == 2:
                for threshold in thresholds:
                    pred = (prob[:, 1] >= threshold).astype(np.int64)
                    score = macro_f1(target, pred, num_classes)
                    if score > best["macro_f1"]:
                        best = {"weight_a": float(weight), "macro_f1": score, "threshold": float(threshold)}
            else:
                pred = prob.argmax(axis=1)
                score = macro_f1(target, pred, num_classes)
                if score > best["macro_f1"]:
                    best = {"weight_a": float(weight), "macro_f1": score, "threshold": None}
        config[task] = best
    return config


def evaluate_ensemble(probs_a, probs_b, targets, config):
    results = {}
    macro_scores = []
    for task, item in config.items():
        if task not in targets:
            continue
        prob = item["weight_a"] * probs_a[task] + (1.0 - item["weight_a"]) * probs_b[task]
        num_classes = prob.shape[1]
        if item["threshold"] is not None and num_classes == 2:
            pred = (prob[:, 1] >= item["threshold"]).astype(np.int64)
        else:
            pred = prob.argmax(axis=1)
        target = targets[task]
        cm = confusion_matrix(target, pred, num_classes)
        class_names = [str(i) for i in range(num_classes)]
        report = classification_report(cm, class_names)
        macro = report["macro_avg"]["f1-score"]
        macro_scores.append(macro)
        results[task] = {
            "accuracy": round(float((pred == target).mean()), 4),
            "macro_f1": macro,
            "weight_a": item["weight_a"],
            "threshold": item["threshold"],
            "confusion_matrix": cm.tolist(),
            "per_class": {name: report[name] for name in class_names},
        }
    return float(np.mean(macro_scores)), results


def main():
    parser = argparse.ArgumentParser(description="验证集校准并测试 dual/signal 概率集成")
    parser.add_argument("--dual-checkpoint", required=True)
    parser.add_argument("--signal-checkpoint", required=True)
    parser.add_argument("--split-indices", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    full_ds = ECGMultiModalDataset(
        JSONL_PATH,
        IMAGE_ROOT,
        is_train=False,
        load_signal=True,
        load_image=True,
        use_signal_augmentation=False,
    )
    split_path = args.split_indices or os.path.join(os.path.dirname(os.path.dirname(args.dual_checkpoint)), "split_indices.json")
    if os.path.isfile(split_path):
        split_indices = load_split_indices(split_path, len(full_ds))
    else:
        split_indices = make_split_indices(len(full_ds), SEED, TRAIN_RATIO, VAL_RATIO)

    loaders = {
        split: DataLoader(
            Subset(full_ds, split_indices[split]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        for split in ["val", "test"]
    }
    head_subtasks = build_head_subtasks(full_ds.vocab, DATA_VERSION)
    models = {
        "dual": {
            "model": load_model(args.dual_checkpoint, head_subtasks, args.device, "cross_attention", "dual"),
            "input_mode": "dual",
        },
        "signal": {
            "model": load_model(args.signal_checkpoint, head_subtasks, args.device, "cross_attention", "signal"),
            "input_mode": "signal",
        },
    }

    val_probs, val_targets = collect_probs(models, loaders["val"], args.device)
    config = calibrate(val_probs["dual"], val_probs["signal"], val_targets)
    val_overall, val_results = evaluate_ensemble(val_probs["dual"], val_probs["signal"], val_targets, config)

    test_probs, test_targets = collect_probs(models, loaders["test"], args.device)
    test_overall, test_results = evaluate_ensemble(test_probs["dual"], test_probs["signal"], test_targets, config)

    payload = {
        "dual_checkpoint": args.dual_checkpoint,
        "signal_checkpoint": args.signal_checkpoint,
        "split_indices": split_path,
        "val_overall_macro_f1": round(val_overall, 6),
        "test_overall_macro_f1": round(test_overall, 6),
        "ensemble_config": config,
        "val_results": val_results,
        "test_results": test_results,
    }
    output_path = os.path.join(args.output_dir, "ensemble_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"已保存 ensemble 结果: {output_path}")
    print(f"Val Macro-F1:  {val_overall:.4f}")
    print(f"Test Macro-F1: {test_overall:.4f}")
    for task, result in sorted(test_results.items(), key=lambda kv: kv[1]["macro_f1"])[:8]:
        print(f"{task}: macro={result['macro_f1']:.4f}, w_dual={result['weight_a']}, thr={result['threshold']}")


if __name__ == "__main__":
    main()
