"""
单样本 / 批量推理脚本 — 输出结构化 ECG 诊断报告 + Qwen 生成说明。

Usage:
  python predict.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --signal-path /path/to/patient.dat --image-path /path/to/patient.png
  python predict.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --jsonl-path /data/ljq24358/ecg_dataset/ecg_jsons/structured_extraction/structured_labels_v2.jsonl --num-samples 10
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime

import numpy as np
import requests
import torch
from PIL import Image

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import *
from config import DATA_VERSION, get_label_config, TRAIN_RATIO, VAL_RATIO
from data.dataset import ECGMultiModalDataset
from data.transforms import get_signal_transform, get_image_transform
from models.model import ECGDiagModel, load_gem_ecg_pretrained

import wfdb


# ---------------------------------------------------------------------------
# 配置区域 (可在此处修改常用参数)
# ---------------------------------------------------------------------------

# ========== Qwen API 配置 ==========
# 请通过环境变量配置 DashScope API Key，避免把密钥提交到 Git。
# export DASHSCOPE_API_KEY="sk-..."
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen-plus"  # 模型名称: qwen-plus, qwen-turbo, qwen-max 等

# ========== 预测配置 ==========
DEFAULT_CHECKPOINT = "outputs/ecg_diag/20260420_155424/checkpoints/best.pt"  # 默认模型路径
DEFAULT_NUM_SAMPLES = 5  # 默认预测样本数量
DEFAULT_DEVICE = "cuda:0"  # 默认设备
DEFAULT_ENABLE_QWEN = False  # 默认不调用外部API；需要时通过 --enable-qwen 开启
DEFAULT_SPLIT = "test"  # 默认数据集划分: train, val, test
OUTPUT_DIR = "outputs/predictions"  # 结果保存目录

# ============================================================
# 以下为内部配置，通常无需修改
# ============================================================

QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_qwen_with_image(structured_report: dict, image_base64: str,
                         api_key: str = None) -> str:
    """
    调用 Qwen API 生成诊断说明。

    Args:
        structured_report: 模型预测的结构化诊断结果
        image_base64: ECG图片的base64编码
        api_key: DashScope API Key

    Returns:
        生成的诊断说明文本
    """
    if api_key is None:
        api_key = QWEN_API_KEY

    if not api_key:
        return "[错误] 未设置 DASHSCOPE_API_KEY 环境变量"

    # 构建提示词
    prompt = build_diagnosis_prompt(structured_report)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    }

    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"[错误] Qwen API 调用失败: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"[错误] Qwen API 响应解析失败: {str(e)}"


def build_diagnosis_prompt(report: dict) -> str:
    """
    根据结构化报告构建 Qwen 提示词。

    Args:
        report: 模型预测的结构化诊断结果

    Returns:
        提示词字符串
    """
    # 解析报告内容
    rr = report.get("rhythm_rate", {})
    ca = report.get("conduction_axis", {})
    volt = report.get("voltage", {})
    qt = report.get("qt_electrolytes", {})
    summary = report.get("summary", {})

    # 构建文本描述
    lines = []
    lines.append("请根据以下心电图AI结构化诊断结果，生成一段简洁的中文诊断说明（200字以内）：")
    lines.append("\n【AI诊断结果】")

    # 心率与心律
    rate = rr.get("rate_level", {}).get("value", "未知")
    rhythm = rr.get("rhythm", {}).get("value", "未知")
    lines.append(f"- 心率水平: {rate} (置信度: {rr.get('rate_level', {}).get('confidence', 0):.2f})")
    lines.append(f"- 心律: {rhythm} (置信度: {rr.get('rhythm', {}).get('confidence', 0):.2f})")

    # 传导与电轴
    axis = ca.get("axis", {}).get("value", "未知")
    pr = ca.get("pr_status", {}).get("value", "未知")
    qrs = ca.get("qrs_width", {}).get("value", "未知")
    cond = ca.get("conduction_status", {}).get("value", "未知")
    lines.append(f"- 电轴: {axis}")
    lines.append(f"- PR间期: {pr}, QRS宽度: {qrs}, 传导状态: {cond}")

    # 电压与肥大
    voltage = volt.get("voltage", {}).get("value", "未知")
    lvh = volt.get("lvh", {}).get("value", 0)
    rvh = volt.get("rvh", {}).get("value", 0)
    lines.append(f"- 电压: {voltage}, 左室肥大: {'是' if lvh else '否'}, 右室肥大: {'是' if rvh else '否'}")

    # QT间期
    qt_status = qt.get("value", "未知")
    lines.append(f"- QT间期: {qt_status}")

    # 总体评估
    is_abnormal = summary.get("is_abnormal", False)
    conf = summary.get("confidence", 0)
    lines.append(f"- 总体评估: {'异常' if is_abnormal else '正常'} (置信度: {conf:.2f})")

    lines.append("\n请生成一段专业、简洁的中文诊断说明，包括主要发现和临床建议（200字以内）。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Label decode: index → human-readable text  / 标签解码：索引 → 可读文本
# ---------------------------------------------------------------------------

def build_inverse_vocab(vocab: dict) -> dict:
    """Build index→text mapping for each vocab category."""
    inv = {}
    for key, values in vocab.items():
        inv[key] = {i: v for i, v in enumerate(values)}
    return inv


# Ischemia: 4 subtypes × 12 leads → 48-dim  / 缺血：4亚型 × 12导联 → 48维
_ISCHEMIA_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
_ISCHEMIA_SUBTYPES = ["st_elevation", "st_depression", "t_wave_alt", "q_wave"]


def decode_ischemia_v2(findings_48: np.ndarray, threshold: float = 0.5):
    """Decode 48-dim sigmoid outputs to lead-level findings (v2)."""
    results = {}
    for si, subtype in enumerate(_ISCHEMIA_SUBTYPES):
        for li, lead in enumerate(_ISCHEMIA_LEADS):
            prob = findings_48[si * 12 + li]
            if prob > threshold:
                if subtype not in results:
                    results[subtype] = []
                results[subtype].append({"lead": lead, "prob": round(float(prob), 3)})
    return results


def decode_ischemia_v4(logits_dict: dict, threshold: float = 0.5) -> dict:
    """Decode v4 binary ischemia predictions."""
    results = {}
    label_map = {
        "ischemia_infarct.st_elevation_present": "st_elevation",
        "ischemia_infarct.st_depression_present": "st_depression",
        "ischemia_infarct.t_wave_abnormal": "t_wave_abnormal",
        "ischemia_infarct.q_wave_present": "q_wave",
    }

    for key, subtype in label_map.items():
        if key in logits_dict:
            v = logits_dict[key]
            idx = v.argmax().item()
            probs = torch.softmax(v, dim=-1)
            is_present = idx == 1  # Binary: 1 = present
            if is_present:
                results[subtype] = {
                    "present": True,
                    "confidence": round(probs[idx].item(), 3),
                }

    return results


def decode_prediction(logits: dict, inv_vocab: dict, data_version: str = "v2") -> dict:
    """Convert raw logits to structured diagnostic report."""
    # Squeeze batch dim: (1, C) → (C,)
    logits = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in logits.items()}
    report = {}

    # Helper: decode a classification logit
    def _decode_cls(key, vocab_key):
        if key not in logits:
            return None
        v = logits[key]
        idx = v.argmax().item()
        probs = torch.softmax(v, dim=-1)
        label = inv_vocab.get(vocab_key, {}).get(idx, f"class_{idx}")
        return {"value": label, "confidence": round(probs[idx].item(), 3)}

    # --- Rhythm & Rate ---
    report["rhythm_rate"] = {
        k: v for k, v in [
            ("rate_level", _decode_cls("rhythm_rate.rate_level", "rate_level")),
            ("rhythm", _decode_cls("rhythm_rate.rhythm", "rhythm")),
        ] if v is not None
    }

    # --- Conduction & Axis ---
    report["conduction_axis"] = {
        k: v for k, v in [
            ("axis", _decode_cls("conduction_axis.axis", "axis")),
            ("pr_status", _decode_cls("conduction_axis.pr_status", "pr_status")),
            ("qrs_width", _decode_cls("conduction_axis.qrs_width", "qrs_width")),
            ("conduction_status", _decode_cls("conduction_axis.conduction_status", "conduction_status")),
        ] if v is not None
    }

    # --- Voltage & Hypertrophy ---
    voltage = {}
    for sub in ["lvh", "rvh"]:
        key = f"voltage.{sub}"
        if key in logits:
            v = logits[key]
            idx = v.argmax().item()
            probs = torch.softmax(v, dim=-1)
            voltage[sub] = {"value": idx, "confidence": round(probs[idx].item(), 3)}
    v_decoded = _decode_cls("voltage.voltage", "voltage")
    if v_decoded:
        voltage["voltage"] = v_decoded
    report["voltage"] = voltage

    # --- Ischemia / Infarct ---
    if data_version == "v4":
        # v4: 4 binary classifications
        isc_results = decode_ischemia_v4(logits, threshold=0.5)
        report["ischemia_infarct"] = {
            "has_finding": len(isc_results) > 0,
            "details": isc_results,
            "num_positive": len(isc_results),
        }
    elif data_version == "v2":
        # v2: 48-dim multi-label
        isc_key = "ischemia_infarct.findings"
        if isc_key in logits:
            probs = torch.sigmoid(logits[isc_key]).detach().cpu().numpy()
            findings = decode_ischemia_v2(probs, threshold=0.5)
            report["ischemia_infarct"] = {
                "has_finding": len(findings) > 0,
                "details": findings,
                "num_positive": sum(len(v) for v in findings.values()),
            }
    else:
        # v3: Handle territory-level predictions (simplified)
        report["ischemia_infarct"] = {"note": "v3 format - to be implemented"}

    # --- QT / Electrolytes ---
    qt_decoded = _decode_cls("qt_electrolytes.qt_status", "qt_status")
    if qt_decoded:
        report["qt_electrolytes"] = qt_decoded

    # --- Summary ---
    sum_key = "summary.is_abnormal"
    if sum_key in logits:
        v = logits[sum_key]
        idx = v.argmax().item()
        probs = torch.softmax(v, dim=-1)
        report["summary"] = {
            "is_abnormal": bool(idx),
            "confidence": round(probs[idx].item(), 3),
        }

    return report


# ---------------------------------------------------------------------------
# Model loading  / 模型加载
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load trained model from checkpoint."""
    # First build a dataset to get vocab
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load a sample dataset to get vocab and head_subtasks
    temp_ds = ECGMultiModalDataset(JSONL_PATH, IMAGE_ROOT, is_train=True, max_samples=100)

    # Import the build_head_subtasks function (shared with train.py)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.train import build_head_subtasks

    head_subtasks = build_head_subtasks(temp_ds.vocab, DATA_VERSION)

    model = ECGDiagModel(
        embed_dim=EMBED_DIM, clip_model_path=CLIP_VIT_PATH,
        fusion_dim=FUSION_DIM, fusion_heads=NUM_HEADS,
        fusion_num_layers=FUSION_NUM_LAYERS,
        head_subtasks=head_subtasks,
        chain_attn_heads=NUM_HEADS, chain_attn_layers=NUM_CHAIN_LAYERS,
        uplift_hidden_dim=UPLIFT_HIDDEN_DIM, uplift_num_layers=UPLIFT_NUM_LAYERS,
        contrastive_hidden_dim=CONTRASTIVE_HIDDEN_DIM,
        contrastive_out_dim=CONTRASTIVE_OUT_DIM,
        contrastive_num_layers=CONTRASTIVE_NUM_LAYERS,
    )

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


# ---------------------------------------------------------------------------
# Inference  / 推理
# ---------------------------------------------------------------------------

def predict_single(model, signal_tensor: torch.Tensor, image_tensor: torch.Tensor,
                   inv_vocab: dict, device: str = "cuda:0") -> dict:
    """Run inference on a single sample."""
    signal = signal_tensor.unsqueeze(0).to(device)
    image = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(signal, image)

    return decode_prediction(out["logits"], inv_vocab, DATA_VERSION)


def main():
    parser = argparse.ArgumentParser(description="ECG Structured Diagnostic Report + Qwen 说明")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Path to trained model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help=f"Device to use (default: {DEFAULT_DEVICE})")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"Number of samples to predict (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["train", "val", "test"],
                        help=f"Dataset split to predict: train/val/test (default: {DEFAULT_SPLIT})")
    parser.add_argument("--jsonl-path", type=str, default=JSONL_PATH)
    parser.add_argument("--image-root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--enable-qwen", action="store_true", default=DEFAULT_ENABLE_QWEN,
                        help="Enable Qwen to generate diagnostic description")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DashScope API Key (overrides script config and env var)")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable saving results to JSON file")
    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Build vocab from FULL dataset (not limited by num_samples)
    full_ds = ECGMultiModalDataset(
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        is_train=True,   # use full training set for complete vocab
    )
    inv_vocab = build_inverse_vocab(full_ds.vocab)

    # Calculate split indices (same as training: 70% train, 10% val, 20% test)
    total_size = len(full_ds)
    n_train = int(total_size * TRAIN_RATIO)
    n_val = int(total_size * VAL_RATIO)
    n_test = total_size - n_train - n_val

    # Get indices for the requested split
    if args.split == "train":
        start_idx = 0
        end_idx = n_train
    elif args.split == "val":
        start_idx = n_train
        end_idx = n_train + n_val
    else:  # test
        start_idx = n_train + n_val
        end_idx = total_size

    # Limit to num_samples
    actual_samples = min(args.num_samples, end_idx - start_idx)
    end_idx = start_idx + actual_samples

    print(f"\n{'='*60}")
    print(f"Dataset split: {args.split} (indices {start_idx}-{end_idx}, total {actual_samples} samples)")
    print(f"  Full dataset: {total_size} samples")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"{'='*60}\n")

    # Create output directory
    if not args.no_save:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"{args.split}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}\n")

    # Store all results for saving
    all_results = []

    print(f"\n{'='*60}")
    print(f"ECG Structured Diagnostic Report — {actual_samples} samples")
    print(f"{'='*60}\n")

    # Check Qwen configuration
    api_key = args.api_key or QWEN_API_KEY
    if args.enable_qwen and not api_key:
        print("⚠️  警告: 未设置 DASHSCOPE_API_KEY，将跳过 Qwen 生成")
        print("   请设置环境变量: export DASHSCOPE_API_KEY=your_key_here")
        print("   或使用 --api-key 参数\n")

    # Predict on the selected split
    for idx in range(start_idx, end_idx):
        sample = full_ds[idx]
        report = predict_single(
            model, sample["signal"], sample["image"], inv_vocab, args.device
        )

        # Prepare result entry
        result_entry = {
            "index": idx,
            "sample_id": sample["id"],
            "split": args.split,
            "prediction": report,
        }

        print(f"--- Sample {idx-start_idx+1}/{actual_samples} (id={sample['id']}, global_idx={idx}) ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        # Show ground truth for comparison
        labels = sample["labels"]
        gt = {}
        if labels["rhythm_rate"].dim() > 0:
            gt["rate_level"] = full_ds.vocab["rate_level"][labels["rhythm_rate"][0]]
            gt["rhythm"] = full_ds.vocab["rhythm"][labels["rhythm_rate"][1]]
        gt["is_abnormal"] = bool(labels["summary"][0])
        result_entry["ground_truth"] = gt
        print(f"  [Ground Truth] {json.dumps(gt, ensure_ascii=False)}")

        # Qwen 生成诊断说明
        if args.enable_qwen and api_key:
            # Get image path from dataset records
            img_rel_path = full_ds.records[idx]["image_paths"][0]
            img_full_path = os.path.join(args.image_root, img_rel_path)

            if os.path.exists(img_full_path):
                print(f"  [Qwen] 正在生成诊断说明...")
                img_base64 = encode_image_to_base64(img_full_path)
                qwen_response = call_qwen_with_image(report, img_base64, api_key)
                print(f"  [Qwen 诊断说明]\n{qwen_response}")
                result_entry["qwen_description"] = qwen_response
            else:
                print(f"  [Qwen] 图片不存在: {img_full_path}")
                result_entry["qwen_description"] = None

        all_results.append(result_entry)
        print()

    # Save results to JSON
    if not args.no_save and all_results:
        output_file = os.path.join(output_dir, "predictions.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "num_samples": len(all_results),
                    "timestamp": timestamp,
                    "qwen_enabled": args.enable_qwen,
                },
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
