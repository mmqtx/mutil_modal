"""
Validate Qwen-extracted structured ECG labels.

Checks:
  - required JSON fields
  - label values against config.LABEL_CONFIGS
  - signal/image file existence
  - class distributions
  - availability of original_cot and summary primary_label

Usage:
  python scripts/validate_structured_labels.py
  python scripts/validate_structured_labels.py --jsonl /path/to/structured_labels_v4.jsonl --limit 1000
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_VERSION, IMAGE_ROOT, JSONL_PATH, get_label_config


def strip_parens(label):
    import re
    return re.sub(r"\s*\(.*?\)\s*", "", str(label)).strip()


def allowed_values(label_config):
    values = {
        "rate_level": {strip_parens(v) for v in label_config["rhythm_rate"]["rate_level"]["classes"]},
        "rhythm": set(label_config["rhythm_rate"]["rhythm"]["classes"]),
        "axis": set(label_config["conduction_axis"]["axis"]["classes"]),
        "pr_status": set(label_config["conduction_axis"]["pr_status"]["classes"]),
        "qrs_width": {strip_parens(v) for v in label_config["conduction_axis"]["qrs_width"]["classes"]},
        "conduction_status": set(label_config["conduction_axis"]["conduction_status"]["classes"]),
        "voltage": set(label_config["voltage"]["voltage"]["classes"]),
        "qt_status": set(label_config["qt_electrolytes"]["qt_status"]["classes"]),
    }
    return values


def check_value(counter, errors, key, value, allowed, line_no):
    normalized = strip_parens(value)
    counter[key][normalized] += 1
    if normalized not in allowed:
        errors.append(f"line {line_no}: invalid {key}={value!r}")


def validate_record(rec, line_no, allowed, counters, errors, image_root, check_files=True):
    for key in ["image_paths", "signal_path", "structured_data"]:
        if key not in rec:
            errors.append(f"line {line_no}: missing top-level field {key}")
            return

    if rec.get("original_cot"):
        counters["meta"]["has_original_cot"] += 1
    if rec.get("structured_data", {}).get("summary_diag", {}).get("primary_label"):
        counters["primary_label"][rec["structured_data"]["summary_diag"]["primary_label"]] += 1

    if check_files:
        signal_path = rec.get("signal_path", "")
        if not signal_path or not os.path.isfile(signal_path):
            counters["meta"]["missing_signal"] += 1
            errors.append(f"line {line_no}: missing signal file {signal_path!r}")
        image_paths = rec.get("image_paths") or []
        if not image_paths:
            counters["meta"]["missing_image"] += 1
            errors.append(f"line {line_no}: empty image_paths")
        else:
            image_path = os.path.join(image_root, image_paths[0])
            if not os.path.isfile(image_path):
                counters["meta"]["missing_image"] += 1
                errors.append(f"line {line_no}: missing image file {image_path!r}")

    sd = rec["structured_data"]
    try:
        rr = sd["rhythm_rate"]
        ca = sd["conduction_axis"]
        vh = sd["voltage_hypertrophy"]
        ii = sd["ischemia_infarct"]
        qt = sd["qt_electrolytes"]
        summary = sd["summary_diag"]
    except KeyError as exc:
        errors.append(f"line {line_no}: missing structured_data field {exc}")
        return

    check_value(counters, errors, "rate_level", rr.get("rate_level"), allowed["rate_level"], line_no)
    check_value(counters, errors, "rhythm", rr.get("rhythm"), allowed["rhythm"], line_no)
    check_value(counters, errors, "axis", ca.get("axis"), allowed["axis"], line_no)
    check_value(counters, errors, "pr_status", ca.get("pr_status"), allowed["pr_status"], line_no)
    check_value(counters, errors, "qrs_width", ca.get("qrs_width"), allowed["qrs_width"], line_no)
    check_value(counters, errors, "conduction_status", ca.get("conduction_status"), allowed["conduction_status"], line_no)
    check_value(counters, errors, "voltage", vh.get("voltage"), allowed["voltage"], line_no)
    check_value(counters, errors, "qt_status", qt.get("qt_status"), allowed["qt_status"], line_no)

    for bool_key in ["lvh", "rvh"]:
        value = vh.get(bool_key)
        counters[bool_key][str(bool(value))] += 1
        if not isinstance(value, bool):
            errors.append(f"line {line_no}: {bool_key} should be bool, got {type(value).__name__}")

    for bool_key in ["st_elevation_present", "st_depression_present", "t_wave_abnormal", "q_wave_present"]:
        value = ii.get(bool_key)
        counters[bool_key][str(bool(value))] += 1
        if DATA_VERSION == "v4" and not isinstance(value, bool):
            errors.append(f"line {line_no}: {bool_key} should be bool, got {type(value).__name__}")

    value = summary.get("is_abnormal")
    counters["is_abnormal"][str(bool(value))] += 1
    if not isinstance(value, bool):
        errors.append(f"line {line_no}: is_abnormal should be bool, got {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Validate structured ECG label JSONL")
    parser.add_argument("--jsonl", default=JSONL_PATH)
    parser.add_argument("--image-root", default=IMAGE_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--no-file-check", action="store_true", default=False)
    args = parser.parse_args()

    label_config = get_label_config()
    allowed = allowed_values(label_config)
    counters = defaultdict(Counter)
    errors = []
    total = 0

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if args.limit is not None and total >= args.limit:
                break
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            validate_record(
                rec,
                line_no,
                allowed,
                counters,
                errors,
                args.image_root,
                check_files=not args.no_file_check,
            )

    print(f"Validated records: {total}")
    print(f"Errors: {len(errors)}")
    for err in errors[:args.max_errors]:
        print(f"  - {err}")
    if len(errors) > args.max_errors:
        print(f"  ... {len(errors) - args.max_errors} more errors")

    print("\nClass distributions:")
    for key in sorted(k for k in counters if k not in {"meta", "primary_label"}):
        count = counters[key]
        n = sum(count.values())
        parts = [f"{name}={value} ({value / max(n, 1) * 100:.1f}%)" for name, value in count.most_common()]
        print(f"  {key}: " + ", ".join(parts))

    print("\nMetadata:")
    meta = counters["meta"]
    print(f"  has_original_cot: {meta.get('has_original_cot', 0)}")
    print(f"  missing_signal: {meta.get('missing_signal', 0)}")
    print(f"  missing_image: {meta.get('missing_image', 0)}")
    print("  primary_label_top10:")
    for name, value in counters["primary_label"].most_common(10):
        print(f"    {name}: {value}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
