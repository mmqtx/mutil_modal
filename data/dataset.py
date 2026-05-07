"""
ECG Multi-Modal Structured Classification Dataset.

Reads JSONL annotations and provides:
  - 1-D signal tensor  (12-lead wfdb, 5000 time-steps)
  - 2-D image tensor   (PNG, 3x336x336, CLIP-normalized)
  - 6 groups of classification targets simulating a clinical diagnostic chain

支持多版本数据 (v2/v3/v4)，通过config.DATA_VERSION配置
"""

import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import wfdb

from .transforms import get_signal_transform, get_image_transform
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_VERSION, get_label_config, TASK_LOSS_WEIGHTS

logger = logging.getLogger(__name__)

# Ischemia encoding constants for v2 (48-dim lead-level)
_ISCHEMIA_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
_ISCHEMIA_SUBTYPES = ["st_elevation", "st_depression", "t_wave_alt", "q_wave"]


def _strip_parens(label: str) -> str:
    """Remove parenthetical qualifiers:  'Fast(>100bpm)' -> 'Fast'."""
    return re.sub(r'\s*\(.*?\)\s*', '', label).strip()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ECGMultiModalDataset(Dataset):
    """Multi-modal ECG dataset with hierarchical structured labels.

    支持多版本数据 (v2/v3/v4)，通过config.DATA_VERSION配置

    Returns per sample:
        signal        : FloatTensor (12, 5000)
        image         : FloatTensor (3, 336, 336)
        labels        : dict  with 6 keys, each mapping to a tensor

        v2版本 (48维导联级缺血):
            - rhythm_rate       : (2,)   int64   [rate_idx, rhythm_idx]
            - conduction_axis   : (4,)   int64   [axis, pr, qrs, cond]
            - voltage           : (3,)   int64   [lvh, rvh, voltage_idx]
            - ischemia_infarct  : (48,)  int64   multi-hot  (4 subtypes x 12 leads)
            - qt_electrolytes   : (1,)   int64   [qt_idx]
            - summary           : (1,)   int64   [is_abnormal]

        v4版本 (二分类缺血):
            - rhythm_rate       : (2,)   int64   [rate_idx, rhythm_idx]
            - conduction_axis   : (4,)   int64   [axis, pr, qrs, cond]
            - voltage           : (3,)   int64   [lvh, rvh, voltage_idx]
            - ischemia_infarct  : (4,)   int64   binary [st_elev, st_dep, t_wave, q_wave]
            - qt_electrolytes   : (1,)   int64   [qt_idx]
            - summary           : (1,)   int64   [is_abnormal]
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        is_train: bool = True,
        seq_length: int = 5000,
        image_size: int = 336,
        max_samples: Optional[int] = None,
        load_signal: bool = True,
        load_image: bool = True,
        use_signal_augmentation: bool = True,
        use_baseline_wander: bool = True,
        use_cutmix: bool = False,
        use_random_masking: bool = True,
    ):
        super().__init__()
        self.image_root = image_root
        self.is_train = is_train
        self.seq_length = seq_length
        self.image_size = image_size
        self.load_signal = load_signal
        self.load_image = load_image

        # Transforms
        self.signal_transform = get_signal_transform(
            is_train,
            seq_length,
            use_augmentation=use_signal_augmentation,
            use_baseline_wander=use_baseline_wander,
            use_cutmix=use_cutmix,
            use_random_masking=use_random_masking,
        )
        self.image_transform  = get_image_transform(is_train, image_size)

        # ---- Parse JSONL & build vocab ----
        self.records: List[dict] = []
        self._load_and_index(jsonl_path, max_samples)

        # Build label vocabularies from the training split
        self.vocab = self._build_vocab()
        logger.info("Dataset loaded: %d records | vocab sizes: %s",
                     len(self.records),
                     {k: len(v) for k, v in self.vocab.items()})

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_and_index(self, path: str, max_samples: Optional[int]):
        with open(path, 'r') as f:
            for line in f:
                if max_samples is not None and len(self.records) >= max_samples:
                    break
                rec = json.loads(line)
                # Validate that both modalities are accessible
                sp = rec.get("signal_path", "")
                ip = rec.get("image_paths", [None])[0]
                if sp and ip:
                    self.records.append(rec)
        if len(self.records) == 0:
            logger.warning("No valid records loaded from %s", path)

    def _build_vocab(self) -> Dict[str, List[str]]:
        """根据DATA_VERSION构建标签词汇表"""
        label_config = get_label_config()

        if DATA_VERSION == "v4":
            # v4版本：二分类缺血
            rate_levels, rhythms = set(), set()
            axes, prs, qrs_ws, conds = set(), set(), set(), set()
            voltages = set()
            qt_statuses = set()

            for rec in self.records:
                sd = rec["structured_data"]

                rr = sd["rhythm_rate"]
                rate_levels.add(_strip_parens(rr["rate_level"]))
                rhythms.add(rr["rhythm"])

                ca = sd["conduction_axis"]
                axes.add(ca["axis"])
                prs.add(ca["pr_status"])
                qrs_ws.add(_strip_parens(ca["qrs_width"]))
                conds.add(ca["conduction_status"])

                vh = sd["voltage_hypertrophy"]
                voltages.add(vh["voltage"])

                qt = sd["qt_electrolytes"]
                qt_statuses.add(qt["qt_status"])

            vocab = {
                "rate_level": sorted(rate_levels),
                "rhythm": sorted(rhythms),
                "axis": sorted(axes),
                "pr_status": sorted(prs),
                "qrs_width": sorted(qrs_ws),
                "conduction_status": sorted(conds),
                "voltage": sorted(voltages),
                "qt_status": sorted(qt_statuses),
            }
        elif DATA_VERSION == "v3":
            # v3版本：区域级缺血
            vocab = self._build_vocab_v3()
        else:
            # v2版本：48维导联级缺血
            vocab = self._build_vocab_v2()

        return vocab

    def _build_vocab_v2(self) -> Dict[str, List[str]]:
        """v2版本词汇表构建（48维导联级缺血）"""
        rate_levels, rhythms = set(), set()
        axes, prs, qrs_ws, conds = set(), set(), set(), set()
        voltages = set()
        qt_statuses = set()
        primary_labels = set()

        for rec in self.records:
            sd = rec["structured_data"]

            rr = sd["rhythm_rate"]
            rate_levels.add(_strip_parens(rr["rate_level"]))
            rhythms.add(rr["rhythm"])

            ca = sd["conduction_axis"]
            axes.add(ca["axis"])
            prs.add(ca["pr_status"])
            qrs_ws.add(_strip_parens(ca["qrs_width"]))
            conds.add(ca["conduction_status"])

            vh = sd["voltage_hypertrophy"]
            voltages.add(vh["voltage"])

            qt = sd["qt_electrolytes"]
            qt_statuses.add(qt["qt_status"])

            sm = sd["summary_diag"]
            primary_labels.add(sm.get("primary_label", "Unknown"))

        vocab = {
            "rate_level": sorted(rate_levels),
            "rhythm": sorted(rhythms),
            "axis": sorted(axes),
            "pr_status": sorted(prs),
            "qrs_width": sorted(qrs_ws),
            "conduction_status": sorted(conds),
            "voltage": sorted(voltages),
            "qt_status": sorted(qt_statuses),
            "primary_label": sorted(primary_labels),
        }
        return vocab

    def _build_vocab_v3(self) -> Dict[str, List[str]]:
        """v3版本词汇表构建（区域级缺血）"""
        rate_levels, rhythms = set(), set()
        axes, prs, qrs_ws, conds = set(), set(), set(), set()
        voltages = set()
        qt_statuses = set()
        # 区域级缺血词汇
        ischemia_territories = set()
        t_wave_types = set()

        for rec in self.records:
            sd = rec["structured_data"]

            rr = sd["rhythm_rate"]
            rate_levels.add(_strip_parens(rr["rate_level"]))
            rhythms.add(rr["rhythm"])

            ca = sd["conduction_axis"]
            axes.add(ca["axis"])
            prs.add(ca["pr_status"])
            qrs_ws.add(_strip_parens(ca["qrs_width"]))
            conds.add(ca["conduction_status"])

            vh = sd["voltage_hypertrophy"]
            voltages.add(vh["voltage"])

            qt = sd["qt_electrolytes"]
            qt_statuses.add(qt["qt_status"])

            # 缺血区域级词汇
            ii = sd.get("ischemia_infarct", {})
            if "st_elevation_territory" in ii:
                ischemia_territories.add("st_elev_" + ii["st_elevation_territory"])
            if "st_depression_territory" in ii:
                ischemia_territories.add("st_dep_" + ii["st_depression_territory"])
            if "t_wave_abnormality" in ii:
                ischemia_territories.add("t_wave_" + ii["t_wave_abnormality"])
            if "q_wave_territory" in ii:
                ischemia_territories.add("q_wave_" + ii["q_wave_territory"])

        vocab = {
            "rate_level": sorted(rate_levels),
            "rhythm": sorted(rhythms),
            "axis": sorted(axes),
            "pr_status": sorted(prs),
            "qrs_width": sorted(qrs_ws),
            "conduction_status": sorted(conds),
            "voltage": sorted(voltages),
            "qt_status": sorted(qt_statuses),
            "ischemia_territory": sorted(ischemia_territories),
        }
        return vocab

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    def _encode_label(self, name: str, value: str) -> int:
        v = self.vocab.get(name, [])
        stripped = _strip_parens(value)
        if stripped in v:
            return v.index(stripped)
        # Fallback: try original value
        if value in v:
            return v.index(value)
        logger.warning("Unknown %s=%s, mapping to 0", name, value)
        return 0

    def _encode_ischemia(self, sd: dict) -> torch.Tensor:
        """根据DATA_VERSION编码缺血标签"""
        if DATA_VERSION == "v4":
            # v4版本：二分类
            return self._encode_ischemia_v4(sd)
        elif DATA_VERSION == "v3":
            # v3版本：区域级多分类
            return self._encode_ischemia_v3(sd)
        else:
            # v2版本：48维导联级
            return self._encode_ischemia_v2(sd)

    def _encode_ischemia_v2(self, sd: dict) -> torch.Tensor:
        """v2版本: 48维导联级多标签"""
        lead2idx = {l: i for i, l in enumerate(_ISCHEMIA_LEADS)}
        vec = torch.zeros(len(_ISCHEMIA_SUBTYPES) * len(_ISCHEMIA_LEADS), dtype=torch.float32)
        for si, subtype in enumerate(_ISCHEMIA_SUBTYPES):
            leads = sd["ischemia_infarct"].get(subtype, [])
            for lead in leads:
                if lead in lead2idx:
                    vec[si * len(_ISCHEMIA_LEADS) + lead2idx[lead]] = 1.0
        return vec

    def _encode_ischemia_v3(self, sd: dict) -> torch.Tensor:
        """v3版本: 区域级多分类"""
        ii = sd["ischemia_infarct"]

        # 每个区域级任务作为一个多分类任务
        st_elev_territory = ii.get("st_elevation_territory", "none")
        st_dep_territory = ii.get("st_depression_territory", "none")
        t_wave_type = ii.get("t_wave_abnormality", "none")
        q_wave_territory = ii.get("q_wave_territory", "none")

        # 编码为整数索引
        vocab = self.vocab.get("ischemia_territory", {})

        return torch.tensor([
            vocab.get("st_elev_" + st_elev_territory, 0),
            vocab.get("st_dep_" + st_dep_territory, 0),
            vocab.get("t_wave_" + t_wave_type, 0),
            vocab.get("q_wave_" + q_wave_territory, 0),
        ], dtype=torch.long)

    def _encode_ischemia_v4(self, sd: dict) -> torch.Tensor:
        """v4版本: 二分类（有/无）"""
        ii = sd["ischemia_infarct"]

        # 4个二分类任务
        st_elev = 1 if ii.get("st_elevation_present", False) else 0
        st_dep = 1 if ii.get("st_depression_present", False) else 0
        t_wave = 1 if ii.get("t_wave_abnormal", False) else 0
        q_wave = 1 if ii.get("q_wave_present", False) else 0

        return torch.tensor([st_elev, st_dep, t_wave, q_wave], dtype=torch.long)

    def _encode_labels(self, sd: dict) -> Dict[str, torch.Tensor]:
        """根据DATA_VERSION编码标签"""
        labels = {}

        # 1. Rhythm / Rate
        rr = sd["rhythm_rate"]
        labels["rhythm_rate"] = torch.tensor([
            self._encode_label("rate_level", rr["rate_level"]),
            self._encode_label("rhythm", rr["rhythm"]),
        ], dtype=torch.long)

        # 2. Conduction / Axis
        ca = sd["conduction_axis"]
        labels["conduction_axis"] = torch.tensor([
            self._encode_label("axis", ca["axis"]),
            self._encode_label("pr_status", ca["pr_status"]),
            self._encode_label("qrs_width", ca["qrs_width"]),
            self._encode_label("conduction_status", ca["conduction_status"]),
        ], dtype=torch.long)

        # 3. Voltage / Hypertrophy
        vh = sd["voltage_hypertrophy"]
        labels["voltage"] = torch.tensor([
            int(vh["lvh"]),
            int(vh["rvh"]),
            self._encode_label("voltage", vh["voltage"]),
        ], dtype=torch.long)

        # 4. Ischemia / Infarct - 根据版本选择编码方式
        labels["ischemia_infarct"] = self._encode_ischemia(sd)

        # 5. QT / Electrolytes
        labels["qt_electrolytes"] = torch.tensor([
            self._encode_label("qt_status", sd["qt_electrolytes"]["qt_status"]),
        ], dtype=torch.long)

        # 6. Summary / Diagnosis
        labels["summary"] = torch.tensor([
            int(sd["summary_diag"]["is_abnormal"]),
        ], dtype=torch.long)

        return labels

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def _load_signal(self, path: str) -> torch.Tensor:
        """Load wfdb signal -> (12, 5000) float32, with NaN/Inf cleanup."""
        try:
            # wfdb.rdsamp auto-appends .hea; strip it if present
            wfdb_path = path.replace(".hea", "")
            signal, _ = wfdb.rdsamp(wfdb_path)
        except Exception as e:
            logger.warning("Failed to load signal %s: %s, returning zeros", path, e)
            signal = np.zeros((5000, 12), dtype=np.float32)

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        # (time, lead) -> (lead, time)
        signal = np.transpose(signal, (1, 0)).astype(np.float32)
        # Add batch dim -> (1, 12, T), apply transform, squeeze
        tensor = torch.from_numpy(signal).unsqueeze(0)
        tensor = self.signal_transform(tensor)
        return tensor.squeeze(0)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        """Load PNG image and apply CLIP-style transform."""
        abs_path = os.path.join(self.image_root, rel_path)
        if not os.path.isfile(abs_path):
            logger.warning("Image not found: %s, returning zeros", abs_path)
            return torch.zeros(3, 336, 336, dtype=torch.float32)
        try:
            img = Image.open(abs_path).convert("RGB")
            return self.image_transform(img)
        except Exception as e:
            logger.warning("Failed to load image %s: %s, returning zeros", abs_path, e)
            return torch.zeros(3, 336, 336, dtype=torch.float32)

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        sd = rec["structured_data"]

        signal = (
            self._load_signal(rec["signal_path"])
            if self.load_signal
            else torch.zeros(12, self.seq_length, dtype=torch.float32)
        )
        image = (
            self._load_image(rec["image_paths"][0])
            if self.load_image
            else torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        )
        labels = self._encode_labels(sd)

        # Handle both "id" (v2/v3) and "file_id" (v4)
        sample_id = rec.get("id", rec.get("file_id", ""))

        return {
            "signal": signal,
            "image":  image,
            "labels": labels,
            "id":     sample_id,
        }
