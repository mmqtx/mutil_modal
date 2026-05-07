"""Project-wide paths and training hyper-parameters.

Aligned with GEM-main/ecg_coca/open_clip/model_configs/coca_ViT-B-32.json
"""

import os

# ---------------------------------------------------------------------------
# Paths  / 路径配置
# ---------------------------------------------------------------------------
DATA_ROOT   = "/data/ljq24358/ecg_dataset"
# 数据版本配置 (v2/v3/v4，每个版本的标签结构不同)
DATA_VERSION = "v4"  # 可选: "v2", "v3", "v4"
JSONL_PATH  = os.path.join(DATA_ROOT, "ecg_jsons/structured_extraction/structured_labels_" + DATA_VERSION + ".jsonl")
IMAGE_ROOT  = os.path.join(DATA_ROOT, "ecg_images")
SIGNAL_ROOT = os.path.join(DATA_ROOT, "ecg_timeseries/mimic-iv/files")

# ---------------------------------------------------------------------------
# Pretrained weights (GEM / CLIP)  / 预训练权重路径
# ---------------------------------------------------------------------------
# GEM ECG-CoCa checkpoint:
#   Download from: https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB
#   File: cpt_wfep_epoch_20.pt
#   Place at: pretrained/cpt_wfep_epoch_20.pt
# GEM ECG 预训练检查点，信号编码器的权重来源
GEM_PRETRAINED_PATH = os.path.join(
    os.path.dirname(__file__), "pretrained", "cpt_wfep_epoch_20.pt"
)

# CLIP Vision Encoder (HuggingFace, same as GEM's llava/model/multimodal_encoder/clip_encoder.py):
#   Download from: https://huggingface.co/openai/clip-vit-large-patch14-336
#   Place at: pretrained/clip-vit-large-patch14-336/
#   Or leave empty to auto-download from HuggingFace.
# CLIP 视觉编码器路径，图像编码器的权重来源
CLIP_VIT_PATH = os.path.join(
    os.path.dirname(__file__), "pretrained", "clip-vit-large-patch14-336"
)

# ---------------------------------------------------------------------------
# GEM-aligned model configuration (from coca_ViT-B-32.json)  / 对齐 GEM 的模型配置
# ---------------------------------------------------------------------------
# Shared embedding dimension (GEM embed_dim=512)
# 共享嵌入维度，信号编码器最终输出维度
EMBED_DIM       = 512

# ECG signal encoder (GEM ecg_cfg)  / ECG 信号编码器参数
ECG_SEQ_LENGTH  = 5000      # 10s @ 500Hz  / 信号长度，10秒×500Hz
ECG_LEAD_NUM    = 12        # number of leads  / 导联数
ECG_PATCH_SIZE  = 50        # patch size in time steps  / 时间维度的 patch 大小
ECG_WIDTH       = 768       # transformer hidden dim  / Transformer 隐藏层维度
ECG_LAYERS      = 12        # transformer layers  / Transformer 层数
ECG_HEADS       = 12        # attention heads (width/head_width = 768/64 = 12)  / 注意力头数
ECG_MLP_RATIO   = 4.0       # MLP expansion ratio  / MLP 扩展倍率
FREEZE_SIGNAL_ENCODER = True  # resource-friendly default; unfreeze for later fine-tuning

# Image encoder uses HuggingFace CLIP ViT-L/14@336px (same as GEM)
# hidden_size=1024, image_size=336, patch_size=14
# 图像编码器使用 HuggingFace CLIP ViT-L/14@336px
IMAGE_SIZE      = 336       # ViT-L/14@336px  / 输入图像尺寸
FREEZE_IMAGE_ENCODER  = True  # resource-friendly default; keeps CLIP stable on 2x4090

# ---------------------------------------------------------------------------
# Downstream task dimensions  / 下游任务维度配置
# ---------------------------------------------------------------------------
FUSION_DIM       = 512     # fused feature dim (signal 512->512, image 1024->512)
                           # 融合特征维度，两模态投影到统一维度

NUM_HEADS        = 8       # attention heads in fusion / chain
                           # 融合模块和诊断链中交叉注意力的头数

NUM_CHAIN_LAYERS = 3       # transformer layers for inter-head self-attention (增加以提高诊断链交互深度)
                           # 诊断链中多头自注意力的层数

FUSION_NUM_LAYERS = 3      # number of cross-attention fusion layers (增加以提高模态融合深度)
                           # 融合模块的交叉注意力层数，层数越多模态交互越深

# 多模态实验开关：
# INPUT_MODE 用于消融实验，dual=信号+图像，signal=只看信号，image=只看图像。
# FUSION_TYPE 用于比较融合策略，cross_attention=交叉注意力，late_concat=简单后融合。
INPUT_MODE = "dual"
FUSION_TYPE = "cross_attention"

# Dropout 配置 (防止过拟合，提高泛化)
FUSION_DROPOUT    = 0.15   # fusion module dropout (增大以提高泛化)
HEAD_DROPOUT      = 0.15   # classification head dropout (增大以提高泛化)

# 分类头内部配置
HEAD_HIDDEN_RATIO = 0.75   # head hidden dim = in_dim * ratio (增大以提高分类头capacity，默认0.5)

# Uplift projections: signal 512 & image 1024 → fusion_dim
# 上投影头：将两模态特征投影到统一的 fusion_dim
UPLIFT_HIDDEN_DIM  = 512   # MLP intermediate hidden dim  / MLP 中间隐藏层维度
UPLIFT_NUM_LAYERS  = 2     # MLP layers (1=single Linear, >=2=MLP with ReLU)
                           # MLP 层数（1=单层线性，>=2=带 ReLU 的多层感知机）

# Contrastive projection heads: signal 512 & image 1024 → proj_dim
# 对比学习投影头：用于 InfoNCE 对比损失的投影
CONTRASTIVE_HIDDEN_DIM = 512   # MLP intermediate hidden dim  / MLP 中间隐藏层维度
CONTRASTIVE_OUT_DIM    = 256   # final projection dim for InfoNCE  / 对比损失最终投影维度
CONTRASTIVE_NUM_LAYERS = 2     # MLP layers  / MLP 层数

# ---------------------------------------------------------------------------
# Output & Logging  / 输出与日志路径
# ---------------------------------------------------------------------------
# Base directory for all experiment outputs  / 实验输出根目录
# Each run creates a subdirectory: OUTPUT_ROOT/<experiment_name>/<timestamp>/
# 每次训练创建子目录: OUTPUT_ROOT/<experiment_name>/<timestamp>/
OUTPUT_ROOT     = os.path.join(os.path.dirname(__file__), "outputs")

# Experiment name  / 实验名称，用于区分不同实验
EXPERIMENT_NAME = "ecg_diag"

# Sub-directory names  / 子目录名称
DIR_CHECKPOINTS = "checkpoints"    # model checkpoints  / 模型检查点
DIR_TENSORBOARD = "tensorboard"    # tensorboard logs  / TensorBoard 日志
DIR_LOGS        = "logs"           # text logs  / 文本日志

# ---------------------------------------------------------------------------
# Data  / 数据配置
# ---------------------------------------------------------------------------
SEED           = 42        # random seed for reproducibility  / 随机种子，保证可复现
NUM_WORKERS    = 4         # dataloader workers  / 数据加载线程数
BATCH_SIZE     = 128        # per GPU  / 每 GPU 批量大小
# Split ratios: train=70%, val=10%, test=20%  / 数据集划分比例
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.10
TEST_RATIO     = 0.20

# Conservative augmentation defaults for clinical labels.
# CutMix is disabled by default because it mixes ECG segments without mixing labels.
USE_SIGNAL_AUGMENTATION = True
USE_BASELINE_WANDER = True
USE_RANDOM_MASKING = True
USE_CUTMIX = False

# Randomly drop one non-text modality during training to improve robustness.
# 0.0 disables modality dropout. Recommended sweep: 0.0, 0.1, 0.2.
MODALITY_DROPOUT_PROB = 0.0

# Static class-balanced focal weights computed from the training split.
USE_STATIC_CLASS_WEIGHTS = True
CLASS_WEIGHT_BETA = 0.999
CLASS_WEIGHT_MAX = 20.0

# ---------------------------------------------------------------------------
# Training  / 训练超参数
# ---------------------------------------------------------------------------
LR             = 1e-4      # learning rate (降低以获得更稳定的收敛) / 学习率
WEIGHT_DECAY   = 0.05      # weight decay  / 权重衰减
WARMUP_STEPS   = 1000      # warmup steps  / 预热步数
EPOCHS         = 30        # total epochs  / 总训练轮数
GRAD_CLIP_NORM = 1.0       # gradient clipping  / 梯度裁剪
AMP_ENABLED    = True      # mixed precision (bf16)  / 混合精度训练
LOG_INTERVAL   = 50        # steps between logging  / 日志打印间隔步数
SAVE_INTERVAL     = 10        # epochs between checkpoints  / 检查点保存间隔轮数
SAVE_EVERY_EPOCH  = False     # whether to save periodic checkpoints  / 是否启用定期保存，False 则只保存 best.pt
SAVE_BEST_LOSS    = False     # 是否额外保存 best_loss.pt；默认关闭以节省磁盘空间

# Focal Loss 参数 (针对类别不平衡)
FOCAL_GAMMA    = 3.0       # focusing parameter (增大以更关注难样本/稀疏类) / 默认2.0，调大至3.0加强对难样本的关注

# 信号-图像全局对比学习 (InfoNCE) / Signal-Image Global Contrastive Learning
CONTRASTIVE_WEIGHT = 0.05     # weight for global contrastive loss (降低以减少对小类的影响) / 全局对比学习损失权重
CONTRASTIVE_TEMP   = 0.07     # temperature for global contrastive  / 全局对比学习温度参数

# ---------------------------------------------------------------------------
# Head-level Contrastive Learning  / 分类头级别的对比学习
# ---------------------------------------------------------------------------
# 即插即用组件：在每个分类头中加入对比学习，使同类样本特征更接近，异类样本特征更远离
# Plug-and-play component: Add contrastive learning to each classification head
# to make features of same class closer and different classes farther apart.
#
# 启用方式 / How to enable:
#   方法1: 修改下面的 USE_HEAD_CONTRASTIVE = True
#   Method 1: Change USE_HEAD_CONTRASTIVE = True below
#   方法2: 命令行参数 --use-head-contrastive
#   Method 2: Command line argument --use-head-contrastive
#
# 禁用方式 / How to disable:
#   修改下面的 USE_HEAD_CONTRASTIVE = False (默认)
#   Change USE_HEAD_CONTRASTIVE = False (default)
#
USE_HEAD_CONTRASTIVE   = False   # keep off by default; enable only for ablations
HEAD_CONTRASTIVE_WEIGHT = 0.05    # 分类头对比学习损失权重 (降低权重，减少对小类的挤压) / Weight for head contrastive loss (0.05-0.3)
HEAD_CONTRASTIVE_TEMP   = 0.15    # 温度参数 (放宽一点，对小类更友好) / Temperature (0.05-0.2, smaller = stricter)
#
# 参数建议 / Parameter recommendations:
#   - WEIGHT: 0.05 (弱) - 0.1 (中) - 0.2+ (强) / 默认 0.1
#   - TEMP: 0.05 (严格) - 0.1 (标准) - 0.2 (宽松) / 默认 0.1
#   - 建议先从小权重开始，观察模型收敛情况再调整
#   - Start with small weight, adjust based on convergence

# ---------------------------------------------------------------------------
# 标签配置 / Label Configuration
# ---------------------------------------------------------------------------
# 支持不同数据版本的标签结构配置 / Support different data versions
LABEL_CONFIGS = {
    # v2: 原始48维导联级缺血标签
    "v2": {
        "rhythm_rate": {
            "rate_level": {"classes": ["Fast(>100bpm)", "Normal(60-100bpm)", "Slow(<60bpm)"], "type": "multi-class"},
            "rhythm": {"classes": ["AFib", "AFlutter", "Ectopic", "Other", "Sinus"], "type": "multi-class"},
        },
        "conduction_axis": {
            "axis": {"classes": ["Extreme", "Left", "Normal", "Right", "Unknown"], "type": "multi-class"},
            "pr_status": {"classes": ["Normal", "Prolonged", "Short", "Unknown"], "type": "multi-class"},
            "qrs_width": {"classes": ["Narrow(<120ms)", "Wide(>=120ms)"], "type": "multi-class"},
            "conduction_status": {"classes": ["Block", "Delay", "Normal", "Other"], "type": "multi-class"},
        },
        "voltage": {
            "lvh": {"classes": 2, "type": "binary"},
            "rvh": {"classes": 2, "type": "binary"},
            "voltage": {"classes": ["Low", "Normal"], "type": "multi-class"},
        },
        "ischemia_infarct": {
            "dim": 48,  # 4 subtypes × 12 leads
            "type": "multi-label",
            "subtypes": ["st_elevation", "st_depression", "t_wave_alt", "q_wave"],
            "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        },
        "qt_electrolytes": {
            "qt_status": {"classes": ["Normal", "Prolonged"], "type": "multi-class"},
        },
        "summary": {
            "is_abnormal": {"classes": 2, "type": "binary"},
        },
    },
    # v3: 合并稀疏心律 + 区域级缺血标签
    "v3": {
        "rhythm_rate": {
            "rate_level": {"classes": ["Fast(>100bpm)", "Normal(60-100bpm)", "Slow(<60bpm)"], "type": "multi-class"},
            "rhythm": {"classes": ["AFib", "Other_Arrhythmia", "Sinus"], "type": "multi-class"},  # 合并后
        },
        "conduction_axis": {
            "axis": {"classes": ["Extreme", "Left", "Normal", "Right"], "type": "multi-class"},  # 合并Unknown
            "pr_status": {"classes": ["Normal", "Prolonged", "Short", "Unknown"], "type": "multi-class"},
            "qrs_width": {"classes": ["Narrow(<120ms)", "Wide(>=120ms)"], "type": "multi-class"},
            "conduction_status": {"classes": ["Block", "Delay", "Normal", "Other"], "type": "multi-class"},
        },
        "voltage": {
            "lvh": {"classes": 2, "type": "binary"},
            "rvh": {"classes": 2, "type": "binary"},
            "voltage": {"classes": ["Low", "Normal"], "type": "multi-class"},
        },
        "ischemia_infarct": {
            "st_elevation_territory": {"classes": ["none", "anterior", "septal", "inferior", "lateral", "multiple", "anterior_septal"], "type": "multi-class"},
            "st_depression_territory": {"classes": ["none", "anterior", "septal", "inferior", "lateral", "multiple", "anterior_septal"], "type": "multi-class"},
            "t_wave_abnormality": {"classes": ["none", "localized", "diffuse"], "type": "multi-class"},
            "q_wave_territory": {"classes": ["none", "anterior", "septal", "inferior", "lateral", "multiple", "anterior_septal"], "type": "multi-class"},
            "type": "multi-class",
        },
        "qt_electrolytes": {
            "qt_status": {"classes": ["Normal", "Prolonged"], "type": "multi-class"},
        },
        "summary": {
            "is_abnormal": {"classes": 2, "type": "binary"},
        },
    },
    # v4: 优化版 - 二分类缺血 + 合并稀疏PR/传导
    "v4": {
        "rhythm_rate": {
            "rate_level": {"classes": ["Fast(>100bpm)", "Normal(60-100bpm)", "Slow(<60bpm)"], "type": "multi-class"},
            "rhythm": {"classes": ["AFib", "Other_Arrhythmia", "Sinus"], "type": "multi-class"},
        },
        "conduction_axis": {
            "axis": {"classes": ["Extreme", "Left", "Normal", "Right"], "type": "multi-class"},
            "pr_status": {"classes": ["Abnormal", "Normal", "Prolonged"], "type": "multi-class"},  # 合并Short+Unknown
            "qrs_width": {"classes": ["Narrow(<120ms)", "Wide(>=120ms)"], "type": "multi-class"},
            "conduction_status": {"classes": ["Block", "Delay", "Normal"], "type": "multi-class"},  # 删除Other
        },
        "voltage": {
            "lvh": {"classes": 2, "type": "binary"},
            "rvh": {"classes": 2, "type": "binary"},
            "voltage": {"classes": ["Low", "Normal"], "type": "multi-class"},
        },
        "ischemia_infarct": {
            # 二分类：有/无
            "st_elevation_present": {"classes": 2, "type": "binary"},
            "st_depression_present": {"classes": 2, "type": "binary"},
            "t_wave_abnormal": {"classes": 2, "type": "binary"},
            "q_wave_present": {"classes": 2, "type": "binary"},
            "type": "multi-label",
        },
        "qt_electrolytes": {
            "qt_status": {"classes": ["Normal", "Prolonged"], "type": "multi-class"},
        },
        "summary": {
            "is_abnormal": {"classes": 2, "type": "binary"},
        },
    },
}

# 当前使用的标签配置 (自动根据DATA_VERSION选择)
def get_label_config():
    """获取当前数据版本的标签配置"""
    return LABEL_CONFIGS.get(DATA_VERSION, LABEL_CONFIGS["v2"])

# ---------------------------------------------------------------------------
# 任务级损失权重配置 / Task-level Loss Weights
# ---------------------------------------------------------------------------
# 为稀疏类别/任务设置更高的损失权重
TASK_LOSS_WEIGHTS = {
    # 稀疏类别权重 (在训练时应用)
    "class_weights": {
        # rhythm分类权重
        "rhythm.Sinus": 1.0,
        "rhythm.AFib": 1.0,
        "rhythm.Other_Arrhythmia": 1.5,  # v3/v4合并后的类别
        # v2版本的稀疏心律
        "rhythm.AFlutter": 5.0,
        "rhythm.Ectopic": 5.0,
        "rhythm.Other": 5.0,

        # 电轴分类权重
        "axis.Normal": 1.0,
        "axis.Left": 1.0,
        "axis.Extreme": 2.0,
        "axis.Right": 3.0,  # 稀疏
        "axis.Unknown": 2.0,

        # PR状态权重
        "pr_status.Normal": 1.0,
        "pr_status.Prolonged": 1.5,
        "pr_status.Short": 5.0,
        "pr_status.Unknown": 3.0,
        "pr_status.Abnormal": 2.0,  # v4合并后

        # 传导状态权重
        "conduction_status.Normal": 1.0,
        "conduction_status.Block": 1.5,
        "conduction_status.Delay": 2.0,
        "conduction_status.Other": 5.0,

        # RVH高权重 (极稀疏)
        "rvh.positive": 20.0,
        "rvh.negative": 1.0,
    },
    # 整体任务权重 (每个诊断头的权重)
    "task_weights": {
        "rhythm_rate": 1.0,
        "conduction_axis": 1.0,
        "voltage": 1.0,
        "ischemia_infarct": 5.0,  # 提高缺血任务权重
        "qt_electrolytes": 2.0,
        "summary": 1.0,
    },
}

# ---------------------------------------------------------------------------
# Classification head dimensions  / 分类头配置
# ---------------------------------------------------------------------------
HEAD_CONFIGS = {
    "rhythm_rate":       {"num_classes": None, "type": "multi-class"},   # auto-filled / 心律心率
    "conduction_axis":   {"num_classes": None, "type": "multi-class"},   # 传导与电轴
    "voltage":           {"num_classes": None, "type": "multi-class"},   # 电压与肥厚
    "ischemia_infarct":   {"dim": None, "type": "multi-label"},    # auto-filled / 缺血与梗死
    "qt_electrolytes":   {"num_classes": None, "type": "multi-class"},   # QT与电解质
    "summary":           {"num_classes": 2,     "type": "binary"},       # 总结（正常/异常）
}

# 根据DATA_VERSION自动填充HEAD_CONFIGS
def auto_fill_head_configs():
    """根据当前数据版本自动填充HEAD_CONFIGS的类别数"""
    label_config = get_label_config()
    filled_configs = {}

    # rhythm_rate
    if "rhythm_rate" in label_config:
        rr_config = label_config["rhythm_rate"]
        filled_configs["rhythm_rate"] = {
            "num_classes": 2 + len(rr_config.get("rhythm", {}).get("classes", [])),
            "type": "multi-class",
        }

    # conduction_axis
    if "conduction_axis" in label_config:
        ca_config = label_config["conduction_axis"]
        num_classes = (
            len(ca_config.get("axis", {}).get("classes", [])) +
            len(ca_config.get("pr_status", {}).get("classes", [])) +
            len(ca_config.get("qrs_width", {}).get("classes", [])) +
            len(ca_config.get("conduction_status", {}).get("classes", []))
        )
        filled_configs["conduction_axis"] = {"num_classes": num_classes, "type": "multi-class"}

    # voltage
    if "voltage" in label_config:
        v_config = label_config["voltage"]
        num_classes = (
            (2 if v_config.get("lvh", {}).get("classes", 2) == 2 else 1) +
            (2 if v_config.get("rvh", {}).get("classes", 2) == 2 else 1) +
            len(v_config.get("voltage", {}).get("classes", []))
        )
        filled_configs["voltage"] = {"num_classes": num_classes, "type": "multi-class"}

    # ischemia_infarct - 根据版本设置不同
    if DATA_VERSION == "v4":
        # v4: 4个二分类任务
        filled_configs["ischemia_infarct"] = {"dim": 4, "type": "multi-label"}
    elif DATA_VERSION == "v3":
        # v3: 区域级多分类，计算总类别数
        ii_config = label_config.get("ischemia_infarct", {})
        num_classes = (
            len(ii_config.get("st_elevation_territory", {}).get("classes", [])) +
            len(ii_config.get("st_depression_territory", {}).get("classes", [])) +
            len(ii_config.get("t_wave_abnormality", {}).get("classes", [])) +
            len(ii_config.get("q_wave_territory", {}).get("classes", []))
        )
        filled_configs["ischemia_infarct"] = {"dim": num_classes, "type": "multi-class"}
    else:
        # v2: 48维多标签
        filled_configs["ischemia_infarct"] = {"dim": 48, "type": "multi-label"}

    # qt_electrolytes
    if "qt_electrolytes" in label_config:
        qt_config = label_config["qt_electrolytes"]
        filled_configs["qt_electrolytes"] = {
            "num_classes": len(qt_config.get("qt_status", {}).get("classes", [])),
            "type": "multi-class",
        }

    # summary保持不变
    filled_configs["summary"] = HEAD_CONFIGS["summary"]

    # 更新HEAD_CONFIGS
    for key, value in filled_configs.items():
        HEAD_CONFIGS[key] = value

    return filled_configs

# Chain order (clinical reasoning sequence)  / 诊断链顺序（模拟临床推理流程）
CHAIN_ORDER = [
    "rhythm_rate",        # 心律心率
    "conduction_axis",    # 传导与电轴
    "voltage",            # 电压与肥厚
    "ischemia_infarct",   # 缺血与梗死
    "qt_electrolytes",    # QT与电解质
    "summary",            # 总结诊断
]
