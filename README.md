# ECG Multi-Modal Diagnostic System

基于信号-图像多模态融合的ECG自动诊断系统，采用6步诊断链结构模拟临床推理流程。

## 项目结构

```
mutil_modal/
├── config.py              # 全局配置（超参数、路径等）
├── scripts/                # 训练、评估、推理脚本
│   ├── train.py           # DDP训练脚本
│   ├── evaluate.py        # 评估脚本（支持train/val/test集）
│   ├── predict.py         # 单样本推理脚本
│   └── heatmap.py         # GradCAM注意力热力图生成
├── data/                   # 数据处理模块
│   ├── dataset.py         # ECG多模态数据集
│   └── transforms.py      # 数据增强
├── models/                 # 模型架构
│   ├── model.py           # ECGDiagModel主模型
│   ├── backbones.py       # 信号/图像编码器
│   ├── fusion.py          # 跨模态融合模块
│   ├── heads.py           # 6步诊断链
│   └── losses.py          # 三层动态补偿损失
├── utils/                  # 工具函数
├── pretrained/            # 预训练权重
│   ├── clip-vit-*/        # CLIP ViT模型
│   └── cpt_*.pt          # GEM ECG预训练权重
├── outputs/                # 训练输出
└── docs/                   # 文档

## 特性

- **多模态输入**：12导联ECG信号 (5000Hz) + 12导联ECG图像
- **预训练编码器**：GEM-ECG (信号) + CLIP ViT-L/14 (图像)
- **6步诊断链**：模拟临床推理流程
- **三层动态补偿损失**：Focal Loss + Learned Class Weights + Uncertainty Weighting
- **分类头对比学习**：即插即用组件，使同类特征更接近，异类特征更远离
- **DDP训练**：支持多GPU分布式训练

## 数据集划分

- **Train**: 70% (21,000样本) - 训练
- **Val**: 10% (3,000样本) - 验证，选择最佳模型
- **Test**: 20% (6,000样本) - 最终评估

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 2x RTX 4090 (24GB)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备预训练权重

```bash
# CLIP ViT (会自动下载)
# GEM ECG预训练权重需要手动下载：
# 下载地址: https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB
# 将 cpt_wfep_epoch_20.pt 放入 pretrained/ 目录
```

### 3. 训练模型

```bash
# 单GPU训练
python scripts/train.py

# 2x4090 DDP训练
torchrun --nproc_per_node=2 scripts/train.py
```

### 4. 评估模型

```bash
# 在test集上评估（推荐）
python scripts/evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

# 在val集上评估
python scripts/evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split val
```

### 5. 生成注意力热力图

```bash
python scripts/heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt
```

### 6. 单样本推理

```bash
python scripts/predict.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt \
                          --signal_path /path/to/ecg.hea \
                          --image_path /path/to/ecg.png
```

## 配置说明

主要配置在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 128 | 批量大小 |
| `LR` | 2e-4 | 学习率 |
| `EPOCHS` | 30 | 训练轮数 |
| `TRAIN_RATIO` | 0.70 | 训练集比例 |
| `VAL_RATIO` | 0.10 | 验证集比例 |
| `TEST_RATIO` | 0.20 | 测试集比例 |
| `FUSION_DIM` | 512 | 融合特征维度 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `USE_HEAD_CONTRASTIVE` | False | 是否启用分类头对比学习 |
| `HEAD_CONTRASTIVE_WEIGHT` | 0.1 | 分类头对比学习权重 (0.05-0.3) |
| `HEAD_CONTRASTIVE_TEMP` | 0.1 | 温度参数 (0.05-0.2) |

## 分类头对比学习

即插即用组件，可在每个分类头中加入对比学习，使同类样本特征更接近，异类样本特征更远离。

### 启用方法

**方法1: 修改 config.py**
```python
# config.py
USE_HEAD_CONTRASTIVE = True   # 启用
HEAD_CONTRASTIVE_WEIGHT = 0.1  # 权重
HEAD_CONTRASTIVE_TEMP = 0.1    # 温度
```

**方法2: 命令行参数**
```bash
# 启用
torchrun --nproc_per_node=2 scripts/train.py --use-head-contrastive

# 自定义参数
torchrun --nproc_per_node=2 scripts/train.py \
    --use-head-contrastive \
    --head-contrastive-weight 0.2 \
    --head-contrastive-temp 0.05

# 禁用（即使config中启用）
torchrun --nproc_per_node=2 scripts/train.py --no-head-contrastive
```

### 参数说明

| 参数 | 范围 | 说明 | 推荐值 |
|------|------|------|--------|
| WEIGHT | 0.05-0.3 | 对比损失权重 | 0.1 |
| TEMP | 0.05-0.2 | 温度（越小越严格） | 0.1 |

**建议**：从小权重开始（0.1），观察模型收敛情况再调整。

## 诊断任务

模型输出6大类诊断结果：

1. **心律心率** (Rhythm & Rate)
   - rate_level: Fast/Normal/Slow
   - rhythm: Sinus/AFib/AFlutter/Ectopic/Other

2. **传导电轴** (Conduction & Axis)
   - axis: Normal/Left/Right/Extreme/Unknown
   - pr_status: Normal/Prolonged/Short/Unknown
   - qrs_width: Narrow/Wide
   - conduction_status: Normal/Delay/Block/Other

3. **电压肥厚** (Voltage & Hypertrophy)
   - lvh: 0/1
   - rvh: 0/1
   - voltage: Normal/Low

4. **缺血梗死** (Ischemia & Infarct)
   - findings: 4 subtypes × 12 leads (48维多标签)

5. **QT电解质** (QT & Electrolytes)
   - qt_status: Normal/Prolonged

6. **总结诊断** (Summary)
   - is_abnormal: 0/1

## 输出目录

```
outputs/ecg_diag/<timestamp>/
├── checkpoints/
│   ├── best.pt          # 最佳模型（Macro-F1最高）
│   └── best_loss.pt     # Loss最低模型（对比用）
├── logs/
│   └── train.log        # 训练日志
├── tensorboard/          # TensorBoard日志
└── evaluation_*/         # 评估结果
    ├── eval_results.json
    ├── summary.png
    └── cm_*.png         # 混淆矩阵
```

## TensorBoard监控

```bash
tensorboard --logdir outputs/ecg_diag/<timestamp>/tensorboard
```

监控指标：
- `train/loss_total`: 总损失
- `train/class_weights/*`: 学习到的类别权重
- `train/log_var/*`: 任务级不确定度权重
- `val/macro_f1_mean`: 验证集Macro-F1均值
- `val/f1_*`: 各子任务F1分数

## 已知问题

- **类别不平衡**：部分类别极度不平衡（如rvh阳性样本仅0.86%），已通过三层动态补偿损失缓解
- **数据依赖**：需要外部数据集 `/data/ljq24358/ecg_dataset/`

## TODO

- [ ] 添加数据增强策略
- [ ] 支持混合精度训练
- [ ] 添加模型压缩/量化
- [ ] 支持更多预训练权重

## 许可证

MIT License
