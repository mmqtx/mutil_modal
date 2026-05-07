# Scripts 使用说明

本目录包含ECG诊断系统的训练、评估、推理和可视化脚本。

## 脚本列表

### train.py - 训练脚本

DDP多GPU训练，支持三层动态补偿损失。

**用法:**
```bash
# 单GPU训练
python train.py

# 2x4090 DDP训练 (推荐)
torchrun --nproc_per_node=2 train.py

# 自定义参数
torchrun --nproc_per_node=2 train.py --epochs 50 --batch-size 64
```

**关键参数:**
- `--epochs`: 训练轮数 (默认: 30)
- `--batch-size`: 批量大小 (默认: 128)
- `--lr`: 学习率 (默认: 1e-4)
- `--contrastive-weight`: 信号-图像全局对比学习权重 (默认: 0.05)
- `--input-mode`: 输入模态消融，`dual`=信号+图像，`signal`=只用信号，`image`=只用图像
- `--fusion-type`: 融合方式，`cross_attention`=交叉注意力，`late_concat`=简单后融合
- `--modality-dropout-prob`: 训练时随机丢弃一个模态的概率，只在 `dual` 模式生效
- `--use-cutmix`: 开启信号 CutMix，默认关闭
- `--resume`: 从检查点恢复训练

**输出:**
- `outputs/ecg_diag/<timestamp>/` - 训练输出目录
- `best.pt` - Macro-F1最高的模型
- `best_loss.pt` - Loss最低的模型（对比用）

---

### evaluate.py - 评估脚本

在指定数据集上生成完整评估报告，包含混淆矩阵和详细指标。

**用法:**
```bash
# 在test集上评估（推荐）
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

# 在val集上评估
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split val

# 在train集上评估（检查过拟合）
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split train

# 使用自定义batch size
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --batch-size 64
```

**参数:**
- `--checkpoint`: 模型检查点路径 (必需)
- `--split`: 数据集选择 {train,val,test} (默认: test)
- `--batch-size`: 批量大小 (默认: 128)
- `--output-dir`: 输出目录 (默认: 自动检测)
- `--input-mode`: 评估时输入模态消融，可用于同一个 checkpoint 比较信号/图像贡献
- `--fusion-type`: 融合结构，需要和训练 checkpoint 时保持一致
- `--thresholds`: 验证集校准得到的 `thresholds_val.json`，仅作用于二分类任务

**输出:**
- `evaluation_<split>/` - 评估结果目录
  - `eval_results.json` - JSON格式详细指标
  - `summary.png` - 所有任务的汇总图
  - `cm_*.png` - 各任务的混淆矩阵图

---

### calibrate_thresholds.py - 验证集阈值校准

对 v4 中的二分类任务在验证集上搜索正类概率阈值，适合用于改善 RVH、缺血相关任务这类少数类召回。

**用法:**
```bash
python scripts/calibrate_thresholds.py \
  --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

# 如果更关注阳性少数类召回和F1
python scripts/calibrate_thresholds.py \
  --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt \
  --objective positive_f1
```

**输出:**
- `thresholds_val.json` - 每个二分类任务的 0.5 默认指标、最佳阈值、阳性比例和验证集 F1

---

### heatmap.py - 注意力热力图

生成GradCAM注意力热力图，可视化模型关注的ECG区域。

**用法:**
```bash
# 默认参数
python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt

# 使用更粗的网格（更大区域）
python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --grid-size 6

# 使用更细的网格
python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --grid-size 12

# 指定样本
python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --sample-idx 100

# 调整透明度
python heatmap.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --alpha 0.7
```

**参数:**
- `--checkpoint`: 模型检查点路径 (必需)
- `--sample-idx`: 指定样本索引 (默认: 自动查找正确分类样本)
- `--grid-size`: 热力图网格大小 (默认: 8)
  - 6 = 更大区域
  - 8 = 默认
  - 12 = 更精细
- `--blur`: 高斯模糊核大小 (默认: 5)
- `--alpha`: 热力图透明度 0-1 (默认: 0.5)

**输出:**
- `heatmaps/` - 热力图目录
  - `gradcam_*.png` - 各步骤单独热力图
  - `gradcam_combined.png` - 6步组合热力图

---

### predict.py - 单样本推理

对单个ECG样本进行诊断预测。

**用法:**
```bash
# 使用模型进行预测
python predict.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt \
                    --signal_path /data/.../ecg.hea \
                    --image_path /data/.../ecg.png

# 保存结果到JSON
python predict.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt \
                    --signal_path /data/.../ecg.hea \
                    --image_path /data/.../ecg.png \
                    --output result.json

# 指定设备
python predict.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt \
                    --signal_path /data/.../ecg.hea \
                    --image_path /data/.../ecg.png \
                    --device cuda:1
```

**参数:**
- `--checkpoint`: 模型检查点路径 (必需)
- `--signal_path`: ECG信号文件路径 (.hea)
- `--image_path`: ECG图像文件路径 (.png)
- `--output`: 输出JSON文件路径 (可选)
- `--device`: 设备 (默认: cuda:0)

**输出格式:**
```json
{
  "predictions": {
    "rhythm_rate.rate_level": "Normal(60-100bpm)",
    "rhythm_rate.rhythm": "Sinus",
    ...
  }
}
```

## 通用技巧

### 查看训练日志
```bash
# 实时查看
tail -f outputs/ecg_diag/<timestamp>/logs/train.log

# 查看最后100行
tail -n 100 outputs/ecg_diag/<timestamp>/logs/train.log
```

### TensorBoard可视化
```bash
tensorboard --logdir outputs/ecg_diag/<timestamp>/tensorboard
```

### 检查模型性能
```bash
# 对比train/val/test性能
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split train
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split val
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt --split test

# 使用验证集阈值校准结果评估二分类任务
python evaluate.py --checkpoint outputs/ecg_diag/<timestamp>/checkpoints/best.pt \
  --split test \
  --thresholds outputs/ecg_diag/<timestamp>/thresholds_val.json
```

### 调试训练
```bash
# 使用小批量大小调试
torchrun --nproc_per_node=2 train.py --batch-size 16 --epochs 1

# 检查梯度
torchrun --nproc_per_node=2 train.py --batch-size 32 --amp
```

### 建议的第一轮模型实验顺序
```bash
# 1. 主实验：双模态 + 交叉注意力
torchrun --nproc_per_node=2 scripts/train.py --name v4_dual_cross

# 2. 信号单模态消融
torchrun --nproc_per_node=2 scripts/train.py --name v4_signal_only --input-mode signal

# 3. 图像单模态消融
torchrun --nproc_per_node=2 scripts/train.py --name v4_image_only --input-mode image

# 4. 简单后融合消融
torchrun --nproc_per_node=2 scripts/train.py --name v4_dual_late_concat --fusion-type late_concat
```
