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
- `--lr`: 学习率 (默认: 2e-4)
- `--contrastive-weight`: 对比学习权重 (默认: 0.1)
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

**输出:**
- `evaluation_<split>/` - 评估结果目录
  - `eval_results.json` - JSON格式详细指标
  - `summary.png` - 所有任务的汇总图
  - `cm_*.png` - 各任务的混淆矩阵图

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
```

### 调试训练
```bash
# 使用小批量大小调试
torchrun --nproc_per_node=2 train.py --batch-size 16 --epochs 1

# 检查梯度
torchrun --nproc_per_node=2 train.py --batch-size 32 --amp
```
