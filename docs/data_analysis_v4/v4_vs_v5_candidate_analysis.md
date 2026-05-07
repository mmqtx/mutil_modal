# v4 数据分析与 v5 候选优化对比报告

## 结论

当前 `v4` 数据版本已经可以作为后续模型与方法优化的主数据版本。  
不建议继续为了追求分布均匀而改动主标签体系，也不建议把临床含义不同的标签硬合并。

更合理的路线是：

- 主训练标签继续使用 `v4`。
- 保留所有原始结构化标签，保证诊断任务可解释。
- 把“v5 候选思路”作为辅助任务或分析维度，而不是替代 v4。
- 后续主要优化方向转向模型结构、loss、采样、阈值校准和多任务学习。

## v4 数据现状

### 基本情况

- 数据文件：`/data/ljq24358/ecg_dataset/ecg_jsons/structured_extraction/structured_labels_v4.jsonl`
- 样本数：30,000
- 数据来源：Qwen/API 从公开 ECG 数据中提取的结构化标签
- 输入模态：
  - ECG 12 导联时序信号
  - ECG 图像
  - 结构化诊断标签
  - 原始 CoT/报告文本仅作为参考，不直接作为训练目标

### v4 已经完成的关键优化

v4 相比 v2/v3 已经解决了最明显的数据稀疏问题：

- 心律标签从更细碎的 5 类合并到 3 类：
  - `AFib`
  - `Other_Arrhythmia`
  - `Sinus`
- 电轴去掉/合并了难学的 `Unknown`。
- PR 状态把 `Short + Unknown` 合并为 `Abnormal`。
- 传导状态删除极少数无效 `Other`。
- 缺血/梗死从区域级或导联级稀疏标签，变成 4 个二分类标签：
  - `st_elevation_present`
  - `st_depression_present`
  - `t_wave_abnormal`
  - `q_wave_present`

这些改动不是为了平均分布，而是为了让标签更稳定、更可学，同时保留临床含义。

## v4 关键标签分布

### 心律心率

| 标签 | 样本数 | 占比 |
|---|---:|---:|
| Normal rate | 21,778 | 72.6% |
| Slow rate | 4,181 | 13.9% |
| Fast rate | 4,041 | 13.5% |
| Sinus rhythm | 25,328 | 84.4% |
| AFib | 2,464 | 8.2% |
| Other_Arrhythmia | 2,208 | 7.4% |

判断：心率分布较好；心律仍有长尾，但 `AFib` 和 `Other_Arrhythmia` 样本量已经足够做多分类训练。

### 传导与电轴

| 标签 | 样本数 | 占比 |
|---|---:|---:|
| Normal axis | 24,218 | 80.7% |
| Left axis | 4,403 | 14.7% |
| Extreme axis | 859 | 2.9% |
| Right axis | 520 | 1.7% |
| PR Normal | 25,472 | 84.9% |
| PR Prolonged | 2,510 | 8.4% |
| PR Abnormal | 2,018 | 6.7% |
| QRS Wide | 5,750 | 19.2% |
| Conduction Block | 5,310 | 17.7% |
| Conduction Delay | 2,966 | 9.9% |

判断：`Right axis` 和 `Extreme axis` 偏少，但不能合并成一个方向未知的标签，因为左偏、右偏、极右/无人区电轴临床意义不同。更适合通过 loss 和辅助标签处理。

### 电压与肥厚

| 标签 | 样本数 | 占比 |
|---|---:|---:|
| LVH positive | 2,790 | 9.3% |
| RVH positive | 258 | 0.9% |
| Low voltage | 2,504 | 8.3% |

判断：`RVH` 是 v4 中最稀疏、最难学的标签。  
不建议把 RVH 和 LVH 直接合并替代原标签，因为二者临床含义不同。可以新增 `hypertrophy_present` 作为辅助任务，但原始 `lvh/rvh` 应保留。

### 缺血与梗死

| 标签 | 阳性数 | 阳性率 |
|---|---:|---:|
| ST elevation | 1,213 | 4.0% |
| ST depression | 3,583 | 11.9% |
| T wave abnormal | 9,177 | 30.6% |
| Q wave | 4,392 | 14.6% |

判断：v4 的缺血标签已经比 v2/v3 更可学。  
不建议只保留一个总的 `ischemia_any` 替代四个标签，因为 ST 抬高、ST 压低、T 波异常、Q 波代表不同临床证据。可以把 `ischemia_any` 和 `st_t_abnormal` 作为辅助标签。

### QT 与总体异常

| 标签 | 样本数 | 占比 |
|---|---:|---:|
| QT prolonged | 5,082 | 16.9% |
| Abnormal ECG | 22,350 | 74.5% |
| Normal ECG | 7,650 | 25.5% |

判断：QT 和总体异常标签分布可以接受。总体异常偏向异常样本，但符合临床数据分布，不需要强行平衡。

## v5 候选优化思路

这里的 `v5` 不是要替代 v4 的主标签体系，而是一个候选的“辅助标签层”思路。

### v5 候选派生标签

基于 v4 标签，可以自然派生出以下更稳定的辅助标签：

| 派生标签 | 阳性数 | 阳性率 | 含义 |
|---|---:|---:|---|
| `rate_abnormal` | 8,222 | 27.4% | 心率过快或过慢 |
| `rhythm_abnormal` | 4,672 | 15.6% | 非窦性心律 |
| `axis_abnormal` | 5,782 | 19.3% | 电轴非正常 |
| `pr_abnormal` | 4,528 | 15.1% | PR 非正常 |
| `qrs_wide` | 5,750 | 19.2% | QRS 增宽 |
| `conduction_abnormal` | 9,620 | 32.1% | PR/QRS/传导状态任一异常 |
| `hypertrophy_present` | 2,985 | 10.0% | LVH 或 RVH 任一阳性 |
| `voltage_or_hypertrophy_abnormal` | 5,397 | 18.0% | 低电压或室肥厚 |
| `ischemia_any` | 12,538 | 41.8% | 任一缺血/梗死证据阳性 |
| `st_t_abnormal` | 10,226 | 34.1% | ST/T 异常 |
| `qt_prolonged` | 5,082 | 16.9% | QT 延长 |

这些标签的优点是分布更稳定，可以帮助模型学习高层临床概念。

### primary_group 候选分组

开放式 `primary_label` 非常碎，不适合直接作为主分类任务。可以归并成高层诊断组：

| primary_group | 样本数 | 占比 |
|---|---:|---:|
| `normal_or_normal_variant` | 7,850 | 26.2% |
| `ischemia_or_st_t_abnormality` | 4,922 | 16.4% |
| `conduction_abnormality` | 4,821 | 16.1% |
| `rate_or_rhythm_abnormality` | 3,717 | 12.4% |
| `other_abnormal` | 2,952 | 9.8% |
| `atrial_fibrillation` | 2,455 | 8.2% |
| `hypertrophy_or_voltage` | 2,015 | 6.7% |
| `qt_abnormality` | 840 | 2.8% |
| `paced_rhythm` | 428 | 1.4% |

这个分组适合用于报告生成、辅助分类或结果分析，但不建议替代 v4 的结构化诊断链标签。

## v4 与 v5 候选对比

| 方面 | v4 主标签 | v5 候选辅助标签 |
|---|---|---|
| 定位 | 主训练数据版本 | 可选辅助任务/分析层 |
| 临床细粒度 | 更细，保留诊断含义 | 更粗，强调高层异常 |
| 分布稳定性 | 部分标签长尾明显 | 多数标签更均衡 |
| 可解释性 | 强，能对应具体诊断项 | 中等，适合概括 |
| 训练难度 | 对稀疏类更难 | 更容易学 |
| 风险 | RVH、Right axis 等长尾难学 | 可能损失细粒度诊断信息 |
| 建议 | 保留为主版本 | 只作为辅助，不替代 |

## 不建议做的优化

### 不建议合并 RVH 和 LVH 替代原标签

可以新增 `hypertrophy_present`，但不能删除或替代 `lvh/rvh`。  
原因是 RVH 和 LVH 临床含义不同，硬合并会让模型失去右室肥厚识别能力。

### 不建议合并电轴方向标签

`Left`、`Right`、`Extreme` 都是有方向意义的诊断信息。  
可以新增 `axis_abnormal` 辅助标签，但不应该替代原始多分类。

### 不建议只保留 `ischemia_any`

`ST elevation`、`ST depression`、`T wave abnormal`、`Q wave` 是不同证据链。  
可以新增 `ischemia_any` 或 `st_t_abnormal`，但原始四个标签应该继续保留。

### 不建议为了分布均匀删除正常样本

正常样本比例约 25.5%，并不过多。  
保留正常样本有利于模型学习“正常 ECG”的边界，不建议下采样破坏真实分布。

## 推荐决策

### 数据主版本

继续使用 `v4` 作为主数据版本。

理由：

- 样本量完整，30,000 条。
- 标签已经经过合理合并。
- 全量结构校验通过。
- 大部分任务样本量足够。
- 剩余长尾问题更适合用模型方法解决，而不是继续改标签。

### 后续优化方向

数据侧不再改主标签，转向方法侧：

- 类别不平衡 loss。
- 稀疏类采样策略。
- 多任务辅助 head。
- 高层辅助标签，例如 `hypertrophy_present`、`ischemia_any`、`conduction_abnormal`。
- 阈值校准和任务级权重。
- 模态缺失鲁棒性。
- 信号/图像融合结构优化。

## 最终建议

数据部分到这里可以阶段性收口。  
后续不要再频繁改数据版本，否则模型实验无法稳定比较。

如果之后确实要做 `v5`，建议它不是“替代 v4”，而是：

- 保留 v4 所有原始标签；
- 额外增加派生辅助标签；
- 用于多任务辅助监督或报告生成；
- 不作为当前第一轮模型优化的必要前提。
