# 优化日志

本文档用于记录 ECG 多模态诊断项目的实现修改、实验结果、验证结果和 Git 提交记录。除 commit message 外，日志内容统一使用中文。

## 2026-05-06

### 仓库整理

- 收紧 `.gitignore`，确保 `data/` 下的代码可以被跟踪，同时本地数据集、训练输出、热力图、checkpoint 和预训练权重不会进入 Git。
- 保留 `pretrained/README.md` 作为 `pretrained/` 目录下唯一跟踪文件，用来记录需要手动准备的外部权重。
- 新增 `DEVELOPMENT_RULES.md`，作为后续修改必须遵守的项目级协作规则。

### 训练管线修复

- 新增确定性的 split 索引，并在每个训练 run 目录保存为 `split_indices.json`。
- 验证集和测试集改为使用 `is_train=False`，并禁用 signal augmentation。
- 修改 DDP 验证/测试流程，让 rank 0 评估完整验证/测试集，而不是只评估 distributed shard。
- 新增保守训练控制：默认冻结 ECG/CLIP encoder，默认关闭 CutMix，默认关闭 head contrastive，增加 modality dropout 参数，并加入静态类别均衡 focal 权重。
- 新增 `scripts/validate_structured_labels.py`，用于校验 Qwen 提取的结构化标签。
- 移除 `scripts/predict.py` 中硬编码的 DashScope key；Qwen 调用改为从环境变量 `DASHSCOPE_API_KEY` 读取，并默认关闭。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py`：通过。
- `conda run -n pytorch python scripts/validate_structured_labels.py --limit 1000`：通过，0 个错误，0 个缺失 signal 文件，0 个缺失 image 文件。

### 当前基线指标

- 现有 run `outputs/ecg_diag/20260420_155424/checkpoints/best.pt` 的基线结果：
  - 测试样本数：6000
  - Overall Macro-F1：0.7437
  - 后续优先优化的弱任务：`voltage.rvh`、`conduction_axis.pr_status`、`qt_electrolytes.qt_status`、`rhythm_rate.rhythm`

### 提交记录

- `b7bc864` - `chore: prepare repository ignore rules and optimization log`
- `39b6e62` - `fix: make dataset splits and evaluation deterministic`
- `266f93c` - `feat: add structured label validation and prediction docs`
- `575e053` - `docs: add project development rules`
- 待提交 - `docs: localize project rules and optimization log`
