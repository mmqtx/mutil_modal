# 优化日志

本文档用于记录 ECG 多模态诊断项目的实现修改、实验结果、验证结果和 Git 提交记录。除 commit message 外，日志内容统一使用中文。

## 2026-05-06

### 仓库整理

- 收紧 `.gitignore`，确保 `data/` 下的代码可以被跟踪，同时本地数据集、训练输出、热力图、checkpoint 和预训练权重不会进入 Git。
- 保留 `pretrained/README.md` 作为 `pretrained/` 目录下唯一跟踪文件，用来记录需要手动准备的外部权重。
- 新增 `DEVELOPMENT_RULES.md`，作为后续修改必须遵守的项目级协作规则。
- 修正规则文档中的结构化数据校验策略：结构化数据应先集中优化并冻结成稳定版本，后续模型结构、loss 和训练策略优化都固定在该数据版本上进行；不要求每次训练前重复校验结构化数据。
- 新增 `docs/data_analysis_v4/v4_vs_v5_candidate_analysis.md`，系统比较 v4 主标签和 v5 候选辅助标签思路，结论是继续使用 v4 作为主数据版本，v5 只作为后续辅助任务候选。

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
- `3005bd0` - `docs: localize project rules and optimization log`
- `ca6b9d8` - `docs: clarify structured data versioning rules`
- `df56fdf` - `docs: add v4 and v5 candidate data analysis`

## 2026-05-07

### 模型与方法优化基础设施

- 新增输入模态消融开关 `INPUT_MODE` / `--input-mode`，支持 `dual`、`signal`、`image` 三种模式，用于判断当前结果主要来自信号、图像还是两者融合。
- 新增融合方式开关 `FUSION_TYPE` / `--fusion-type`，保留默认 `cross_attention`，并新增轻量 `late_concat` 后融合基线，用于判断复杂交叉注意力是否真的优于简单融合。
- 修改模型前向逻辑：单模态消融时保持模型接口不变，跳过不用的 encoder 并屏蔽对应模态特征，同时自动关闭信号-图像全局对比损失，避免单模态实验被无意义的跨模态对比项干扰。
- 同步更新训练与离线评估脚本，使训练、验证、测试和消融评估使用一致的输入模态与融合配置。
- 新增 `scripts/calibrate_thresholds.py`，用于在验证集上为 v4 二分类任务搜索正类概率阈值，后续可重点改善 RVH、缺血相关任务等少数类的召回和 F1。
- 更新 `scripts/README.md`，记录第一轮建议实验顺序：双模态交叉注意力、信号单模态、图像单模态、双模态简单后融合。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py`：通过。
- `conda run -n pytorch python scripts/evaluate.py --help`：通过。
- `conda run -n pytorch python scripts/calibrate_thresholds.py --help`：通过。
- `conda run -n pytorch python scripts/train.py --help`：通过。

### 提交记录

- 待提交 - `feat: add multimodal ablation and threshold calibration`
