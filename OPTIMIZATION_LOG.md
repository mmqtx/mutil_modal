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

- `1298ebc` - `feat: add multimodal ablation and threshold calibration`

### 实验启动记录

- 已启动完整主实验 `v4_dual_cross`：
  - 启动时间：2026-05-07 16:23
  - 命令：`torchrun --standalone --nproc_per_node=2 scripts/train.py --name v4_dual_cross`
  - run 目录：`outputs/v4_dual_cross/20260507_162349`
  - launcher 日志：`outputs/experiment_launcher/v4_dual_cross_20260507_162343.log`
  - 配置：v4 数据，双模态输入，`cross_attention` 融合，默认冻结信号/图像 encoder，默认关闭 CutMix 和 head contrastive，启用静态类别均衡 focal 权重。
  - 启动检查：成功加载 30000 条记录，train/val/test 为 21000/3000/6000；成功加载本地 CLIP 和 GEM 预训练权重；训练 step 已开始，未出现 OOM。

### 提交记录

- `d1cec5d` - `docs: record v4 dual cross experiment launch`

### 中断恢复与评估修正

- 服务器因 `/home` 空间不足中断后，确认 `v4_dual_cross` 训练进程已退出，日志停在 epoch 6；最佳 checkpoint 来自 epoch 5，验证集 macro-F1 为 0.7668。
- 使用 `outputs/v4_dual_cross/20260507_162349/checkpoints/best.pt` 做完整 test 评估，发现离线评估脚本漏报 v4 的 4 个 `ischemia_infarct` 二分类任务。
- 修复 `scripts/evaluate.py`，将 v4 缺血/梗死 4 个二分类任务纳入离线评估报告。
- 为 `scripts/evaluate.py` 新增 `--thresholds` 参数，可加载 `thresholds_val.json` 并对二分类任务应用验证集校准阈值。
- 使用未校准阈值的完整 15 任务 test 结果：
  - Overall Macro-F1：0.7779
  - 主要弱项：`voltage.rvh` 0.6189、`conduction_axis.pr_status` 0.6757、`qt_electrolytes.qt_status` 0.6868、`ischemia_infarct.st_elevation_present` 0.7031、`conduction_axis.conduction_status` 0.7044。
- 使用验证集阈值校准后，完整 15 任务 test Overall Macro-F1 提升到 0.7831；主要改善来自 `voltage.rvh`、`qt_electrolytes.qt_status`、`ischemia_infarct.st_depression_present` 和 `ischemia_infarct.t_wave_abnormal`。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py`：通过。
- `python scripts/evaluate.py --checkpoint outputs/v4_dual_cross/20260507_162349/checkpoints/best.pt --split test --output-dir outputs/v4_dual_cross/20260507_162349/evaluation_test_full_v4`：通过，生成 15 任务完整评估。
- `python scripts/calibrate_thresholds.py --checkpoint outputs/v4_dual_cross/20260507_162349/checkpoints/best.pt --objective macro_f1`：通过，生成 `thresholds_val.json`。
- `python scripts/evaluate.py --checkpoint outputs/v4_dual_cross/20260507_162349/checkpoints/best.pt --split test --thresholds outputs/v4_dual_cross/20260507_162349/thresholds_val.json --output-dir outputs/v4_dual_cross/20260507_162349/evaluation_test_full_v4_thresholded`：通过。

### 提交记录

- 待提交 - `fix: include v4 ischemia tasks in evaluation`

### 恢复训练与磁盘控制

- `/home` 空间不足导致训练中断后，清理了不必要的大模型产物：
  - 删除 `best_loss.pt`，保留按 macro-F1 选择的 `best.pt`；
  - 删除 smoke test run `outputs/v4_dual_cross_probe`；
  - 删除空 launcher 日志；
  - 删除原始 `v4_dual_cross` 的旧 `best.pt`，保留完整评估结果和阈值文件；当前继续使用 resume run 的 `best.pt`。
- 修改训练脚本：默认不再保存 `best_loss.pt`，新增 `--save-best-loss` 作为显式开关，避免后续实验再次占用额外约 2GB 空间。
- 从 `v4_dual_cross` 的最佳 checkpoint 恢复训练为 `v4_dual_cross_resume`，run 目录为 `outputs/v4_dual_cross_resume/20260507_175837`。
- 恢复训练在 epoch 7 达到最佳验证 macro-F1 0.7714，之后长期未刷新，判断为收益平台期，因此提前停止，保留 `best.pt`。

### 当前最佳结果

- `v4_dual_cross_resume/20260507_175837/checkpoints/best.pt` 未校准完整 15 任务 test：
  - Overall Macro-F1：0.7834
- 使用验证集阈值校准后的完整 15 任务 test：
  - Overall Macro-F1：0.7890
  - 主要弱项：`voltage.rvh` 0.6667、`conduction_axis.pr_status` 0.6925、`qt_electrolytes.qt_status` 0.7010、`ischemia_infarct.st_elevation_present` 0.7128、`conduction_axis.conduction_status` 0.7269。
- 阶段判断：继续同配置训练收益有限，下一步应做模态贡献判断，优先启动 `signal only` 实验，确认图像分支对弱项是帮助还是干扰。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py`：通过。
- `python scripts/evaluate.py --checkpoint outputs/v4_dual_cross_resume/20260507_175837/checkpoints/best.pt --split test --output-dir outputs/v4_dual_cross_resume/20260507_175837/evaluation_test_full_v4`：通过。
- `python scripts/calibrate_thresholds.py --checkpoint outputs/v4_dual_cross_resume/20260507_175837/checkpoints/best.pt --objective macro_f1`：通过，生成 `thresholds_val.json`。
- `python scripts/evaluate.py --checkpoint outputs/v4_dual_cross_resume/20260507_175837/checkpoints/best.pt --split test --thresholds outputs/v4_dual_cross_resume/20260507_175837/thresholds_val.json --output-dir outputs/v4_dual_cross_resume/20260507_175837/evaluation_test_full_v4_thresholded`：通过。

### 提交记录

- 待提交 - `chore: reduce checkpoint storage during training`
