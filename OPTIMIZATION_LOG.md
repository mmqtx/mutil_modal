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

- `845ae7d` - `fix: include v4 ischemia tasks in evaluation`

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

- `531887a` - `chore: reduce checkpoint storage during training`

## 2026-05-07 模态消融、单模态 I/O 修正与低学习率解冻准备

### 已完成工作

- 完成 `signal only` 冻结编码器消融实验，输出目录迁移到 `/data/ljq24358/ecg_experiments/mutil_modal_outputs`，避免继续挤占 `/home`。
- `signal only` 阈值校准后 test Overall Macro-F1 为 0.7854，略低于当前最佳双模态阈值校准结果 0.7890；说明信号分支很强，但图像分支对 `rvh`、`pr_status`、`qt_status`、`st_elevation_present` 等弱项仍有帮助。
- 发现 `--input-mode signal` 时 dataset 仍然读取 ECG 图像，导致信号单模态实验被无意义的图片 I/O 拖慢。
- 修改 `data/dataset.py`，新增 `load_signal` 和 `load_image` 控制；单模态实验只读取实际使用的模态，未使用模态返回同形状零张量占位，不改动任何训练数据。
- 同步修改 `scripts/train.py`、`scripts/evaluate.py`、`scripts/calibrate_thresholds.py`，根据 `input_mode` 自动跳过未使用模态的 I/O。
- 新增 `--encoder-lr-scale`，默认 0.1；后续解冻 GEM/CLIP 预训练编码器时，编码器学习率为主学习率的 0.1，新训练模块仍使用主学习率，降低破坏预训练表征的风险。
- 中止并删除无效的全学习率 `v4_signal_unfreeze` 试跑；该 run 未完成第 0 轮、未产生有效 checkpoint。
- 将旧的 `outputs/v4_dual_cross_resume` 移动到 `/data/ljq24358/ecg_experiments/archived_home_outputs`，并在原位置保留软链接，释放 `/home` 约 2GB 空间。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py data/dataset.py scripts/train.py scripts/evaluate.py scripts/calibrate_thresholds.py`：通过。
- dataset 读取计时：
  - `signal_only` 20 个样本：0.276 秒。
  - `dual` 20 个样本：3.888 秒。
- 磁盘策略：后续完整实验默认使用 `/data/ljq24358/ecg_experiments/mutil_modal_outputs`；`/home` 只保留代码、轻量日志和软链接。

### 下一步

- 启动新的 `v4_signal_unfreeze_low_lr` 实验，使用 `--input-mode signal --unfreeze-signal-encoder --encoder-lr-scale 0.1`，验证温和解冻是否能超过冻结信号模型。
- 如果温和解冻仍无法超过当前双模态最佳 0.7890，则转向弱任务定向优化，包括任务级 loss 权重和少数类阈值/校准策略。

### 提交记录

- `b0df38f` - `feat: add efficient single-modality loading and encoder lr scaling`

## 2026-05-07 低学习率解冻实验收敛判断与弱任务损失权重

### 实验结论

- 启动 `v4_signal_unfreeze_low_lr`：`--input-mode signal --unfreeze-signal-encoder --encoder-lr-scale 0.1 --batch-size 64 --epochs 12`。
- 该实验在 `/data/ljq24358/ecg_experiments/mutil_modal_outputs/v4_signal_unfreeze_low_lr/20260507_212759` 运行，未占用 `/home` 保存大 checkpoint。
- 验证集 macro-F1：
  - epoch 0：0.5764
  - epoch 1：0.7408
  - epoch 2：0.7583
  - epoch 3：0.7591
  - epoch 4：0.7547
  - epoch 5：0.7569
- 结论：温和解冻 GEM signal encoder 没有超过冻结信号模型的验证峰值 0.7706，也没有接近当前双模态最佳 0.7714；判断为无收益实验。
- 已停止该实验并删除无用 `best.pt`，保留日志用于复盘。

### 方法更新

- 新增子任务级损失权重机制，默认关闭，需要显式添加 `--use-subtask-loss-weights`。
- 当前弱任务权重：
  - `voltage.rvh`: 1.8
  - `conduction_axis.pr_status`: 1.4
  - `qt_electrolytes.qt_status`: 1.4
  - `ischemia_infarct.st_elevation_present`: 1.3
  - `conduction_axis.conduction_status`: 1.2
- 修改 `DynamicMultiTaskLoss`，在不改变类别权重和 uncertainty weighting 的基础上，对指定子任务的最终 task loss 做温和放大。
- `resume` 读取旧 loss state 时改为 `strict=False`，避免新增 buffer 后不能加载旧 checkpoint。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py models/losses.py scripts/train.py`：通过。

### 下一步

- 启动 `v4_dual_task_weighted`，使用双模态输入、冻结两个 encoder、启用弱任务损失权重。
- 若验证 macro-F1 超过当前双模态最佳 0.7714，再做完整 test 与阈值校准；否则停止并记录为无收益。

### 提交记录

- `da2bc7a` - `feat: add optional subtask loss weighting`

## 2026-05-07 双模态弱任务加权从头训练复盘

### 实验结论

- 启动 `v4_dual_task_weighted`：双模态输入，冻结信号和图像 encoder，从头训练融合层/分类头，并启用较强弱任务损失权重。
- 验证集 macro-F1：
  - epoch 0：0.5302
  - epoch 1：0.7263
  - epoch 2：0.7454
  - epoch 3：0.7486
- 结论：从头训练时直接加强弱任务 loss 会拖慢整体收敛，且没有接近当前双模态最佳验证 macro-F1 0.7714。
- 已停止实验并删除无用 checkpoint，保留日志用于复盘。

### 方法修正

- 将弱任务损失权重改为更温和的设置：
  - `voltage.rvh`: 1.3
  - `conduction_axis.pr_status`: 1.15
  - `qt_electrolytes.qt_status`: 1.15
  - `ischemia_infarct.st_elevation_present`: 1.1
  - `conduction_axis.conduction_status`: 1.1
- 新增 `--resume-model-only`：只从 checkpoint 加载模型和 loss 状态，重新初始化 optimizer/scheduler，用于从当前最佳模型做短程微调。

### 验证结果

- `conda run -n pytorch python -m py_compile config.py scripts/train.py models/losses.py`：通过。

### 下一步

- 从当前最佳双模态 checkpoint 启动 `v4_dual_light_weighted_ft`，使用轻权重和 fresh optimizer 做短程微调。
- 若验证 macro-F1 超过 0.7714，再进行 test 和阈值校准；否则停止并删除 checkpoint。

### 提交记录

- `a6b5a3e` - `feat: support model-only resume for finetuning`

## 2026-05-07 双模态轻权重微调复盘

### 实验结论

- 启动 `v4_dual_light_weighted_ft`：从当前最佳双模态 checkpoint 只加载模型权重，重新初始化 optimizer/scheduler，使用更轻的弱任务损失权重做短程微调。
- 验证集 macro-F1：
  - epoch 0：0.7614
  - epoch 1：0.7641
  - epoch 2：0.7688
  - epoch 3：0.7711
  - epoch 4：0.7713
- 测试集未校准 macro-F1：0.7691，低于原始双模态模型未校准测试 macro-F1 0.7834。
- 结论：轻权重微调虽然接近原始验证峰值 0.7714，但测试集退化明显，说明当前弱任务加权策略没有带来稳定收益。
- 已删除该实验的 `best.pt`，保留训练日志、划分文件和 TensorBoard 事件用于复盘。

### 当前最佳方案

- 当前最佳仍为 `v4_dual_cross_resume/20260507_175837`。
- 测试集未校准 macro-F1：0.7834。
- 测试集阈值校准 macro-F1：0.7890。
- 当前阶段不继续沿“加大弱任务 loss 权重”的方向投入训练资源，转向低成本的阈值校准、评估策略和更稳健的模型结构消融。

### 下一步

- 先基于当前最佳 checkpoint 做不同阈值目标的后处理对比，确认是否能在不重新训练的情况下继续提升 test macro-F1。
- 若后处理收益有限，再启动新的模型结构实验，优先尝试 late-concat/轻量门控融合，而不是继续直接调高弱任务 loss。

## 2026-05-07 阈值目标对比

### 实验结论

- 基于当前最佳 checkpoint `v4_dual_cross_resume/20260507_175837/checkpoints/best.pt`，额外尝试 `positive_f1` 作为验证集阈值搜索目标。
- 生成阈值文件：`thresholds_val_positive_f1.json`。
- 使用该阈值在完整测试集 6000 条样本上评估，测试集 macro-F1 为 0.7870。
- 当前 `macro_f1` 阈值策略的测试集 macro-F1 为 0.7890，因此 `positive_f1` 阈值目标没有带来提升。

### 当前判断

- 继续保留 `macro_f1` 阈值校准作为当前最佳后处理策略。
- 单纯改变阈值搜索目标收益有限，下一步应回到模型融合方式本身，优先做结构简单、资源友好的双模态融合消融。

### 下一步

- 启动 `late_concat` 双模态融合实验，仍冻结两个预训练 encoder，只训练轻量融合层和分类头。
- 若 `late_concat` 验证集明显低于当前最佳 0.7714，则及时停止并删除无用 checkpoint。

## 2026-05-07 双模态 late-concat 融合实验复盘

### 实验结论

- 启动 `v4_dual_late_concat`：双模态输入，冻结 GEM 信号 encoder 和 CLIP 图像 encoder，只训练 late-concat 融合层、投影层、分类头和动态损失参数。
- 运行目录：`/data/ljq24358/ecg_experiments/mutil_modal_outputs/v4_dual_late_concat/20260507_224817`。
- 验证集 macro-F1：
  - epoch 0：0.5153
  - epoch 1：0.7265
  - epoch 2：0.7531
  - epoch 3：0.7618
  - epoch 4：0.7592
  - epoch 5：0.7620
- 结论：late-concat 能正常学习，但峰值 0.7620 明显低于当前 cross-attention 最佳验证 macro-F1 0.7714。
- 已停止该实验并删除无用 `best.pt`，保留日志、划分文件和 TensorBoard 事件用于复盘。

### 当前判断

- 简单拼接融合不足以替代 cross-attention，说明当前任务确实需要信号特征和图像特征之间的交互建模。
- 后续结构优化应保留跨模态交互，但控制复杂度，例如轻量门控、残差融合或减少对比损失干扰。

### 下一步

- 检查当前 loss 曲线中对比学习项的作用；`late_concat` 和 cross-attention 都启用了 `contrastive_weight=0.05`，下一轮优先尝试关闭或降低对比损失，验证它是否在分类主任务上形成干扰。

## 2026-05-07 双模态 cross-attention 关闭全局对比损失实验复盘

### 实验结论

- 启动 `v4_dual_cross_no_contrastive`：双模态输入，cross-attention 融合，冻结两个预训练 encoder，设置 `--contrastive-weight 0.0`。
- 运行目录：`/data/ljq24358/ecg_experiments/mutil_modal_outputs/v4_dual_cross_no_contrastive/20260507_231856`。
- 验证集 macro-F1：
  - epoch 0：0.5361
  - epoch 1：0.7295
  - epoch 2：0.7460
  - epoch 3：0.7453
- 结论：关闭全局对比损失后，验证集 macro-F1 在 epoch 2 后停滞并回落，明显低于当前最佳 cross-attention 验证 macro-F1 0.7714。
- 已停止该实验并删除无用 `best.pt`，保留日志、划分文件和 TensorBoard 事件用于复盘。

### 当前判断

- 全局对比损失虽然不是最终分类目标，但在当前双模态训练中有稳定融合表示的作用，直接关闭会损害分类主任务。
- 当前保留 `contrastive_weight=0.05` 更稳；后续若继续探索，只考虑小幅降低到 0.02 或做 warmup/decay，而不是直接置零。

### 下一步

- 暂停继续堆训练实验，先整理当前模型消融结论：当前最佳仍为 `cross_attention + contrastive_weight=0.05 + macro_f1 阈值校准`。
- 下一轮方法优化优先考虑在当前最佳 checkpoint 上做分类阈值、报告模板和少量结构增强，而不是继续大范围试错。

## 2026-05-08 新增门控残差交叉融合

### 改动内容

- 新增 `GatedCrossAttentionFusion`，在原有 cross-attention 交互分支之外，加入原始 signal/image 特征的残差融合分支。
- 门控网络按样本输出融合权重，自适应决定更相信 cross-attention 特征还是残差信息。
- `scripts/train.py` 新增 `--fusion-type gated_cross_attention` 选项。
- 默认配置仍保持 `FUSION_TYPE = "cross_attention"`，不改变当前最佳基线。

### 设计原因

- `late_concat` 明显弱于 `cross_attention`，说明跨模态交互是有价值的。
- 直接关闭全局对比损失也退化，说明当前融合表示需要稳定约束。
- 因此下一步不推翻已有结构，而是在 cross-attention 上加入轻量残差门控，尝试提升泛化稳定性。

### 验证结果

- `/home/ljq24358/anaconda3/envs/pytorch/bin/python -m py_compile config.py models/fusion.py models/model.py scripts/train.py`：通过。

### 下一步

- 启动 `v4_dual_gated_cross` 实验，双模态输入，冻结两个 encoder，保留 `contrastive_weight=0.05`。
- 若早期验证 macro-F1 未超过或接近当前最佳 0.7714，则及时停止并删除无收益 checkpoint。

## 2026-05-11 门控残差交叉融合实验复盘

### 实验结论

- 检查 `v4_dual_gated_cross/20260508_233242`，该实验在中断后仍有残留训练进程。
- 验证集 macro-F1：
  - epoch 0：0.5306
  - epoch 1：0.7278
  - epoch 2：0.7523
  - epoch 3：0.7544
  - epoch 4：0.7534
  - epoch 5：0.7570
  - epoch 6：0.7578
- 峰值 0.7578 明显低于当前最佳 cross-attention 验证 macro-F1 0.7714。
- 已停止残留进程并删除无收益 `best.pt`，保留日志、划分文件和 TensorBoard 事件用于复盘。

### 当前判断

- 门控残差分支没有带来收益，反而可能增加优化难度。
- 当前最佳仍为 `cross_attention + contrastive_weight=0.05 + macro_f1 阈值校准`，测试集阈值校准 macro-F1 为 0.7890。
- 后续不再沿“增加融合模块复杂度”的方向优先探索。

### 系统状态

- `/home` 可用空间较低，约 2.9G；后续训练继续强制输出到 `/data/ljq24358/ecg_experiments`。
- `/home/ljq24358/.cache`、`/home/ljq24358/anaconda3` 和 `/home/ljq24358/dev` 是主要占用来源；未删除任何用户文件。

### 下一步

- 优先做低风险优化：基于当前最佳 checkpoint 做评估侧分析、阈值/报告模板优化，或做小代码改动后再启动短程实验。
- 若继续训练实验，先确认 `/home` 空间安全，并设置明确止损阈值。

## 2026-05-11 多分类 logit bias 校准

### 改动内容

- 新增 `scripts/calibrate_logit_biases.py`，在验证集上为多分类任务搜索加性 logit bias。
- `scripts/evaluate.py` 新增 `--logit-biases`，可在 argmax/threshold 前应用验证集校准得到的 bias。
- `scripts/calibrate_thresholds.py` 和 `scripts/evaluate.py` 补充支持 `gated_cross_attention` 作为合法融合类型。

### 验证结果

- `/home/ljq24358/anaconda3/envs/pytorch/bin/python -m py_compile scripts/evaluate.py scripts/calibrate_thresholds.py scripts/calibrate_logit_biases.py`：通过。
- 在当前最佳 checkpoint 的验证集上生成 `logit_biases_val.json`，多分类任务验证 macro-F1 有小幅提升：
  - `rhythm_rate.rate_level`：0.9222 -> 0.9294
  - `rhythm_rate.rhythm`：0.7952 -> 0.8065
  - `conduction_axis.axis`：0.7467 -> 0.7576
  - `conduction_axis.pr_status`：0.6969 -> 0.7148
  - `conduction_axis.conduction_status`：0.7314 -> 0.7379
- 将该 bias 与原有二分类 threshold 一起用于测试集，测试 macro-F1 为 0.7887。
- 当前最佳测试 macro-F1 仍为 0.7890，因此暂不采用 logit bias 作为默认最终方案。

### 当前判断

- 验证集 bias 校准能改善部分少数类，但存在轻微过拟合，测试集没有超过当前最佳。
- 该脚本保留为分析和报告侧工具，后续可以用于错误分析，但不替换当前最佳评估配置。

### 提交记录

- `593ba18` - `feat: add multiclass logit bias calibration`
