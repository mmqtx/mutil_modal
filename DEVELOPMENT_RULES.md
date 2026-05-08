# 开发规则

本文档记录本项目后续修改必须遵守的协作规则。之后如果我们形成新的约定，就继续补充和优化这个文件。

## 基本工作流

- 修改前先阅读相关代码，并检查当前 Git 状态。
- 每次修改尽量只解决一个明确问题。
- 优先沿用项目已有写法和目录结构，不轻易引入新抽象。
- 修复某个问题时，不顺手重写无关代码。
- 不删除用户创建的文件或修改，除非用户明确要求。
- 任何实验、清理或重构都不能删除原始训练数据、结构化数据、论文资料和用户手动创建的项目文件。
- 测试生成的 checkpoint、日志、缓存、临时评估结果等可再生文件，可以在确认无收益后清理，但清理前必须确认路径位于实验输出目录或缓存目录。
- 连续实验时保持轻量心跳检查：关注训练进程、验证指标、GPU 状态和磁盘空间，根据结果决定继续、停止或切换方向。
- 为节省对话资源，状态更新保持简短，只报告关键指标、判断和下一步动作。

## 语言规则

- `DEVELOPMENT_RULES.md` 和 `OPTIMIZATION_LOG.md` 必须使用中文书写。
- 每次更新 `OPTIMIZATION_LOG.md` 时，记录内容必须使用中文。
- Git commit message 使用英文，保持简短清晰。
- 代码中的说明性注释、脚本帮助文本和项目内文档默认使用中文；只有库接口名、参数名、报错原文或第三方约定需要英文时才保留英文。

## 环境

- 检查、脚本和 smoke test 默认使用本地 Anaconda 环境 `pytorch`。
- 优先使用下面这种命令形式：

```bash
conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py
```

- 只有在需要训练或 GPU 验证时才运行 CUDA/DDP 命令。

## 数据和产物

- 不提交 `/data/ljq24358/ecg_dataset` 下的外部数据集。
- 不提交预训练权重、checkpoint、训练输出、热力图、日志、缓存或预测结果文件。
- 保留 `pretrained/README.md`，用来说明需要手动准备哪些外部权重。
- 如果出现新的生成目录，提交前先更新 `.gitignore`。
- Ubuntu 系统盘 `/home` 空间紧张，禁止把大模型、checkpoint、完整实验输出或大规模中间结果写入项目目录或系统盘。
- 完整训练、消融实验、评估输出和可再生大文件统一写入 `/data/ljq24358/ecg_experiments`。
- 启动训练前后都要关注 `df -h /home /data`，如果 `/home` 可用空间过低，先清理可再生输出或暂停实验。

## 训练和评估

- 验证集和测试集必须使用 `is_train=False`，不能使用训练增强。
- 训练/验证/测试划分必须可复现，并在创建训练 run 时保存。
- DDP 训练中，验证/测试指标必须覆盖完整 split，不能只评估单个分片。
- 长时间训练只保留必要的监听会话，避免打开过多后台进程或交互进程。
- 若实验早期指标明显落后当前最佳，并且趋势已经停滞，应及时停止并删除无收益 checkpoint，保留日志用于复盘。
- 默认训练策略要适配 `2 * RTX 4090` 的资源限制：
  - 默认冻结大 encoder；
  - 默认关闭 signal CutMix；
  - 默认关闭 head contrastive；
  - 更重的训练选项只作为显式消融实验开启。

## 结构化数据版本

- 结构化数据只在“创建、清洗、合并、重标注或冻结数据版本”时集中校验，不要求每次训练前重复校验。
- 当前目标是尽可能优化已有 Qwen 结构化数据，保证字段合法、类别分布合理、缺失文件最少，形成一个稳定的最佳数据版本。
- 一旦确定最佳数据版本，后续模型结构、loss、训练策略和消融实验都必须固定在这一版数据上进行，避免数据变化干扰实验可比性。
- 如果确实需要修改结构化数据，必须视为新的数据版本，并记录修改原因、分布变化和验证结果。
- 使用 `scripts/validate_structured_labels.py` 检查字段结构、合法标签、文件存在性和类别分布；该脚本用于数据版本治理，不是每次训练的必跑步骤。
- 不直接把隐藏 CoT 文本作为训练目标。
- 自然语言报告应基于 `structured_report` 预测结果生成，减少幻觉。

## 密钥

- 不硬编码 API key、token、密码或其他私密凭据。
- Qwen/DashScope 调用必须从环境变量 `DASHSCOPE_API_KEY` 读取密钥。
- 如果在已跟踪代码中发现密钥，提交前必须移除。

## 提交前检查

至少运行：

```bash
conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py
```

如果创建、修改或冻结结构化数据版本，还要运行：

```bash
conda run -n pytorch python scripts/validate_structured_labels.py --limit 1000
```

如果修改了训练逻辑，在条件允许时先跑小规模 smoke test，再启动完整训练。

## 优化日志

- 有实际行为变化、训练策略变化、数据处理变化或协作规则变化时，必须更新 `OPTIMIZATION_LOG.md`。
- 优化日志必须使用中文记录。
- 记录内容包括：
  - 日期；
  - 解决的问题；
  - 修改的文件或模块；
  - 验证命令和结果；
  - 提交后的 commit hash。

## Git 规则

- staging 前检查 `git status --short --ignored`。
- 尽量按修改意图拆分 commit。
- commit message 使用英文，例如：
  - `fix: make dataset splits deterministic`
  - `feat: add structured label validation`
  - `docs: update development rules`
- 本地验证通过后推送到 `origin main`，除非用户明确要求不要推送。

## 完成前检查清单

结束一次修改前确认：

- 没有 staging 大模型、数据集、输出目录或缓存文件。
- 没有 API key 或其他密钥。
- 检查命令使用 `pytorch` conda 环境运行。
- 如果修改影响行为或工作流，已更新 `OPTIMIZATION_LOG.md`。
- 本地 commit 已推送；如果没有推送，需要清楚说明原因。
