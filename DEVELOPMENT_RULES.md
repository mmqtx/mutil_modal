# 开发规则

本文档记录本项目后续修改必须遵守的协作规则。之后如果我们形成新的约定，就继续补充和优化这个文件。

## 基本工作流

- 修改前先阅读相关代码，并检查当前 Git 状态。
- 每次修改尽量只解决一个明确问题。
- 优先沿用项目已有写法和目录结构，不轻易引入新抽象。
- 修复某个问题时，不顺手重写无关代码。
- 不删除用户创建的文件或修改，除非用户明确要求。

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

## 训练和评估

- 验证集和测试集必须使用 `is_train=False`，不能使用训练增强。
- 训练/验证/测试划分必须可复现，并在创建训练 run 时保存。
- DDP 训练中，验证/测试指标必须覆盖完整 split，不能只评估单个分片。
- 默认训练策略要适配 `2 * RTX 4090` 的资源限制：
  - 默认冻结大 encoder；
  - 默认关闭 signal CutMix；
  - 默认关闭 head contrastive；
  - 更重的训练选项只作为显式消融实验开启。

## 结构化标签

- Qwen 提取的结构化标签用于训练前必须先校验。
- 使用 `scripts/validate_structured_labels.py` 检查字段结构、合法标签、文件存在性和类别分布。
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

如果修改了数据管线，还要运行：

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
