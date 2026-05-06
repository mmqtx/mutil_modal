# Development Rules

This document defines the project-level working rules for future changes. Update it whenever we agree on a better workflow.

## Core Workflow

- Read the relevant code and current Git state before editing.
- Keep each change focused on one purpose.
- Prefer existing project patterns over new abstractions.
- Do not rewrite unrelated code while fixing a specific issue.
- Do not remove user-created files or changes unless explicitly requested.

## Environment

- Use the local Anaconda environment named `pytorch` for checks, scripts, and smoke tests.
- Prefer commands in this form:

```bash
conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py
```

- Use CUDA/DDP commands only when the task requires training or GPU validation.

## Data And Artifacts

- Never commit external datasets under `/data/ljq24358/ecg_dataset`.
- Never commit pretrained weights, checkpoints, generated outputs, heatmaps, logs, caches, or prediction dumps.
- Keep `pretrained/README.md` tracked so required external weights are documented.
- If a new generated directory appears, update `.gitignore` before committing.

## Training And Evaluation

- Validation and test datasets must use `is_train=False` and must not use training augmentation.
- Training/validation/test splits must be deterministic and saved when a run is created.
- In DDP training, validation/test metrics must cover the full split, not only one distributed shard.
- Default training should remain resource-friendly for `2 * RTX 4090`:
  - freeze large encoders by default,
  - keep signal CutMix off by default,
  - keep head contrastive off by default,
  - enable heavier options only as explicit ablations.

## Structured Labels

- Qwen-extracted structured labels must be validated before being used for training.
- Use `scripts/validate_structured_labels.py` to check schema, legal labels, file existence, and class distribution.
- Do not train directly on hidden CoT text as a target.
- Natural-language reports should be grounded in `structured_report` predictions to reduce hallucination.

## Secrets

- Do not hard-code API keys, tokens, passwords, or private credentials.
- Qwen/DashScope calls must read `DASHSCOPE_API_KEY` from the environment.
- If a secret is found in tracked code, remove it before committing.

## Testing Before Commit

At minimum, run:

```bash
conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py
```

For data-pipeline changes, also run:

```bash
conda run -n pytorch python scripts/validate_structured_labels.py --limit 1000
```

For training changes, run a small smoke test when practical before full training.

## Optimization Log

- Update `OPTIMIZATION_LOG.md` for meaningful changes.
- Record:
  - date,
  - problem addressed,
  - files or modules changed,
  - validation commands and results,
  - commit hash after committing.

## Git Rules

- Check `git status --short --ignored` before staging.
- Split commits by intent when practical.
- Use clear commit messages, for example:
  - `fix: make dataset splits deterministic`
  - `feat: add structured label validation`
  - `docs: update development rules`
- Push to `origin main` after successful local validation unless the user asks not to.

## Review Checklist

Before finishing a change, confirm:

- No large model/data/output files are staged.
- No API keys or secrets are present.
- Checks were run in the `pytorch` conda environment.
- `OPTIMIZATION_LOG.md` was updated when the change affects behavior or workflow.
- Local commits are pushed or the reason for not pushing is clearly documented.
