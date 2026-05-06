# Optimization Log

This file records implementation changes, experiment results, and Git commits for the ECG multimodal diagnostic project.

## 2026-05-06

### Repository Hygiene

- Tightened `.gitignore` so code under `data/` remains trackable while local datasets, outputs, heatmaps, checkpoints, and pretrained model weights stay out of Git.
- Kept `pretrained/README.md` as the only tracked file under `pretrained/` to document required external weights.

### Training Pipeline Fixes

- Added deterministic split indices saved as `split_indices.json` in each run directory.
- Changed validation/test datasets to use `is_train=False` and disabled signal augmentation.
- Changed DDP validation/test flow so rank 0 evaluates the full validation/test split instead of a distributed shard.
- Added conservative training controls: frozen ECG/CLIP encoders by default, CutMix off by default, head contrastive off by default, modality dropout flag, and static class-balanced focal weights.
- Added `scripts/validate_structured_labels.py` for Qwen-extracted structured label checks.
- Removed hard-coded DashScope key from `scripts/predict.py`; Qwen calls now read `DASHSCOPE_API_KEY` from the environment and are disabled by default.

### Validation

- `conda run -n pytorch python -m py_compile config.py data/*.py models/*.py scripts/*.py`: passed.
- `conda run -n pytorch python scripts/validate_structured_labels.py --limit 1000`: passed with 0 errors, 0 missing signal files, and 0 missing image files.

### Metrics

- Baseline from existing run `outputs/ecg_diag/20260420_155424/checkpoints/best.pt`:
  - Test samples: 6000
  - Overall Macro-F1: 0.7437
  - Weak tasks to prioritize: `voltage.rvh`, `conduction_axis.pr_status`, `qt_electrolytes.qt_status`, `rhythm_rate.rhythm`

### Commits

- `b7bc864` - `chore: prepare repository ignore rules and optimization log`
- `39b6e62` - `fix: make dataset splits and evaluation deterministic`
- `266f93c` - `feat: add structured label validation and prediction docs`
