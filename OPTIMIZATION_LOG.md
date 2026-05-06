# Optimization Log

This file records implementation changes, experiment results, and Git commits for the ECG multimodal diagnostic project.

## 2026-05-06

### Repository Hygiene

- Tightened `.gitignore` so code under `data/` remains trackable while local datasets, outputs, heatmaps, checkpoints, and pretrained model weights stay out of Git.
- Kept `pretrained/README.md` as the only tracked file under `pretrained/` to document required external weights.

### Training Pipeline Fixes

- Planned fixes for deterministic split indices, validation/test transforms, full validation in DDP, conservative augmentation defaults, and resource-friendly encoder freezing.

### Metrics

- Baseline from existing run `outputs/ecg_diag/20260420_155424/checkpoints/best.pt`:
  - Test samples: 6000
  - Overall Macro-F1: 0.7437
  - Weak tasks to prioritize: `voltage.rvh`, `conduction_axis.pr_status`, `qt_electrolytes.qt_status`, `rhythm_rate.rhythm`

### Commit

- Pending initial repository commits.
