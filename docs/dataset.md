# 数据集格式说明

## 数据集路径

- JSONL标注文件: `/data/ljq24358/ecg_dataset/ecg_jsons/structured_extraction/structured_labels_v2.jsonl`
- ECG图像根目录: `/data/ljq24358/ecg_dataset/ecg_images/`
- ECG信号文件: `/data/ljq24358/ecg_dataset/ecg_timeseries/`

## JSONL格式

每行一个JSON对象，包含以下字段：

```json
{
  "id": "45280303",
  "image_paths": ["relative/path/to/ecg.png"],
  "signal_path": "/path/to/ecg.hea",
  "structured_data": {
    "rhythm_rate": {
      "rate_level": "Normal(60-100bpm)",
      "rhythm": "Sinus"
    },
    "conduction_axis": {
      "axis": "Normal",
      "pr_status": "Normal",
      "qrs_width": "Narrow(<120ms)",
      "conduction_status": "Normal"
    },
    "voltage_hypertrophy": {
      "lvh": false,
      "rvh": false,
      "voltage": "Normal"
    },
    "ischemia_infarct": {
      "st_elevation": [],
      "st_depression": [],
      "t_wave_alt": [],
      "q_wave": []
    },
    "qt_electrolytes": {
      "qt_status": "Normal"
    },
    "summary_diag": {
      "is_abnormal": false,
      "primary_label": "Normal ECG"
    }
  },
  "original_cot": "..."
}
```

## 数据集类别统计

训练集 (27,000样本) 主要类别分布：

| 任务 | 类别 | 样本数 | 比例 |
|------|------|--------|------|
| rhythm.rhythm | Sinus | 22,800 | 84.4% |
| rhythm.rhythm | AFib | 2,213 | 8.2% |
| rhythm.rhythm | AFlutter | 389 | 1.4% |
| axis | Normal | 21,752 | 80.6% |
| axis | Extreme | 270 | 1.0% |
| rvh | 阴性 | 26,769 | 99.1% |
| rvh | 阳性 | 231 | 0.9% |

## 12导联标准

ECG使用标准12导联：I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

## 信号规格

- 采样率: 500 Hz
- 时长: 10秒
- 导联数: 12
- 数据格式: WFDB .hea 文件

## 图像规格

- 分辨率: 336×336 像素
- 格式: PNG
- 颜色: RGB
