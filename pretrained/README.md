# 预训练模型权重

本目录用于存放预训练模型权重文件。

## 1. ECG Signal Encoder 权重 (必须)

从 GEM 项目下载 ECG-CoCa 预训练权重：
- 下载地址: https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB?usp=sharing
- 文件名: `cpt_wfep_epoch_20.pt`
- 放置路径: `pretrained/cpt_wfep_epoch_20.pt`

## 2. Image Encoder 权重 (必须)

使用 HuggingFace CLIP ViT-L/14@336px 预训练权重（与 GEM 一致）:
- 下载地址: https://huggingface.co/openai/clip-vit-large-patch14-336
- 下载整个模型目录，或程序会自动通过 transformers 加载
- 放置路径: `pretrained/clip-vit-large-patch14-336/`

## 目录结构

```
pretrained/
├── README.md                        # 本说明文件
├── cpt_wfep_epoch_20.pt             # GEM ECG-CoCa 预训练权重 (需手动下载)
└── clip-vit-large-patch14-336/      # CLIP ViT-L/14@336 权重 (需手动下载或自动下载)
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```
