# MDT-AH — Modality Discrepancy Transformer for Ambivalence and Hesitancy Recognition

[中文文档](./README_zh.md)

**11th ABAW Competition: MTL Challenge Team Registration**

**Team Name:** CASIA-26

**Lead Researcher:**
- Bin Liu (liubin@nlpr.ia.ac.cn)

**Team Members:**
- Shiyu Luo (luoshiyu221@mails.ucas.ac.cn)
- Yu Wang (wangyu230@mails.ucas.ac.cn)
- Chenxi Huang (huangchenqian22@mails.ucas.ac.cn)
- Jiawen Huang (huangjiawen25@mails.ucas.ac.cn)
- Qi Zhang (zhangqi2025@ia.ac.cn)
- Zhaoxiang Xiao (xuegaodef@163.com)

**Contact Members:**
- Jiawen Huang (huangjiawen25@mails.ucas.ac.cn)
- Qi Zhang (zhangqi2025@ia.ac.cn)

---

> **Note:** This code is built upon and modified from [ConflictAwareAH](https://github.com/sbelharbi/bah-dataset) (Bekhouche et al., 2026):
>
> ```bibtex
> @inproceedings{conflictawareah2026,
>   title={Conflict-Aware Multimodal Fusion for Ambivalence and Hesitancy Recognition},
>   author={Bekhouche, Salah Eddine and Telli, Hichem and Benlamoudi, Azeddine and Herrouz, Salah Eddine and Taleb-Ahmed, Abdelmalik and Hadid, Abdenour},
>   year={2026}
> }
> ```

---

Code for the ABAW11 Ambivalence/Hesitancy Challenge. MDT enriches the original 6-token conflict-aware design to a 9-token representation — three modality embeddings, three absolute-difference features, and three Hadamard-product discrepancy features — processed by a Transformer with FiLM-based text-conditioned modulation and LoRA fine-tuning.

## Results

| Split | Macro F1 | F1 (A-H) | F1 (No-AH) | Accuracy |
|-------|----------|----------|------------|----------|
| Labelled Test (525 videos) | **0.7408** | 0.7857 | 0.6959 | 0.7486 |
| Private Leaderboard (151 videos) | **0.7368** | 0.7887 | — | — |

## Best Configuration

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
    --use_lora --lora_r 8 --lora_alpha 16 \
    --conflict_type both \
    --use_film \
    --focal_gamma 2.0 \
    --use_cutmix \
    --warmup_epochs 5 \
    --num_windows 3 

python scripts/predict.py \
    --checkpoints outputs/runs/<RUN_TIMESTAMP>/best_model.pt \
    --split test_unlabeled --num_windows 5 --output outputs/submission.csv
```

## Setup

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n mdt-ah python=3.10 -y
conda activate mdt-ah

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required for audio loading)
# conda install -c conda-forge ffmpeg
```

## Data

Place the BAH dataset in the `data/` folder. Expected structure:

```
data/
  data/                    # labeled split
    split/                 # train.txt, val.txt, test.txt
    Videos/
    cropped-aligned-faces/
    transcription/
  test_unlabeled/          # challenge test set
    split/
    Videos/
    cropped-aligned-faces/
    transcription/
```

Obtain the BAH dataset from the [ABAW Challenge](https://abaw.github.io/) / [BAH dataset](https://github.com/sbelharbi/bah-dataset).

## Pre-extract Audio (recommended)

Run once before training for faster data loading:

```bash
conda run -n mdt-ah python scripts/extract_audio.py
```

## Training

### Default Configuration

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

### Ablation Experiment Options

```bash
# Discrepancy feature types
--conflict_type abs              # absolute-difference only (6-token CA-AH baseline)
--conflict_type both             # abs + Hadamard-product (9-token MDT)

# Disable discrepancy (3-token: v+a+t only, for ablation)
--no_conflict

# FiLM modulation
--use_film                       # text-conditioned FiLM on video/audio

# LoRA fine-tuning
--use_lora --lora_r R --lora_alpha A

# Focal Loss
--focal_gamma G --focal_alpha A

# CutMix data augmentation
--use_cutmix --cutmix_prob P --cutmix_alpha ALPHA

# LR warmup
--warmup_epochs W

# Multi-window training
--num_windows K                  # K uniformly-spaced windows, mean pooling

# Gated fusion (optional)
--use_gated_diff                 # gated difference discrepancy features
--use_gated_fusion               # gated fusion instead of Transformer
```

## Inference / Prediction

### Prediction

```bash
python scripts/predict.py \
    --checkpoints outputs/runs/<RUN_TIMESTAMP>/best_model.pt \
    --split test_unlabeled --num_windows 5 --output outputs/submission.csv
```

### Evaluate on Labeled Test Set

```bash
python scripts/predict.py \
    --checkpoints outputs/runs/<RUN_TIMESTAMP>/best_model.pt \
    --split test --num_windows 5 --output outputs/submission_test.csv
```

## Architecture

MDT uses a 9-token discrepancy representation: three modality embeddings (v, a, t) plus six discrepancy features — three absolute-difference (|v−a|, |v−t|, |a−t|) and three Hadamard-product (W(v⊙a), W(v⊙t), W(a⊙t)) via learned projections. A 2-layer Transformer attends over these tokens, followed by an MLP classifier. FiLM modulation conditions video/audio on text, and a text-only auxiliary head is blended with the full multimodal output at inference.

- **Encoders**: VideoMAE-Base, HuBERT-Base, RoBERTa-GoEmotions (frozen, LoRA-tuned)
- **FiLM**: Text-conditioned Feature-wise Linear Modulation on video/audio before discrepancy computation
- **LoRA**: Low-rank adaptation on query/value projections of each encoder
- **Fusion**: 9-token Transformer → MLP classifier
- **Late Fusion**: Text-guided blend

## Code Structure

```
MDT-AH/
├── bah/                          # core package
│   ├── datasets/
│   │   ├── base.py               # BaseDataset class
│   │   └── abaw10_ah.py          # ABAW10 A-H dataset loader
│   ├── models/
│   │   ├── components.py         # AttentionPool, Collate, CutMix
│   │   └── conflict_aware_ah.py  # ConflictAwareAHModel (MDT backbone)
│   └── registry.py               # Registry for models/datasets
├── scripts/
│   ├── train.py                  # training script
│   ├── train.sh                  # training launcher (bash)
│   ├── predict.py                # inference / submission script
│   └── extract_audio.py          # audio pre-extraction
├── outputs/
│   └── outputs_7all1/            # best MDT checkpoint & results
├── data/                         # dataset directory
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite both the original ConflictAwareAH work and our MDT paper:

```bibtex
@inproceedings{conflictawareah2026,
  title={Conflict-Aware Multimodal Fusion for Ambivalence and Hesitancy Recognition},
  author={Bekhouche, Salah Eddine and Telli, Hichem and Benlamoudi, Azeddine and Herrouz, Salah Eddine and Taleb-Ahmed, Abdelmalik and Hadid, Abdenour},
  year={2026}
}

@inproceedings{mdt2026,
  title={Modality Discrepancy Transformer for Ambivalence and Hesitancy Recognition},
  author={Liu, Bin and Luo, Shiyu and Wang, Yu and Huang, Chenxi and Huang, Jiawen and Zhang, Qi and Xiao, Zhaoxiang},
  booktitle={ECCV 2026 Workshop — 11th ABAW Competition},
  year={2026}
}
```
