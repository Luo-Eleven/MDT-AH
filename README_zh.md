# MDT-AH — 面向矛盾与犹豫识别的模态差异Transformer

<div align="center">

[![English](https://img.shields.io/badge/English-README-blue?style=flat-square)](./README.md)

</div>

**第11届 ABAW 竞赛**

**团队名称：** CASIA-26

**负责人：**
- Bin Liu (liubin@nlpr.ia.ac.cn)

**团队成员：**
- Shiyu Luo (luoshiyu221@mails.ucas.ac.cn)
- Yu Wang (wangyu230@mails.ucas.ac.cn)
- Chenxi Huang (huangchenqian22@mails.ucas.ac.cn)
- Jiawen Huang (huangjiawen25@mails.ucas.ac.cn)
- Qi Zhang (zhangqi2025@ia.ac.cn)
- Zhaoxiang Xiao (xuegaodef@163.com)

**联系人：**
- Jiawen Huang (huangjiawen25@mails.ucas.ac.cn)
- Qi Zhang (zhangqi2025@ia.ac.cn)

---

> **注：** 本代码基于 [ConflictAwareAH](https://github.com/sbelharbi/bah-dataset)（Bekhouche 等, 2026）修改构建：
>
> ```bibtex
> @inproceedings{conflictawareah2026,
>   title={Conflict-Aware Multimodal Fusion for Ambivalence and Hesitancy Recognition},
>   author={Bekhouche, Salah Eddine and Telli, Hichem and Benlamoudi, Azeddine and Herrouz, Salah Eddine and Taleb-Ahmed, Abdelmalik and Hadid, Abdenour},
>   year={2026}
> }
> ```

---

本项目为 ABAW11 矛盾/犹豫（Ambivalence/Hesitancy, A/H）识别挑战赛代码。MDT 将原始的 6-token 冲突感知设计扩展为 9-token 表示——包含三个模态嵌入、三个绝对差特征和三个 Hadamard 积差异特征，并通过 Transformer 进行融合，配合基于 FiLM 的文本条件调制和 LoRA 微调。

## 实验结果

| 数据集划分 | Macro F1 | F1 (A/H) | F1 (无A/H) | 准确率 |
|-----------|----------|----------|------------|--------|
| 有标签测试集 (525 视频) | **0.7408** | 0.7857 | 0.6959 | 0.7486 |
| 私有排行榜 (151 视频) | **0.7368** | 0.7887 | — | — |

## 最佳配置

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

## 环境配置

```bash
# 创建 conda 环境（推荐 Python 3.10）
conda create -n mdt-ah python=3.10 -y
conda activate mdt-ah

# 安装依赖
pip install -r requirements.txt

# 安装 ffmpeg（音频加载必需）
# conda install -c conda-forge ffmpeg
```

## 数据准备

将 BAH 数据集放入 `data/` 文件夹。预期目录结构：

```
data/
  data/                    # 有标签数据
    split/                 # train.txt, val.txt, test.txt
    Videos/
    cropped-aligned-faces/
    transcription/
  test_unlabeled/          # 挑战赛测试集
    split/
    Videos/
    cropped-aligned-faces/
    transcription/
```

从 [ABAW Challenge](https://abaw.github.io/) / [BAH 数据集](https://github.com/sbelharbi/bah-dataset) 获取 BAH 数据集。

## 音频预提取（推荐）

训练前运行一次以加速数据加载：

```bash
conda run -n mdt-ah python scripts/extract_audio.py
```

## 训练

### 默认配置

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

### 消融实验参数选项

```bash
# 差异特征类型
--conflict_type abs              # 仅绝对差（6-token，CA-AH 基线）
--conflict_type both             # 绝对差 + Hadamard 积（9-token，MDT）

# 禁用差异特征（3-token: v+a+t，用于消融实验）
--no_conflict

# FiLM 调制
--use_film                       # 基于文本条件的 FiLM 调制视频/音频

# LoRA 微调
--use_lora --lora_r R --lora_alpha A

# Focal Loss
--focal_gamma G --focal_alpha A

# CutMix 数据增强
--use_cutmix --cutmix_prob P --cutmix_alpha ALPHA

# 学习率预热
--warmup_epochs W

# 多窗口训练
--num_windows K                  # K 个均匀分布的窗口，均值池化

# 门控融合（可选）
--use_gated_diff                 # 门控差分差异特征
--use_gated_fusion               # 门控融合替代 Transformer
```

## 推理 / 预测

### 预测

```bash
python scripts/predict.py \
    --checkpoints outputs/runs/<RUN_TIMESTAMP>/best_model.pt \
    --split test_unlabeled --num_windows 5 --output outputs/submission.csv
```

### 在有标签测试集上评估

```bash
python scripts/predict.py \
    --checkpoints outputs/runs/<RUN_TIMESTAMP>/best_model.pt \
    --split test --num_windows 5 --output outputs/submission_test.csv
```

## 模型架构

MDT 采用 9-token 差异表示：三个模态嵌入 $(v, a, t)$ 加上六个差异特征——三个绝对差 $(|v-a|, |v-t|, |a-t|)$ 和三个 Hadamard 积 $(W(v \odot a), W(v \odot t), W(a \odot t))$，通过可学习的线性投影计算。一个 2 层 Transformer 对这些 token 进行自注意力计算，随后通过 MLP 分类器输出结果。FiLM 调制使视频和音频以文本为条件进行调制，推理时将仅文本辅助头与完整多模态输出进行融合。

- **编码器**：VideoMAE-Base, HuBERT-Base, RoBERTa-GoEmotions（冻结 + LoRA 微调）
- **FiLM**：基于文本条件的 Feature-wise Linear Modulation，在差异计算前调制视频/音频
- **LoRA**：对每个编码器的 query/value 投影进行低秩适配
- **融合**：9-token Transformer → MLP 分类器
- **后融合**：文本引导融合

## 代码结构

```
MDT-AH/
├── bah/                          # 核心包
│   ├── datasets/
│   │   ├── base.py               # 基础数据集类
│   │   └── abaw10_ah.py          # ABAW10 A-H 数据集加载器
│   ├── models/
│   │   ├── components.py         # AttentionPool, Collate, CutMix
│   │   └── conflict_aware_ah.py  # ConflictAwareAHModel（MDT 主干网络）
│   └── registry.py               # 模型/数据集注册器
├── scripts/
│   ├── train.py                  # 训练脚本
│   ├── train.sh                  # 训练启动脚本 (bash)
│   ├── predict.py                # 推理/提交脚本
│   └── extract_audio.py          # 音频预提取
├── outputs/
│   └── outputs_7all1/            # 最佳 MDT 检查点与结果
├── data/                         # 数据集目录
├── requirements.txt
├── README.md                     # 英文文档
└── README_zh.md                  # 中文文档（本文件）
```

## 引用

如果使用本代码，请同时引用原始 ConflictAwareAH 工作和我们的 MDT 论文：

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
