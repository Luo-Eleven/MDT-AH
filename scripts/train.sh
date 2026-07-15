#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Training launcher for the ABAW10 Conflict-Aware A-H model
#
# Each run is self-contained under its own timestamped directory:
#   outputs/runs/<TIMESTAMP>/train.log       (full training log)
#   outputs/runs/<TIMESTAMP>/metrics.csv     (per-epoch metrics, one row per epoch)
#   outputs/runs/<TIMESTAMP>/config.json     (full hyper-parameter snapshot)
#   outputs/runs/<TIMESTAMP>/best_model.pt   (best checkpoint by Macro F1)
#   outputs/runs/<TIMESTAMP>/last_model.pt   (latest checkpoint, for resuming)
#
# Usage:
#   bash scripts/train.sh                      # default settings
#   CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh   # pick a specific GPU
#   bash scripts/train.sh --conflict_type both --use_film --focal_gamma 2.0
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export TF_ENABLE_ONEDNN_OPTS=0          # silence TF/CUDA duplicate-plugin warnings
export TOKENIZERS_PARALLELISM=false     # silence HuggingFace tokenizer fork warning
export PYTHONUNBUFFERED=1               # flush Python output immediately
export TRANSFORMERS_OFFLINE=1           # use cached HF models; skip network update-checks
export HF_HUB_OFFLINE=1                # same for hf_hub

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# ── Training ──────────────────────────────────────────────────────────────────
# Key model choices (aligned with SOTA from ABAW8):
#   text_model  : RoBERTa trained on GoEmotions  (was: bert-base-uncased)
#   audio_model : HuBERT-base                    (was: wav2vec2-base)
#   video_model : VideoMAE-base                  (unchanged)
#
# unfreeze_top_k 2: unfreeze top-2 layers of each frozen encoder with 10x
#   lower LR for gentle task-specific adaptation.
#
# ══════════════════════════════════════════════════════════════════════════════
# 消融实验参数 (取消注释以启用)
# ══════════════════════════════════════════════════════════════════════════════

conda run --no-capture-output -n conda3.10 python "$SCRIPT_DIR/train.py" \
    --data_root          "$ROOT/data"      \
    --output_dir         "$ROOT/outputs"   \
    --video_model        "MCG-NJU/videomae-base"                    \
    --audio_model        "facebook/hubert-base-ls960"               \
    --text_model         "SamLowe/roberta-base-go_emotions" \
    --epochs             60               \
    --batch_size         4                \
    --grad_accum_steps   4                \
    --num_workers        4                \
    --lr                 3e-5             \
    --weight_decay       1e-2             \
    --dropout            0.5              \
    --num_frames         16               \
    --img_size           224              \
    --audio_sr           16000            \
    --max_text_len       128              \
    --log_every          10               \
    --early_stopping_patience 15         \
    --class_weight       auto             \
    --freeze_encoders                     \
    --unfreeze_top_k     2                \
    --text_loss_weight   0.5              \
    --text_blend         0.6              \
    --skip_audio_extraction               \
    "$@"

# ══════════════════════════════════════════════════════════════════════════════
# 消融实验组合示例 (通过 "$@" 传递或直接取消注释):
# ══════════════════════════════════════════════════════════════════════════════
#
# ── 改进1: 冲突特征类型 ──────────────────────────────────────────────────────
#   --conflict_type cosine          # 仅余弦相似度冲突 (3cos, 共6token)
#   --conflict_type both            # 绝对差+余弦 (6冲突, 共9token)
#
# ── 改进2: FiLM调制 ──────────────────────────────────────────────────────────
#   --use_film                      # 文本条件FiLM调制视频/音频
#
# ── 改进3: Focal Loss ────────────────────────────────────────────────────────
#   --focal_gamma 2.0               # 融合分支focal loss (gamma=2)
#   --focal_alpha 0.25
#
# ── 改进4: LR Warmup ─────────────────────────────────────────────────────────
#   --warmup_epochs 5               # 前5 epoch线性warmup
#
# ── 改进5: CutMix数据增强 ────────────────────────────────────────────────────
#   --use_cutmix                    # 视频+音频CutMix增强
#   --cutmix_prob 0.5
#   --cutmix_alpha 1.0
#
# ── 改进6: LoRA微调 ──────────────────────────────────────────────────────────
#   --use_lora                      # LoRA替代unfreeze_top_k
#   --lora_r 8
#   --lora_alpha 16
#   (使用LoRA时需移除 --unfreeze_top_k, 脚本会自动处理)
#
# ── 改进7: 多窗口训练 ────────────────────────────────────────────────────────
#   --num_windows 3                 # 每视频等距采样3个窗口, 池化embedding
#
# ── 组合消融示例 ─────────────────────────────────────────────────────────────
#   bash scripts/train.sh --conflict_type both --use_film --focal_gamma 2.0
#   bash scripts/train.sh --use_lora --lora_r 8 --warmup_epochs 5 --use_cutmix
#   bash scripts/train.sh --conflict_type both --use_film --num_windows 3
#
# ── 改进8: 门控差分配置 ─────────────────────────────────────────────────
#   --use_gated_diff                # 门控差分冲突特征
#   --use_gated_fusion              # 门控融合替代Transformer
#   --fix_audio_mask                # 精确音频mask下采样
#
# ══════════════════════════════════════════════════════════════════════════════
