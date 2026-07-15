"""
Training script – Conflict-Aware Multimodal A-H Recognition
============================================================

Split strategy:
    train  → 'train_val'  (train + val sets merged)
    eval   → 'test'

Audio pre-extraction (run once, then training loads WAV directly):
    conda run -n conda3.12 python scripts/extract_audio.py

Usage:
    conda run -n conda3.12 python scripts/train.py
    conda run -n conda3.12 python scripts/train.py --batch_size 4 --epochs 20 --freeze_encoders
    conda run -n conda3.12 python scripts/train.py --skip_audio_extraction   # if WAVs exist
    conda run -n conda3.12 python scripts/train.py --conflict_type both --use_film --focal_gamma 2.0
"""
from __future__ import annotations

import os
import sys

# Force offline mode before any HuggingFace imports – prevents network calls
# that crash with RuntimeError when the httpx client has been closed.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bah.datasets.abaw10_ah import ABAW10_AH_Dataset
from bah.models import ConflictAwareAHModel, multimodal_collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(run_dir: str) -> logging.Logger:
    """Create a logger that writes simultaneously to stdout and to <run_dir>/train.log."""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'train.log')

    fmt     = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger  = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f'Log file: {log_path}')
    return logger


def setup_metrics_csv(run_dir: str) -> tuple[str, csv.DictWriter, object]:
    """Open (or append to) a CSV file that records per-epoch metrics."""
    csv_path = os.path.join(run_dir, 'metrics.csv')
    exists   = os.path.isfile(csv_path)
    fh       = open(csv_path, 'a', newline='', encoding='utf-8')
    fields   = ['epoch', 'train_loss', 'val_loss',
                'val_macro_f1', 'val_f1_ah', 'val_f1_no_ah', 'val_accuracy', 'lr']
    writer   = csv.DictWriter(fh, fieldnames=fields)
    if not exists:
        writer.writeheader()
    return csv_path, writer, fh


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train ConflictAwareAHModel')

    # ── Data / I/O ─────────────────────────────────────────────────────────
    p.add_argument('--data_root',     default=str(ROOT / 'data'),   type=str)
    p.add_argument('--output_dir',    default=str(ROOT / 'outputs'), type=str)
    p.add_argument('--train_split', default='train_val', type=str)
    p.add_argument('--val_split', default='test', type=str)
    p.add_argument('--resume',        default=None, type=str)

    # ── Model architecture ─────────────────────────────────────────────────
    p.add_argument('--video_model',   default='MCG-NJU/videomae-base',                    type=str)
    p.add_argument('--audio_model',   default='facebook/hubert-base-ls960',               type=str)
    p.add_argument('--text_model',    default='SamLowe/roberta-base-go_emotions', type=str)
    p.add_argument('--embed_dim',     default=768,  type=int)
    p.add_argument('--hidden_dim',    default=512,  type=int)
    p.add_argument('--num_layers',    default=2,    type=int)
    p.add_argument('--num_heads',     default=8,    type=int)
    p.add_argument('--dropout',       default=0.5,  type=float)

    # ── Encoder fine-tuning ────────────────────────────────────────────────
    p.add_argument('--freeze_encoders', action='store_true',
                   help='Freeze pre-trained encoder weights')
    p.add_argument('--unfreeze_top_k', default=0, type=int,
                   help='Unfreeze top K layers of each encoder with 10x lower LR')
    p.add_argument('--use_lora', action='store_true',
                   help='Use LoRA fine-tuning instead of unfreeze_top_k')
    p.add_argument('--lora_r', default=8, type=int, help='LoRA rank')
    p.add_argument('--lora_alpha', default=16, type=int, help='LoRA alpha scaling')

    # ── Fusion / Conflict features ─────────────────────────────────────────
    p.add_argument('--fusion_type', default='6token', choices=['6token', 'concat'])
    p.add_argument('--conflict_type', default='abs', choices=['abs', 'cosine', 'both'],
                   help='abs=|v-a|, cosine=v⊙a→proj, both=abs+cosine (6 conflict tokens)')
    p.add_argument('--no_conflict', action='store_true',
                   help='Ablation: zero out conflict features (v+a+t only)')
    p.add_argument('--use_film', action='store_true',
                   help='Text-conditioned FiLM modulation on video/audio embeddings')
    p.add_argument('--fix_audio_mask', action='store_true',
                   help='Use adaptive_avg_pool1d for accurate audio mask down-sampling')
    p.add_argument('--use_gated_diff', action='store_true',
                   help='Use learnable per-dimension gating on |v-a| conflict features')
    p.add_argument('--use_gated_fusion', action='store_true',
                   help='Use attention-weighted fusion instead of Transformer')

    # ── Data loading ───────────────────────────────────────────────────────
    p.add_argument('--num_frames',    default=16,   type=int)
    p.add_argument('--num_windows',   default=1,    type=int,
                   help='Number of uniformly-spaced 16-frame windows per video. '
                        '>1 enables multi-window mean-pool training')
    p.add_argument('--img_size',      default=224,  type=int)
    p.add_argument('--audio_sr',      default=16_000, type=int)
    p.add_argument('--max_text_len',  default=128,  type=int)

    # ── Training hyperparameters ───────────────────────────────────────────
    p.add_argument('--batch_size',    default=4,    type=int)
    p.add_argument('--num_workers',   default=4,    type=int)
    p.add_argument('--epochs',        default=50,   type=int)
    p.add_argument('--lr',            default=3e-5, type=float)
    p.add_argument('--weight_decay',  default=1e-2, type=float)
    p.add_argument('--grad_accum_steps', default=4, type=int)
    p.add_argument('--early_stopping_patience', default=7, type=int)
    p.add_argument('--warmup_epochs', default=0, type=int,
                   help='Number of linear LR warmup epochs before cosine annealing')

    # ── Loss ───────────────────────────────────────────────────────────────
    p.add_argument('--class_weight',  default='auto', choices=['auto', 'equal'])
    p.add_argument('--label_smoothing', default=0.0, type=float)
    p.add_argument('--text_loss_weight', default=0.5, type=float,
                   help='Weight for text-only auxiliary loss branch')
    p.add_argument('--text_blend', default=0.6, type=float,
                   help='Inference blend weight for text logit')
    p.add_argument('--focal_gamma', default=0.0, type=float,
                   help='Focal loss gamma for full-fusion branch. 0=disabled (BCE)')
    p.add_argument('--focal_alpha', default=0.25, type=float,
                   help='Focal loss alpha balancing factor')

    # ── Data augmentation ──────────────────────────────────────────────────
    p.add_argument('--use_cutmix', action='store_true',
                   help='Apply video+audio CutMix augmentation')
    p.add_argument('--cutmix_prob', default=0.5, type=float)
    p.add_argument('--cutmix_alpha', default=1.0, type=float)

    # ── Modality / ablation flags ──────────────────────────────────────────
    p.add_argument('--active_modalities', default='video,audio,text', type=str)
    p.add_argument('--use_window_transcript', action='store_true',
                   help='Use window-aligned transcript instead of full')

    # ── Logging / execution ────────────────────────────────────────────────
    p.add_argument('--log_every',     default=10,   type=int)
    p.add_argument('--skip_audio_extraction', action='store_true')
    p.add_argument('--extract_workers', default=4, type=int)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: list[int], targets: list[int]) -> dict:
    """Compute Macro F1 and per-class metrics."""
    n = len(preds)
    if n == 0:
        return {'macro_f1': 0.0, 'f1_ah': 0.0, 'f1_no_ah': 0.0, 'ap': 0.0, 'accuracy': 0.0}

    per_class = {}
    for cls in (0, 1):
        tp = sum(p == cls and t == cls for p, t in zip(preds, targets))
        fp = sum(p == cls and t != cls for p, t in zip(preds, targets))
        fn = sum(p != cls and t == cls for p, t in zip(preds, targets))
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {'prec': prec, 'rec': rec, 'f1': f1}

    macro_f1 = (per_class[0]['f1'] + per_class[1]['f1']) / 2.0
    accuracy  = sum(p == t for p, t in zip(preds, targets)) / n

    return {
        'macro_f1': macro_f1,
        'f1_ah':    per_class[1]['f1'],
        'f1_no_ah': per_class[0]['f1'],
        'ap':       per_class[1]['prec'],
        'accuracy': accuracy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(dataset) -> torch.Tensor:
    """Return pos_weight for BCEWithLogitsLoss based on training label distribution."""
    labels   = dataset.labels
    n_pos    = sum(1 for l in labels if l == 1)
    n_neg    = sum(1 for l in labels if l == 0)
    if n_pos == 0:
        return torch.tensor(1.0)
    weight = n_neg / n_pos
    return torch.tensor(weight, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Focal Loss for binary classification with logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits:     (B,) raw logits
        targets:    (B,) float labels in [0, 1]
        gamma:      focusing parameter. 0 = standard BCE
        alpha:      class balancing factor for positive class
        pos_weight: optional per-sample weight for positive class
    Returns:
        scalar loss
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight)

    # p_t = p if y==1 else 1-p  →  exp(-bce) gives p_t
    pt = torch.exp(-bce)

    # alpha_t = alpha if y==1 else 1-alpha
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    focal_weight = alpha_t * (1.0 - pt) ** gamma
    return (focal_weight * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def move_to_device(batch: dict, device: torch.device, tokenizer, max_len: int,
                   use_window_transcript: bool = False) -> tuple:
    """Tokenize text and move all tensors to device."""
    video      = batch['video'].to(device)
    audio      = batch['audio'].to(device)
    audio_mask = batch['audio_mask'].to(device)
    labels     = batch['label'].to(device)

    transcript_key = 'transcript' if use_window_transcript else 'transcript_full'
    text_enc = tokenizer(
        batch[transcript_key],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    text_enc = {k: v.to(device) for k, v in text_enc.items()}

    return video, audio, audio_mask, text_enc, labels


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    tokenizer,
    device:    torch.device,
    args:      argparse.Namespace,
    epoch:     int,
    log:       logging.Logger,
) -> float:
    model.train()
    total_loss  = 0.0
    valid_steps = 0
    nan_skipped = 0
    t0 = time.time()

    accum = args.grad_accum_steps
    optimizer.zero_grad()

    tw = args.text_loss_weight
    ls = args.label_smoothing
    focal_gamma = getattr(args, 'focal_gamma', 0.0)
    focal_alpha = getattr(args, 'focal_alpha', 0.25)
    active = set(args.active_modalities.split(','))
    use_win = getattr(args, 'use_window_transcript', False)

    for step, batch in enumerate(loader, 1):
        video, audio, audio_mask, text_enc, labels = move_to_device(
            batch, device, tokenizer, args.max_text_len, use_window_transcript=use_win)

        with torch.cuda.amp.autocast(enabled=args.amp):
            full_logit, text_logit = model(
                video, audio, text_enc, audio_mask, active_modalities=active)

            # ── Prepare targets ────────────────────────────────────────────
            # If CutMix was applied, labels are already float mixed values;
            # skip label smoothing in that case.
            cutmix_active = batch.get('cutmix_lambda') is not None
            if cutmix_active:
                targets = labels.float()  # already mixed by CutMix
            else:
                targets = labels.float() * (1.0 - ls) + 0.5 * ls

            # ── Full-fusion loss: Focal or BCE ─────────────────────────────
            if focal_gamma > 0:
                loss_full = focal_bce_with_logits(
                    full_logit.squeeze(-1), targets,
                    gamma=focal_gamma, alpha=focal_alpha)
            else:
                loss_full = criterion(full_logit.squeeze(-1), targets)

            # ── Text-only loss: always BCE ────────────────────────────────
            loss_text = criterion(text_logit.squeeze(-1), targets)

            loss = (1.0 - tw) * loss_full + tw * loss_text
            loss = loss / accum

        if torch.isnan(loss) or torch.isinf(loss):
            nan_skipped += 1
            log.warning(f'  [epoch {epoch:03d}  step {step:04d}] NaN/Inf loss — skipping step')
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if step % accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss  += loss.item() * accum
        valid_steps += 1

        if step % args.log_every == 0:
            avg     = total_loss / valid_steps
            elapsed = time.time() - t0
            log.info(f'  [epoch {epoch:03d}  step {step:04d}/{len(loader):04d}]'
                     f'  loss={avg:.4f}  elapsed={elapsed:.1f}s')

    if nan_skipped:
        log.warning(f'  [epoch {epoch:03d}] {nan_skipped} steps skipped due to NaN/Inf loss')

    return total_loss / valid_steps if valid_steps > 0 else float('nan')


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    tokenizer,
    device:    torch.device,
    args:      argparse.Namespace,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []

    tb = args.text_blend
    tw = args.text_loss_weight
    active = set(args.active_modalities.split(','))
    use_win = getattr(args, 'use_window_transcript', False)

    for batch in loader:
        video, audio, audio_mask, text_enc, labels = move_to_device(
            batch, device, tokenizer, args.max_text_len, use_window_transcript=use_win)

        with torch.cuda.amp.autocast(enabled=args.amp):
            full_logit, text_logit = model(
                video, audio, text_enc, audio_mask, active_modalities=active)
            loss_full = criterion(full_logit.squeeze(-1), labels.float())
            loss_text = criterion(text_logit.squeeze(-1), labels.float())
            loss = (1.0 - tw) * loss_full + tw * loss_text
        total_loss += loss.item()

        full_prob = torch.sigmoid(full_logit.squeeze(-1))
        text_prob = torch.sigmoid(text_logit.squeeze(-1))
        blended   = tb * text_prob + (1.0 - tb) * full_prob

        probs = blended.cpu().tolist()
        all_probs.extend(probs if isinstance(probs, list) else [probs])
        all_targets.extend(labels.cpu().tolist())

    # ── Threshold sweep ────────────────────────────────────────────────────
    best_f1, best_thresh = 0.0, 0.5
    for t_int in range(25, 75):
        t = t_int / 100.0
        preds_t = [int(p > t) for p in all_probs]
        f1_t = compute_metrics(preds_t, all_targets)['macro_f1']
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    all_preds = [int(p > best_thresh) for p in all_probs]
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss']       = total_loss / len(loader)
    metrics['threshold']  = best_thresh
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_audio_extraction(args: argparse.Namespace):
    """Pre-extract all mp4 audio to WAV before training starts."""
    import subprocess as sp
    extract_script = str(ROOT / 'scripts' / 'extract_audio.py')
    print('\nPre-extracting audio (run once – skipped automatically when WAVs exist) …')
    cmd = [sys.executable, extract_script,
           '--data_root', args.data_root,
           '--sample_rate', str(args.audio_sr),
           '--workers', str(args.extract_workers)]
    result = sp.run(cmd)
    if result.returncode not in (0, 1):
        print('  WARNING: audio extraction encountered errors – training will fall back to ffmpeg.')
    print()


def apply_lora_to_encoders(model, args, device, log):
    """Apply LoRA adapters to all three frozen encoders."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        log.error('peft library not installed. Install with: pip install peft>=0.10.0')
        raise

    lora_r = args.lora_r
    lora_alpha = args.lora_alpha

    # VideoMAE — ViT-style attention
    lora_config_video = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=['query', 'value'],
        lora_dropout=args.dropout,
    )
    model.video_encoder = get_peft_model(model.video_encoder, lora_config_video)

    # HuBERT / wav2vec2 — special naming
    lora_config_audio = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.dropout,
    )
    model.audio_encoder = get_peft_model(model.audio_encoder, lora_config_audio)

    # RoBERTa / BERT — standard attention naming
    lora_config_text = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=['query', 'value'],
        lora_dropout=args.dropout,
    )
    model.text_encoder = get_peft_model(model.text_encoder, lora_config_text)

    # Re-assign to model attributes so they stay accessible
    for name in ('video_encoder', 'audio_encoder', 'text_encoder'):
        setattr(model, '_peft_' + name, True)

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  LoRA trainable params: {lora_params:,}')


def main():
    args      = parse_args()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name  = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_dir = os.path.join(args.output_dir, 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    log = setup_logger(run_dir)
    csv_path, csv_writer, csv_fh = setup_metrics_csv(run_dir)

    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    # ── Parameter interaction validation ───────────────────────────────────
    use_lora = getattr(args, 'use_lora', False)
    if use_lora and args.unfreeze_top_k > 0:
        log.warning('  use_lora + unfreeze_top_k are mutually exclusive. Using LoRA; setting unfreeze_top_k=0.')
        args.unfreeze_top_k = 0
    if use_lora and not args.freeze_encoders:
        log.info('  use_lora requires freeze_encoders. Auto-enabling freeze_encoders.')
        args.freeze_encoders = True

    conflict_type = getattr(args, 'conflict_type', 'abs')
    fusion_type = getattr(args, 'fusion_type', '6token')
    if conflict_type in ('cosine', 'both') and fusion_type == 'concat' and not args.no_conflict:
        log.warning(f'  conflict_type={conflict_type} + fusion_type=concat may produce large d_model. '
                    f'Recommend using fusion_type=6token.')

    # ── Log config ─────────────────────────────────────────────────────────
    log.info('=' * 60)
    log.info('Conflict-Aware A-H Training')
    log.info(f'  run_name     : {run_name}')
    log.info(f'  device       : {device}')
    log.info(f'  data_root    : {args.data_root}')
    log.info(f'  output_dir   : {args.output_dir}')
    log.info(f'  freeze_enc        : {args.freeze_encoders}')
    log.info(f'  unfreeze_top_k    : {args.unfreeze_top_k}')
    log.info(f'  use_lora          : {use_lora}')
    if use_lora:
        log.info(f'  lora_r / alpha    : {args.lora_r} / {args.lora_alpha}')
    log.info(f'  amp               : {args.amp}')
    log.info(f'  batch_size        : {args.batch_size}')
    log.info(f'  grad_accum_steps  : {args.grad_accum_steps}')
    log.info(f'  effective_batch   : {args.batch_size * args.grad_accum_steps}')
    log.info(f'  epochs            : {args.epochs}')
    log.info(f'  lr                : {args.lr}')
    log.info(f'  warmup_epochs     : {args.warmup_epochs}')
    log.info(f'  class_weight      : {args.class_weight}')
    log.info(f'  early_stop_pat.   : {args.early_stopping_patience}')
    log.info(f'  text_loss_weight  : {args.text_loss_weight}')
    log.info(f'  text_blend        : {args.text_blend}')
    log.info(f'  label_smoothing   : {args.label_smoothing}')
    log.info(f'  focal_gamma       : {args.focal_gamma}')
    log.info(f'  active_modalities : {args.active_modalities}')
    log.info(f'  train_split       : {args.train_split}')
    log.info(f'  val_split         : {args.val_split}')
    log.info(f'  fusion_type       : {fusion_type}')
    log.info(f'  conflict_type     : {conflict_type}')
    log.info(f'  conflict: {"disabled" if args.no_conflict else "enabled"}')
    log.info(f'  use_film          : {getattr(args, "use_film", False)}')
    log.info(f'  fix_audio_mask    : {getattr(args, "fix_audio_mask", False)}')
    log.info(f'  use_gated_diff    : {getattr(args, "use_gated_diff", False)}')
    log.info(f'  use_gated_fusion  : {getattr(args, "use_gated_fusion", False)}')
    log.info(f'  num_windows       : {args.num_windows}')
    log.info(f'  use_cutmix        : {getattr(args, "use_cutmix", False)}')
    log.info(f'  text: {"window-aligned" if args.use_window_transcript else "full"} transcript')
    log.info(f'  run_dir           : {run_dir}')
    log.info(f'  metrics csv       : {csv_path}')
    log.info('=' * 60)

    # ── Audio pre-extraction ───────────────────────────────────────────────
    if not args.skip_audio_extraction:
        run_audio_extraction(args)

    # ── Datasets ──────────────────────────────────────────────────────────
    log.info('\nLoading datasets …')
    num_windows = getattr(args, 'num_windows', 1)
    dataset_kwargs = dict(
        root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        audio_sample_rate=args.audio_sr,
        num_windows=num_windows,
    )
    # Validation always uses 1 window
    val_kwargs = dict(dataset_kwargs, num_windows=1)
    train_dataset = ABAW10_AH_Dataset(split=args.train_split, random_frames_crop=True,  **dataset_kwargs)
    val_dataset   = ABAW10_AH_Dataset(split=args.val_split,   random_frames_crop=False, **val_kwargs)
    n_pos = sum(1 for l in train_dataset.labels if l == 1)
    n_neg = sum(1 for l in train_dataset.labels if l == 0)
    log.info(f'  train samples : {len(train_dataset)}  (pos={n_pos}, neg={n_neg})')
    log.info(f'  val   samples : {len(val_dataset)}')

    # ── Collate: optionally wrap with CutMix ───────────────────────────────
    use_cutmix = getattr(args, 'use_cutmix', False)
    if use_cutmix:
        from bah.models.components import CutMixCollate
        train_collate = CutMixCollate(
            multimodal_collate_fn,
            prob=args.cutmix_prob,
            alpha=args.cutmix_alpha,
        )
        log.info(f'  CutMix enabled: prob={args.cutmix_prob}, alpha={args.cutmix_alpha}')
    else:
        train_collate = multimodal_collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    log.info('\nBuilding model …')
    model = ConflictAwareAHModel(
        video_model=args.video_model,
        audio_model=args.audio_model,
        text_model=args.text_model,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_encoders=args.freeze_encoders,
        unfreeze_top_k=args.unfreeze_top_k,
        no_conflict=args.no_conflict,
        fusion_type=fusion_type,
        conflict_type=conflict_type,
        use_film=getattr(args, 'use_film', False),
        num_windows=num_windows,
        fix_audio_mask=getattr(args, 'fix_audio_mask', False),
        use_gated_diff=getattr(args, 'use_gated_diff', False),
        use_gated_fusion=getattr(args, 'use_gated_fusion', False),
    ).to(device)

    # ── Apply LoRA if requested ────────────────────────────────────────────
    if use_lora:
        apply_lora_to_encoders(model, args, device, log)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  total params     : {total_params:,}')
    log.info(f'  trainable params : {trainable_params:,}')

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)

    # ── Loss / optimiser / scheduler / AMP scaler ─────────────────────────
    if args.class_weight == 'auto':
        pos_weight = compute_pos_weight(train_dataset).to(device)
        log.info(f'\nClass weights (pos_weight for With A-H): {pos_weight.item():.3f}')
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ── Optimizer param groups (for unfreeze_top_k) ───────────────────────
    if args.unfreeze_top_k > 0 and args.freeze_encoders and not use_lora:
        encoder_ids = set()
        for enc in (model.video_encoder, model.audio_encoder, model.text_encoder):
            for p in enc.parameters():
                if p.requires_grad:
                    encoder_ids.add(id(p))
        encoder_params = [p for p in model.parameters()
                          if p.requires_grad and id(p) in encoder_ids]
        other_params   = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in encoder_ids]
        param_groups = [
            {'params': other_params,   'lr': args.lr},
            {'params': encoder_params, 'lr': args.lr / 10.0},
        ]
        log.info(f'  optimizer: 2 param groups — head lr={args.lr:.2e},'
                 f' encoder top-{args.unfreeze_top_k} lr={args.lr/10:.2e}')
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Scheduler: warmup + cosine ─────────────────────────────────────────
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-4, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 0.01)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])
        log.info(f'  scheduler: linear warmup ({warmup_epochs} epochs) + cosine annealing')
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Optional resume ───────────────────────────────────────────────────
    start_epoch    = 1
    best_f1        = 0.0
    best_ckpt_path = os.path.join(run_dir, 'best_model.pt')

    if args.resume and os.path.isfile(args.resume):
        log.info(f'\nResuming from {args.resume}')
        ckpt        = torch.load(args.resume, map_location=device)
        # LoRA: load peft adapters if they exist alongside the checkpoint
        if use_lora:
            peft_resume_dir = os.path.join(os.path.dirname(args.resume), 'peft_adapters')
            if os.path.isdir(peft_resume_dir):
                # Find latest epoch adapter file
                epoch_str = os.path.basename(args.resume).replace('last_model.pt', '').replace('best_model.pt', '')
                adapter_path = os.path.join(peft_resume_dir, f'adapter_{epoch_str}.pt')
                if not os.path.isfile(adapter_path):
                    # Fallback: find any adapter file
                    adapter_files = sorted([f for f in os.listdir(peft_resume_dir) if f.startswith('adapter')])
                    if adapter_files:
                        adapter_path = os.path.join(peft_resume_dir, adapter_files[-1])
                if os.path.isfile(adapter_path):
                    peft_state = torch.load(adapter_path, map_location=device)
                    for enc_name, state in peft_state.items():
                        getattr(model, enc_name).load_state_dict(state, strict=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_f1     = ckpt.get('best_f1', 0.0)
        log.info(f'  resumed at epoch {start_epoch}, best_macro_f1={best_f1:.4f}')

    # ── Training loop ─────────────────────────────────────────────────────
    log.info('\nStarting training …\n')
    epochs_no_improve = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                tokenizer, device, args, epoch, log)

            val_metrics = evaluate(
                model, val_loader, criterion,
                tokenizer, device, args)

            lr_now = scheduler.get_last_lr()
            if isinstance(lr_now, list):
                lr_now = lr_now[0]
            scheduler.step()

            log.info(
                f'Epoch {epoch:03d}/{args.epochs:03d}'
                f'  train_loss={train_loss:.4f}'
                f'  val_loss={val_metrics["loss"]:.4f}'
                f'  val_macro_f1={val_metrics["macro_f1"]:.4f}'
                f'  val_f1_ah={val_metrics["f1_ah"]:.4f}'
                f'  val_f1_no_ah={val_metrics["f1_no_ah"]:.4f}'
                f'  val_acc={val_metrics["accuracy"]:.4f}'
                f'  thresh={val_metrics["threshold"]:.2f}'
                f'  lr={lr_now:.2e}'
            )

            csv_writer.writerow({
                'epoch':        epoch,
                'train_loss':   f'{train_loss:.6f}',
                'val_loss':     f'{val_metrics["loss"]:.6f}',
                'val_macro_f1': f'{val_metrics["macro_f1"]:.6f}',
                'val_f1_ah':    f'{val_metrics["f1_ah"]:.6f}',
                'val_f1_no_ah': f'{val_metrics["f1_no_ah"]:.6f}',
                'val_accuracy': f'{val_metrics["accuracy"]:.6f}',
                'lr':           f'{lr_now:.2e}',
            })
            csv_fh.flush()

            ckpt = {
                'epoch':        epoch,
                'run_name':     run_name,
                'model':        model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'scheduler':    scheduler.state_dict(),
                'scaler':       scaler.state_dict(),
                'best_f1':      best_f1,
                'val_metrics':  val_metrics,
                'best_thresh':  val_metrics.get('threshold', 0.5),
                'args':         vars(args),
            }
            # Ensure run_dir exists (belt-and-suspenders for LoRA/peft edge cases)
            os.makedirs(run_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(run_dir, 'last_model.pt'))
            # LoRA: also save peft adapters to a subdirectory for easy reload
            if use_lora:
                peft_dir = os.path.join(run_dir, 'peft_adapters')
                from peft import get_peft_model_state_dict
                peft_state = {}
                for enc_name in ('video_encoder', 'audio_encoder', 'text_encoder'):
                    enc = getattr(model, enc_name)
                    peft_state[enc_name] = get_peft_model_state_dict(enc)
                os.makedirs(peft_dir, exist_ok=True)
                torch.save(peft_state, os.path.join(peft_dir, f'adapter_epoch{epoch:03d}.pt'))

            if val_metrics['macro_f1'] > best_f1:
                best_f1           = val_metrics['macro_f1']
                ckpt['best_f1']   = best_f1
                epochs_no_improve = 0
                os.makedirs(run_dir, exist_ok=True)
                torch.save(ckpt, best_ckpt_path)
                # LoRA: save peft adapters for best model too
                if use_lora:
                    best_peft_path = os.path.join(peft_dir, 'adapter_best.pt')
                    torch.save(peft_state, best_peft_path)
                log.info(f'  ★ new best Macro F1 = {best_f1:.4f}'
                         f'  (thresh={val_metrics["threshold"]:.2f})'
                         f'  →  {best_ckpt_path}')
            else:
                epochs_no_improve += 1
                log.info(f'  no improvement ({epochs_no_improve}/{args.early_stopping_patience})')
                if epochs_no_improve >= args.early_stopping_patience:
                    log.info(f'Early stopping triggered after epoch {epoch}.')
                    break

            log.info('')

    finally:
        csv_fh.close()

    log.info('Training complete.')
    log.info(f'Best val Macro F1 : {best_f1:.4f}')
    log.info(f'Best ckpt         : {best_ckpt_path}')
    log.info(f'Metrics CSV       : {csv_path}')


if __name__ == '__main__':
    main()
