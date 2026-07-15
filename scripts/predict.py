"""
Inference / submission script for the ABAW10 A-H challenge.
============================================================

Generates predictions on the private test set (test_unlabeled) using the best
saved checkpoint(s) and writes the submission CSV required by the challenge.

Key features
------------
  Multi-window inference
    Instead of evaluating only the first 16 frames of each video, the model
    is run on N uniformly-spaced 16-frame windows and their sigmoid scores are
    averaged.  This ensures the full temporal extent of every video is covered.

  Checkpoint ensemble
    Pass multiple checkpoint paths via --checkpoints to average their sigmoid
    outputs before thresholding.  Ensembling typically adds +1–2% Macro F1.

  Automatic threshold
    If the checkpoint contains a 'best_thresh' key (written by train.py's
    threshold sweep), that value is used automatically.  Override with
    --threshold.

Submission format (per challenge spec):
    With probabilities: video_id,probability_of_class_0,probability_of_class_1,label_prediction
    Labels only:       video_id,label_prediction
    Use --output_format probabilities (default) or labels_only

Usage:
    # Single checkpoint, single window (deterministic baseline)
    conda run -n conda3.10 python scripts/predict.py

    # Multi-window inference (recommended)
    conda run -n conda3.10 python scripts/predict.py --num_windows 5

    # Ensemble of checkpoints with multi-window inference
    conda run -n conda3.10 python scripts/predict.py \\
        --checkpoints outputs/best_model.pt outputs/run2_best.pt \\
        --num_windows 5

    # Evaluate on the labeled test set (for local Macro F1 tracking)
    conda run -n conda3.10 python scripts/predict.py \\
        --split test --num_windows 5

    # Use model from Hugging Face (no checkpoint path needed)
    conda run -n conda3.10 python scripts/predict.py \\
        --hf_repo Bekhouche/ConflictAwareAH --split test --num_windows 5
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
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bah.datasets.abaw10_ah import ABAW10_AH_Dataset
from bah.models import ConflictAwareAHModel, multimodal_collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_best_ckpt() -> str:
    """Return the best_model.pt from the most recently created run directory."""
    runs_dir = ROOT / 'outputs' / 'runs'
    if runs_dir.is_dir():
        candidates = sorted(
            [d / 'best_model.pt' for d in runs_dir.iterdir()
             if d.is_dir() and (d / 'best_model.pt').exists()],
            key=lambda p: p.parent.name,   # sort by timestamp name
        )
        if candidates:
            return str(candidates[-1])
    return str(ROOT / 'outputs' / 'best_model.pt')   # legacy fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate submission CSV')
    p.add_argument('--hf_repo', type=str, default=None,
                   help='Hugging Face repo ID (e.g. Bekhouche/ConflictAwareAH). '
                        'Downloads best_model.pt automatically; no --checkpoints needed.')
    p.add_argument('--checkpoints', nargs='+',
                   default=None,
                   help='One or more checkpoint paths. Ignored when --hf_repo is set. '
                        'Defaults to the most recently created run\'s best checkpoint.')
    # Legacy alias kept for backwards-compatibility
    p.add_argument('--checkpoint', default=None, type=str,
                   help='Single checkpoint (alias for --checkpoints); '
                        'ignored when --checkpoints has multiple entries or --hf_repo is set.')
    p.add_argument('--output',       default=str(ROOT / 'outputs' / 'submission.csv'), type=str)
    p.add_argument('--data_root',    default=str(ROOT / 'data'),   type=str)
    p.add_argument('--split',        default='test_unlabeled',      type=str,
                   help='Dataset split to run inference on (test | test_unlabeled)')
    p.add_argument('--batch_size',   default=4,    type=int)
    p.add_argument('--num_workers',  default=4,    type=int)
    p.add_argument('--num_frames',   default=16,   type=int)
    p.add_argument('--img_size',     default=224,  type=int)
    p.add_argument('--audio_sr',     default=16_000, type=int)
    p.add_argument('--max_text_len', default=128,  type=int)
    p.add_argument('--amp',          action='store_true', default=True)
    p.add_argument('--threshold',    default=None, type=float,
                   help='Sigmoid threshold for binary prediction.  When None the '
                        "threshold stored in the checkpoint ('best_thresh') is "
                        'used; falls back to 0.5 if not present.')
    p.add_argument('--num_windows',  default=1,    type=int,
                   help='Number of 16-frame windows sampled per video.  '
                        'Window logits are averaged before thresholding.  '
                        '1 = deterministic (first frames); >1 = random sampling '
                        'across the full video (recommended: 5).')
    p.add_argument('--text_blend',  default=0.6,  type=float,
                   help='Blend weight for the text-only logit at inference. '
                        'final_prob = text_blend * text_prob + (1-text_blend) * full_prob. '
                        'Uses value stored in checkpoint when not specified here.')
    p.add_argument('--active_modalities', default='video,audio,text', type=str,
                   help='Comma-separated modalities for ablation (video,audio,text). '
                        'Default: all three.')
    p.add_argument('--output_format', default='probabilities', choices=['probabilities', 'labels_only'],
                   help='Output format: "probabilities" = video_id,prob_0,prob_1,label; '
                        '"labels_only" = video_id,label')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_lora_to_encoders(model, lora_r: int = 8, lora_alpha: int = 16) -> None:
    """Apply LoRA adapters to all three frozen encoders (mirrors train.py)."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "This checkpoint was trained with --use_lora. "
            "Please install peft: pip install peft>=0.10.0"
        )

    # VideoMAE style attention
    model.video_encoder = get_peft_model(
        model.video_encoder,
        LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=['query', 'value']),
    )
    # HuBERT style attention
    model.audio_encoder = get_peft_model(
        model.audio_encoder,
        LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=['q_proj', 'v_proj']),
    )
    # RoBERTa style attention
    model.text_encoder = get_peft_model(
        model.text_encoder,
        LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=['query', 'value']),
    )

    print(f'    LoRA adapters applied (r={lora_r}, alpha={lora_alpha})')

def load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
) -> tuple[ConflictAwareAHModel, AutoTokenizer, float, argparse.Namespace]:
    """Load model, tokenizer, best threshold, and original args from a checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    print(f'  Loading: {ckpt_path}')
    ckpt      = torch.load(ckpt_path, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])

    # Infer fusion_type from checkpoint keys (concat vs 6token)
    state_keys = set(ckpt['model'].keys())
    fusion_type = getattr(ckpt_args, 'fusion_type', None)
    if fusion_type is None:
        fusion_type = '6token' if any('fusion_transformer' in k for k in state_keys) else 'concat'

    # Detect whether this checkpoint was trained with LoRA
    use_lora = getattr(ckpt_args, 'use_lora', False)

    model_kwargs = dict(
        video_model=getattr(ckpt_args, 'video_model', 'MCG-NJU/videomae-base'),
        audio_model=getattr(ckpt_args, 'audio_model', 'facebook/hubert-base-ls960'),
        text_model =getattr(ckpt_args, 'text_model',  'SamLowe/roberta-base-go_emotions'),
        embed_dim  =getattr(ckpt_args, 'embed_dim',   768),
        hidden_dim =getattr(ckpt_args, 'hidden_dim',  512),
        num_transformer_layers=getattr(ckpt_args, 'num_layers', 2),
        num_heads  =getattr(ckpt_args, 'num_heads',   8),
        dropout    =0.0,            # no dropout at inference
        no_conflict=getattr(ckpt_args, 'no_conflict', False),
        fusion_type=fusion_type,
        conflict_type=getattr(ckpt_args, 'conflict_type', 'abs'),
        use_film=getattr(ckpt_args, 'use_film', False),
        fix_audio_mask=getattr(ckpt_args, 'fix_audio_mask', False),
        use_gated_diff=getattr(ckpt_args, 'use_gated_diff', False),
        use_gated_fusion=getattr(ckpt_args, 'use_gated_fusion', False),
        num_windows=getattr(ckpt_args, 'num_windows', 1),
    )

    # LoRA requires frozen encoders (adapters trained on frozen base)
    if use_lora:
        model_kwargs['freeze_encoders'] = True
        print('    LoRA checkpoint detected — freezing encoders, applying adapters')

    model = ConflictAwareAHModel(**model_kwargs).to(device)

    if use_lora:
        _apply_lora_to_encoders(
            model,
            lora_r=getattr(ckpt_args, 'lora_r', 8),
            lora_alpha=getattr(ckpt_args, 'lora_alpha', 16),
        )

    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'    epoch={ckpt["epoch"]}  best_f1={ckpt.get("best_f1", 0):.4f}')

    text_model_id = getattr(ckpt_args, 'text_model',
                            'SamLowe/roberta-base-go_emotions')
    tokenizer  = AutoTokenizer.from_pretrained(text_model_id, local_files_only=True)
    best_thresh = ckpt.get('best_thresh', 0.5)
    return model, tokenizer, best_thresh, ckpt_args


def run_one_pass(
    model:      ConflictAwareAHModel,
    loader:     DataLoader,
    tokenizer,
    device:     torch.device,
    amp:        bool,
    max_len:    int,
    text_blend: float = 0.6,
    use_window_transcript: bool = False,
    active_modalities: set | None = None,
) -> dict[str, float]:
    """
    Run one forward pass over the loader.

    Returns a dict mapping video_path → blended sigmoid probability.
    The model returns (full_logit, text_logit); the final probability is
        text_blend * sigmoid(text_logit) + (1-text_blend) * sigmoid(full_logit)
    """
    path_to_prob: dict[str, float] = {}

    with torch.no_grad():
        for batch in loader:
            video      = batch['video'].to(device)
            audio      = batch['audio'].to(device)
            audio_mask = batch['audio_mask'].to(device)

            transcript_key = 'transcript' if use_window_transcript else 'transcript_full'
            text_enc = tokenizer(
                batch[transcript_key],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            text_enc = {k: v.to(device) for k, v in text_enc.items()}

            with torch.cuda.amp.autocast(enabled=amp):
                full_logit, text_logit = model(video, audio, text_enc, audio_mask,
                                                active_modalities=active_modalities)

            full_prob = torch.sigmoid(full_logit.squeeze(-1))
            text_prob = torch.sigmoid(text_logit.squeeze(-1))
            blended   = text_blend * text_prob + (1.0 - text_blend) * full_prob

            probs = blended.cpu().tolist()
            if not isinstance(probs, list):
                probs = [probs]

            for path, prob in zip(batch['video_path'], probs):
                path_to_prob[path] = prob

    return path_to_prob


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _download_from_hf(repo_id: str) -> str:
    """Download best_model.pt from Hugging Face Hub. Returns local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")
    path = hf_hub_download(repo_id=repo_id, filename="best_model.pt", local_dir=None)
    return path


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint list: --hf_repo > --checkpoints > default
    if args.hf_repo:
        print(f'Downloading from Hugging Face: {args.hf_repo}')
        ckpt_paths = [_download_from_hf(args.hf_repo)]
        print(f'  → {ckpt_paths[0]}')
    elif args.checkpoints is not None:
        ckpt_paths = args.checkpoints
    elif args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = [_find_latest_best_ckpt()]

    print(f'Checkpoints : {ckpt_paths}')
    print(f'Split       : {args.split}')
    print(f'Num windows : {args.num_windows}')
    print(f'Device      : {device}')

    # ── Load all models (one per checkpoint) ──────────────────────────────
    models, tokenizers, thresh_votes, text_blend_votes = [], [], [], []
    use_window_transcript = False
    for ckpt_path in ckpt_paths:
        m, tok, thresh, ckpt_args = load_model_from_ckpt(ckpt_path, device)
        models.append(m)
        tokenizers.append(tok)
        thresh_votes.append(thresh)
        text_blend_votes.append(getattr(ckpt_args, 'text_blend', 0.6))
        use_window_transcript = use_window_transcript or getattr(ckpt_args, 'use_window_transcript', False)

    # Threshold: CLI arg > mean of checkpoint thresholds
    if args.threshold is not None:
        threshold = args.threshold
        print(f'Threshold   : {threshold:.3f} (from --threshold)')
    else:
        threshold = sum(thresh_votes) / len(thresh_votes)
        print(f'Threshold   : {threshold:.3f} (averaged from checkpoints)')

    # Text blend: CLI arg overrides; otherwise use checkpoint-stored value
    text_blend = args.text_blend if args.text_blend != 0.6 else (
        sum(text_blend_votes) / len(text_blend_votes))
    print(f'Text blend  : {text_blend:.2f}  (text*{text_blend:.2f} + full*{1-text_blend:.2f})')

    # Active modalities: parse comma-separated string into set
    active_modalities = set(args.active_modalities.split(','))
    print(f'Modalities  : {active_modalities}')

    # ── Multi-window + multi-checkpoint inference ─────────────────────────
    # video_path → list of sigmoid probs (one per window × per checkpoint)
    all_probs: dict[str, list[float]] = defaultdict(list)

    for ckpt_idx, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        print(f'\nCheckpoint {ckpt_idx + 1}/{len(models)}:')

        for win_idx in range(args.num_windows):
            # Window 0 is always deterministic (first frames); subsequent
            # windows use random sampling to cover the full video.
            random_crop = (win_idx > 0) or (args.num_windows == 1 and False)
            # For single-window keep deterministic; for multi-window
            # use random crop for all windows to get diverse coverage.
            if args.num_windows > 1:
                random_crop = True

            dataset = ABAW10_AH_Dataset(
                root=args.data_root,
                split=args.split,
                num_frames=args.num_frames,
                img_size=args.img_size,
                audio_sample_rate=args.audio_sr,
                random_frames_crop=random_crop,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=multimodal_collate_fn,
                pin_memory=(device.type == 'cuda'),
            )
            if ckpt_idx == 0 and win_idx == 0:
                print(f'  {len(dataset)} videos  (random_crop={random_crop})')

            print(f'  window {win_idx + 1}/{args.num_windows} '
                  f'(random={random_crop}) …', end=' ', flush=True)
            probs = run_one_pass(model, loader, tokenizer, device,
                                 args.amp, args.max_text_len, text_blend,
                                 use_window_transcript=use_window_transcript,
                                 active_modalities=active_modalities)
            for path, prob in probs.items():
                all_probs[path].append(prob)
            print('done')

    # ── Aggregate: average all probs per video ────────────────────────────
    video_paths  = sorted(all_probs.keys())
    final_probs  = [sum(all_probs[p]) / len(all_probs[p]) for p in video_paths]
    final_preds  = [int(prob > threshold) for prob in final_probs]

    # ── Evaluate if labels are available (split='test') ───────────────────
    if args.split == 'test':
        # Re-read labels from the split file
        split_file = os.path.join(args.data_root, 'data', 'split', 'test.txt')
        path_to_label: dict[str, int] = {}
        with open(split_file, 'r') as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    path_to_label[parts[0]] = int(parts[1])
        labels = [path_to_label.get(p, -1) for p in video_paths]
        labeled = [(p, t) for p, t in zip(final_preds, labels) if t >= 0]
        if labeled:
            from scripts.train import compute_metrics
            preds_l = [p for p, _ in labeled]
            tgts_l  = [t for _, t in labeled]
            m = compute_metrics(preds_l, tgts_l)
            print(f'\nEval on labeled test set:')
            print(f'  Macro F1 = {m["macro_f1"]:.4f}')
            print(f'  F1 A-H   = {m["f1_ah"]:.4f}')
            print(f'  F1 No-AH = {m["f1_no_ah"]:.4f}')
            print(f'  Accuracy = {m["accuracy"]:.4f}')

    # ── Write submission CSV ───────────────────────────────────────────────
    # Per challenge spec:
    #   With probabilities: video_id,probability_of_class_0,probability_of_class_1,label_prediction
    #   Labels only:        video_id,label_prediction
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if args.output_format == 'probabilities':
            writer.writerow(['video_id', 'probability_of_class_0', 'probability_of_class_1', 'label_prediction'])
            for path, pred, prob in zip(video_paths, final_preds, final_probs):
                prob_0 = 1.0 - prob  # P(absence of A/H)
                prob_1 = prob        # P(presence of A/H)
                writer.writerow([path, f'{prob_0:.6f}', f'{prob_1:.6f}', pred])
        else:
            writer.writerow(['video_id', 'label_prediction'])
            for path, pred in zip(video_paths, final_preds):
                writer.writerow([path, pred])

    ah_count    = sum(final_preds)
    no_ah_count = len(final_preds) - ah_count
    print(f'\nPredictions written to: {args.output}')
    print(f'  total         : {len(final_preds)}')
    print(f'  With A-H  (1) : {ah_count}  ({ah_count/len(final_preds)*100:.1f}%)')
    print(f'  No A-H    (0) : {no_ah_count}  ({no_ah_count/len(final_preds)*100:.1f}%)')
    print(f'  Threshold     : {threshold:.3f}')
    print(f'  Windows/video : {args.num_windows} × {len(ckpt_paths)} ckpts'
          f' = {args.num_windows * len(ckpt_paths)} passes')


if __name__ == '__main__':
    main()
