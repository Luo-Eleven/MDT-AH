"""
Reusable building blocks shared across models in this package.
"""
from __future__ import annotations

import random
from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Learnable soft-attention pooling over a sequence.

    Maps (B, L, D) → (B, D) by computing a weighted sum of tokens, where
    weights come from a single linear query.  This is strictly more expressive
    than mean-pooling and costs almost nothing extra.

    Args:
        dim: feature dimension D.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x    : (B, L, D)
            mask : (B, L) bool tensor – True where tokens should be *ignored*
                   (same convention as PyTorch's `src_key_padding_mask`).
                   Pass None to attend over all tokens.
        Returns:
            (B, D)
        """
        scores = self.query(x)             # (B, L, 1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)   # (B, L, 1)
        return (weights * x).sum(dim=1)          # (B, D)


# ---------------------------------------------------------------------------
# DataLoader collate helper
# ---------------------------------------------------------------------------

def multimodal_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate for ABAW10_AH_Dataset samples.

    The main challenge is that audio windows can have different lengths after
    time-aligned cropping.  We pad all waveforms to the longest in the batch
    and return an `audio_mask` (True = padding position) so the model can
    ignore padded positions.

    Text transcripts are left as a list of strings so the caller can run them
    through a HuggingFace tokenizer *after* collation (typical pattern).

    Supports both single-window (T, C, H, W) and multi-window (K, T, C, H, W)
    video/audio shapes.

    Returns a dict with:
        video              : (B, T, C, H, W)     or (B, K, T, C, H, W)
        audio              : (B, max_samples)     or (B, K, max_samples)
        audio_mask         : (B, max_samples)     or (B, K, max_samples)
        transcript         : List[str]
        transcript_full    : List[str]
        transcript_chunks  : List[List[dict]]
        time_window        : List of tuples or List of lists
        label              : (B,)  long
        video_path         : List[str]
    """
    videos        = torch.stack([s['video']  for s in batch])
    labels        = torch.stack([s['label']  for s in batch])
    transcripts   = [s['transcript']       for s in batch]
    trans_full    = [s['transcript_full']   for s in batch]
    trans_chunks  = [s['transcript_chunks'] for s in batch]
    time_windows  = [s['time_window']       for s in batch]
    video_paths   = [s['video_path']        for s in batch]

    # Pad audio to the longest waveform in the batch.
    # Multi-window: audio is (K, samples) per sample — pad per-window across batch.
    audio_list = [s['audio'] for s in batch]
    sample_audio = audio_list[0]

    if sample_audio.dim() == 2:
        # Multi-window: each sample has (K, samples_variable)
        K = sample_audio.shape[0]
        # Find max length across all samples and windows
        max_len = max(a.shape[1] for a in audio_list)
        audio_padded = torch.zeros(len(batch), K, max_len)
        audio_mask   = torch.ones(len(batch), K, max_len, dtype=torch.bool)
        for i, a in enumerate(audio_list):
            audio_padded[i, :, :a.shape[1]] = a
            audio_mask[i,  :, :a.shape[1]] = False
    else:
        # Single-window: each sample has (samples_variable,)
        max_len      = max(a.shape[0] for a in audio_list)
        audio_padded = torch.zeros(len(batch), max_len)
        audio_mask   = torch.ones(len(batch), max_len, dtype=torch.bool)
        for i, a in enumerate(audio_list):
            audio_padded[i, :a.shape[0]] = a
            audio_mask[i,  :a.shape[0]] = False

    return {
        'video':             videos,
        'audio':             audio_padded,
        'audio_mask':        audio_mask,
        'transcript':        transcripts,
        'transcript_full':   trans_full,
        'transcript_chunks': trans_chunks,
        'time_window':       time_windows,
        'label':             labels,
        'video_path':        video_paths,
    }


# ---------------------------------------------------------------------------
# CutMix Data Augmentation
# ---------------------------------------------------------------------------

class CutMixCollate:
    """
    Collate wrapper that applies CutMix augmentation to a batch.

    Video CutMix: randomly replaces a spatial region across all frames of one
        sample with the same region from another sample, mixing labels by area ratio.
    Audio CutMix: randomly replaces a temporal segment of one sample's waveform
        with the corresponding segment from another sample, mixing labels by time ratio.

    The mixed label is stored directly in the batch dict as a float tensor
    replacing the original integer label. The `cutmix_mix` field is added
    to indicate whether CutMix was applied (bool per sample).

    Args:
        inner_collate: base collate function (e.g. multimodal_collate_fn).
        prob:   probability of applying CutMix to a batch.
        alpha:  Beta distribution concentration parameter for lambda sampling.
    """

    def __init__(self, inner_collate, prob: float = 0.5, alpha: float = 1.0):
        self.inner_collate = inner_collate
        self.prob = prob
        self.alpha = alpha

    def __call__(self, batch: List[dict]) -> dict:
        collated = self.inner_collate(batch)
        B = collated['video'].shape[0]

        if B < 2 or random.random() > self.prob:
            collated['cutmix_lambda'] = None
            collated['cutmix_perm'] = None
            return collated

        # ── Sample lambda and permutation ──────────────────────────────────
        lam = random.betavariate(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 for label consistency
        perm = torch.randperm(B)

        # ── Video CutMix (spatial) ─────────────────────────────────────────
        video = collated['video']          # (B, T, C, H, W) or (B, K, T, C, H, W)
        _, _, H, W = video.shape[-4:]   # last 4 dims are (T,C,H,W) or (K,T,C,H,W)→(T,C,H,W)
        has_kw = (video.dim() == 6)        # (B, K, T, C, H, W)

        cut_ratio = (1.0 - lam) ** 0.5     # area ratio for the cutout
        cut_w = max(1, int(W * cut_ratio))
        cut_h = max(1, int(H * cut_ratio))
        cx = random.randint(0, W - cut_w)
        cy = random.randint(0, H - cut_h)

        if has_kw:
            video[:, :, :, :, cy:cy + cut_h, cx:cx + cut_w] = \
                video[perm][:, :, :, :, cy:cy + cut_h, cx:cx + cut_w].clone()
        else:
            video[:, :, :, cy:cy + cut_h, cx:cx + cut_w] = \
                video[perm][:, :, :, cy:cy + cut_h, cx:cx + cut_w].clone()

        # ── Audio CutMix (temporal) ────────────────────────────────────────
        audio = collated['audio']          # (B, samples) or (B, K, samples)
        audio_mask = collated['audio_mask']
        has_ka = (audio.dim() == 3)        # (B, K, samples)
        audio_dim = 2 if has_ka else 1
        audio_len = audio.shape[audio_dim]

        cut_audio_len = max(1, int(audio_len * cut_ratio))
        audio_start = random.randint(0, audio_len - cut_audio_len)
        audio_slice = slice(audio_start, audio_start + cut_audio_len)

        if has_ka:
            audio[:, :, audio_slice] = audio[perm][:, :, audio_slice].clone()
            audio_mask[:, :, audio_slice] = audio_mask[perm][:, :, audio_slice].clone()
        else:
            audio[:, audio_slice] = audio[perm][:, audio_slice].clone()
            audio_mask[:, audio_slice] = audio_mask[perm][:, audio_slice].clone()

        # ── Mix labels ─────────────────────────────────────────────────────
        original_labels = collated['label'].float()
        mixed_labels = lam * original_labels + (1.0 - lam) * original_labels[perm]
        collated['label'] = mixed_labels  # replace int labels with float mix

        # Store CutMix metadata for loss computation awareness
        collated['cutmix_lambda'] = torch.full((B,), lam, dtype=torch.float32)
        collated['cutmix_perm'] = perm

        return collated
