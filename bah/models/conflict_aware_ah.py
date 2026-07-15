"""
Conflict-Aware Multimodal Model for Ambivalence/Hesitancy (A-H) recognition.

Architecture overview
─────────────────────

  video  (B, T, C, H, W)  ──► VideoMAE  ──► proj ──► attn-pool ──► v  (B, D)
  audio  (B, samples)      ──► HuBERT    ──► proj ──► attn-pool ──► a  (B, D)
  text   {tokenizer dict}  ──► RoBERTa   ──► proj ──► attn-pool ──► t  (B, D)
  (full video transcript used — not just the 16-frame window slice)

  Text-only logit (auxiliary late-fusion head):
      t  ──► text_head  →  text_logit  (B, 1)

  Conflict features  (explicit modal disagreement signal):
      |v − a|,  |v − t|,  |a − t|                              each (B, D)
      (optional) cos_va = Linear(v ⊙ a), cos_vt, cos_at         each (B, D)

  FiLM modulation (optional):
      v_mod = γ_v(t)*v + β_v(t),   a_mod = γ_a(t)*a + β_a(t)

  Multi-window training (optional):
      video (B, K, T, C, H, W) → reshape B*K → encode → reshape B,K,D → mean → (B,D)

  Fusion:
      cat tokens → TransformerEncoder × num_layers  →  flatten → classifier

  Full-fusion logit:
      LayerNorm → Linear → GELU → Dropout → Linear(1)  →  full_logit (B, 1)

  Final prediction (at inference):
      prob = text_blend * sigmoid(text_logit) + (1 − text_blend) * sigmoid(full_logit)

Training recipe
───────────────
  loss_full = BCE / FocalLoss(full_logit, y_smooth)
  loss_text = BCE(text_logit, y_smooth)
  loss = (1 - text_loss_weight) * loss_full + text_loss_weight * loss_text

Expected input shapes (batch size B = 4)
─────────────────────────────────────────
  video.shape  = (4, 16, 3, 224, 224)            — single window
               = (4, K, 16, 3, 224, 224)          — multi-window
  audio.shape  = (4, samples)                     — single window
               = (4, K, samples)                   — multi-window
  text         = AutoTokenizer(…)(full_transcripts, ...)   # full video transcripts!
"""
from __future__ import annotations

import os

# Force offline mode before any HuggingFace imports
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from bah.registry import MODELS
from bah.models.components import AttentionPool


@MODELS.register('ConflictAwareAHModel')
class ConflictAwareAHModel(nn.Module):
    """
    Conflict-Aware Multimodal Model.

    Args:
        video_model:            HuggingFace model ID for the video encoder.
                                Must be a VideoMAE-style model that accepts
                                `pixel_values` shaped (B, C, T, H, W).
        audio_model:            HuggingFace model ID for the audio encoder.
                                Supports wav2vec2-style and HuBERT models.
        text_model:             HuggingFace model ID for the text encoder.
                                Any BERT/RoBERTa-compatible model works.
        embed_dim:              Common embedding size after projection.
                                All three modalities are projected to this dim.
        hidden_dim:             Hidden size of the MLP classifier head.
        num_transformer_layers: Depth of the temporal Transformer.
        num_heads:              Number of attention heads in the Transformer.
        dropout:                Dropout rate used in Transformer + classifier.
        freeze_encoders:        If True, pre-trained encoder weights are frozen
                                (useful for limited GPU memory).
        unfreeze_top_k:         After freezing, unfreeze the top K transformer
                                layers of each encoder for task-specific tuning.
                                0 = keep everything frozen (default).
        no_conflict:            If True, zero out conflict features (ablation:
                                v+a+t only, no |v-a| etc.).
        fusion_type:            "6token" = stack [v,a,t,c_va,c_vt,c_at] as 6 tokens,
                                apply Transformer (attention over tokens), then
                                flatten. "concat" = legacy: single 6D token.
        conflict_type:          'abs' = absolute difference features (default).
                                'cosine' = element-wise product + projection.
                                'both' = both abs and cosine (6 conflict tokens).
        use_film:               If True, apply text-conditioned FiLM modulation
                                to video and audio embeddings before fusion.
        num_windows:            Number of uniformly-spaced frame windows per video.
                                1 = single window (default). >1 = multi-window
                                training with mean pooling across windows.
    """

    def __init__(
        self,
        video_model: str = 'MCG-NJU/videomae-base',
        audio_model: str = 'facebook/hubert-base-ls960',
        text_model:  str = 'SamLowe/roberta-base-go_emotions',
        embed_dim:   int = 768,
        hidden_dim:  int = 512,
        num_transformer_layers: int = 2,
        num_heads:   int = 8,
        dropout:     float = 0.3,
        freeze_encoders: bool = False,
        unfreeze_top_k:  int  = 0,
        no_conflict: bool = False,
        fusion_type: str = '6token',
        conflict_type: str = 'abs',
        use_film: bool = False,
        num_windows: int = 1,
        fix_audio_mask: bool = False,
        use_gated_diff: bool = False,
        use_gated_fusion: bool = False,
    ):
        super().__init__()
        self.no_conflict = no_conflict
        self.fusion_type = fusion_type if fusion_type in ('6token', 'concat') else '6token'
        self.conflict_type = conflict_type if conflict_type in ('abs', 'cosine', 'both') else 'abs'
        self.use_film = use_film
        self.num_windows = num_windows
        self.fix_audio_mask = fix_audio_mask
        self.use_gated_diff = use_gated_diff
        self.use_gated_fusion = use_gated_fusion

        # ── Pre-trained encoders (all loaded via AutoModel for uniformity) ─
        # local_files_only=True avoids network calls when the models are already cached
        self.video_encoder = AutoModel.from_pretrained(video_model, local_files_only=True)
        self.audio_encoder = AutoModel.from_pretrained(audio_model, local_files_only=True)
        self.text_encoder  = AutoModel.from_pretrained(text_model, local_files_only=True)

        if freeze_encoders:
            for encoder in (self.video_encoder, self.audio_encoder, self.text_encoder):
                for param in encoder.parameters():
                    param.requires_grad = False

        # Optionally unfreeze the top K transformer layers for fine-tuning.
        # These layers see a lower learning-rate (set in the optimizer param_groups).
        if freeze_encoders and unfreeze_top_k > 0:
            for encoder in (self.video_encoder, self.audio_encoder, self.text_encoder):
                self._unfreeze_top_layers(encoder, unfreeze_top_k)

        # ── Linear projections → common embed_dim ───────────────────────
        # Each encoder may have a different hidden_size; projections decouple
        # the choice of pre-trained model from the fusion architecture.
        self.video_proj = nn.Linear(self.video_encoder.config.hidden_size, embed_dim)
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, embed_dim)
        self.text_proj  = nn.Linear(self.text_encoder.config.hidden_size,  embed_dim)

        # ── Per-modality attention pooling ───────────────────────────────
        self.video_pool = AttentionPool(embed_dim)
        self.audio_pool = AttentionPool(embed_dim)
        self.text_pool  = AttentionPool(embed_dim)

        # ── FiLM modulation layers (optional) ─────────────────────────────
        if self.use_film:
            self.film_v = nn.Linear(embed_dim, embed_dim * 2)
            self.film_a = nn.Linear(embed_dim, embed_dim * 2)

        # ── Gated-difference layers (optional) ──────────────────────────
        if self.use_gated_diff:
            self.diff_gate_va = nn.Linear(embed_dim * 2, embed_dim)
            self.diff_gate_vt = nn.Linear(embed_dim * 2, embed_dim)
            self.diff_gate_at = nn.Linear(embed_dim * 2, embed_dim)

        # ── Cosine-style conflict projections (optional) ──────────────────
        if self.conflict_type in ('cosine', 'both'):
            self.cos_va_proj = nn.Linear(embed_dim, embed_dim)
            self.cos_vt_proj = nn.Linear(embed_dim, embed_dim)
            self.cos_at_proj = nn.Linear(embed_dim, embed_dim)

        # ── Compute number of tokens dynamically ──────────────────────────
        #   modality tokens: always v, a, t = 3
        if self.no_conflict:
            num_conflict_tokens = 0
        elif self.conflict_type == 'both':
            num_conflict_tokens = 6   # 3 abs + 3 cosine
        else:
            num_conflict_tokens = 3   # abs only or cosine only

        num_tokens = 3 + num_conflict_tokens
        fusion_dim = embed_dim * num_tokens

        # ── Fusion module ───────────────────────────────────────────────
        if self.use_gated_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.fusion_transformer = None
            self.temporal_transformer = None
        elif self.fusion_type == '6token':
            # num_tokens tokens of dim D; attention over tokens is meaningful
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.fusion_transformer = nn.TransformerEncoder(
                enc_layer,
                num_layers=num_transformer_layers,
            )
            self.temporal_transformer = None  # legacy
        else:
            # Legacy: 1 token of dim fusion_dim
            enc_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(
                enc_layer,
                num_layers=num_transformer_layers,
            )
            self.fusion_transformer = None

        # ── Full-fusion classifier head ──────────────────────────────────
        if not self.use_gated_fusion:
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.classifier = None

        # ── Text-only auxiliary head (late-fusion branch) ─────────────────
        # Trained jointly to make the text embedding independently discriminative.
        # At inference, blend: text_blend * sigmoid(text_logit) + (1-text_blend)
        # * sigmoid(full_logit).  Since RoBERTa-GoEmotions text alone achieves
        # ~77% F1, weighting it more heavily at inference consistently helps.
        self.text_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helper: partial encoder unfreezing
    # ------------------------------------------------------------------

    @staticmethod
    def _unfreeze_top_layers(encoder: nn.Module, k: int) -> None:
        """
        Unfreeze the top `k` transformer layers of `encoder`.

        Handles the three common attribute paths used by HuggingFace models:
          - ViT / VideoMAE:  encoder.encoder.layer
          - Wav2Vec2 / HuBERT: encoder.encoder.layers
          - BERT / RoBERTa: encoder.encoder.layer
        Falls back to unfreezing all parameters if no matching path is found.
        """
        layers = None
        for attr_path in ('encoder.layer', 'encoder.layers', 'encoder.blocks'):
            obj = encoder
            found = True
            for part in attr_path.split('.'):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and hasattr(obj, '__len__') and len(obj) > 0:
                layers = obj
                break

        if layers is None:
            for param in encoder.parameters():
                param.requires_grad = True
            return

        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Per-modality encoding
    # ------------------------------------------------------------------

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, T, C, H, W)  –  dataset output format
        Returns:
            (B, embed_dim)

        HuggingFace VideoMAE expects pixel_values in (B, T, C, H, W) order,
        which matches the dataset output directly – no permute needed.
        """
        hidden = self.video_encoder(pixel_values=video).last_hidden_state
        hidden = self.video_proj(hidden)         # (B, L_v, embed_dim)
        return self.video_pool(hidden)           # (B, embed_dim)

    def _encode_audio(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            audio:      (B, samples)   –  raw mono waveform at 16 kHz
            audio_mask: (B, samples)   –  True at padding positions
                        (produced by `multimodal_collate_fn`)
        Returns:
            (B, embed_dim)
        """
        # wav2vec2 accepts an attention_mask of 1=real, 0=pad
        attn_mask = (~audio_mask).long() if audio_mask is not None else None
        hidden = self.audio_encoder(
            input_values=audio,
            attention_mask=attn_mask,
        ).last_hidden_state
        hidden = self.audio_proj(hidden)         # (B, L_a, embed_dim)

        # Build a sequence-level mask for attention pooling from the
        # encoder output, which is down-sampled relative to the input.
        pool_mask = None
        if audio_mask is not None:
            L_a = hidden.shape[1]
            if self.fix_audio_mask:
                pool_mask = F.adaptive_avg_pool1d(
                    audio_mask.float().unsqueeze(1), L_a
                ).squeeze(1) > 0.5
            else:
                pool_mask = audio_mask[:, ::audio_mask.shape[1] // L_a][:, :L_a]

        return self.audio_pool(hidden, mask=pool_mask)   # (B, embed_dim)

    def _encode_text(
        self,
        text: dict,
    ) -> torch.Tensor:
        """
        Args:
            text: output of a HuggingFace tokenizer (input_ids, attention_mask, …)
        Returns:
            (B, embed_dim)
        """
        hidden = self.text_encoder(**text).last_hidden_state   # (B, L_t, D)
        hidden = self.text_proj(hidden)                        # (B, L_t, embed_dim)

        # BERT attention_mask: 1 = real token, 0 = padding → invert for pool_mask
        text_pad_mask = (text.get('attention_mask') == 0) if 'attention_mask' in text else None
        return self.text_pool(hidden, mask=text_pad_mask)      # (B, embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video:      torch.Tensor,           # (B, T, C, H, W) or (B, K, T, C, H, W)
        audio:      torch.Tensor,           # (B, samples) or (B, K, samples)
        text:       dict,                   # tokenizer output dict (full transcript)
        audio_mask: torch.Tensor | None = None,  # (B, samples) or (B, K, samples), True=pad
        active_modalities: set | None = None,    # subset of {'video','audio','text'}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            full_logit : (B, 1)  –  multimodal fusion logit.
            text_logit : (B, 1)  –  text-only logit (auxiliary late-fusion head).

        active_modalities: when provided, embeddings for excluded modalities are
            zeroed before fusion.  This enables modality ablation studies without
            changing the model architecture or training code.
            Default (None) = all three modalities active.

        Multi-window mode (num_windows > 1):
            video shape (B, K, T, C, H, W) → reshape (B*K, T, C, H, W) → encode
            → reshape (B, K, D) → mean-pool → (B, D). Same for audio.

        Training:
            loss = (1 - w) * BCE(full_logit, y) + w * BCE(text_logit, y)

        Inference:
            prob = text_blend * sigmoid(text_logit)
                 + (1 - text_blend) * sigmoid(full_logit)
        """
        if active_modalities is None:
            active_modalities = {'video', 'audio', 'text'}

        # ── Get batch size ────────────────────────────────────────────────
        B = video.shape[0]

        # ── Per-modality embeddings ──────────────────────────────────────
        # Multi-window: reshape (B, K, ...) → (B*K, ...), encode, reshape back
        if self.num_windows > 1 and video.dim() == 6:
            # video: (B, K, T, C, H, W)
            K = video.shape[1]
            video_flat = video.reshape(B * K, *video.shape[2:])
            v = self._encode_video(video_flat)           # (B*K, D)
            v = v.reshape(B, K, -1).mean(dim=1)          # (B, D)
        else:
            v = self._encode_video(video)                # (B, D)

        if self.num_windows > 1 and audio.dim() == 3:
            # audio: (B, K, samples), audio_mask: (B, K, samples)
            K = audio.shape[1]
            audio_flat = audio.reshape(B * K, -1)
            if audio_mask is not None:
                am_flat = audio_mask.reshape(B * K, -1)
                a = self._encode_audio(audio_flat, am_flat)  # (B*K, D)
            else:
                a = self._encode_audio(audio_flat)            # (B*K, D)
            a = a.reshape(B, K, -1).mean(dim=1)               # (B, D)
        else:
            a = self._encode_audio(audio, audio_mask)         # (B, D)

        t = self._encode_text(text)                           # (B, D)

        # Zero out inactive modalities (ablation).  The encoders still run
        # (simplest approach; avoids conditional graph branching), but their
        # output is masked so they cannot influence the fusion or any loss.
        if 'video' not in active_modalities:
            v = torch.zeros_like(v)
        if 'audio' not in active_modalities:
            a = torch.zeros_like(a)
        if 'text' not in active_modalities:
            t = torch.zeros_like(t)

        # ── Text-only auxiliary logit ─────────────────────────────────────
        text_logit = self.text_head(t)             # (B, 1)

        # ── FiLM modulation (optional) ────────────────────────────────────
        if self.use_film:
            gamma_v, beta_v = self.film_v(t).chunk(2, dim=-1)   # each (B, D)
            gamma_a, beta_a = self.film_a(t).chunk(2, dim=-1)
            v = gamma_v * v + beta_v
            a = gamma_a * a + beta_a
            # t stays unchanged as the control signal

        # ── Conflict features ────────────────────────────────────────────
        tokens = [v, a, t]   # modality tokens

        if not self.no_conflict:
            if self.conflict_type in ('abs', 'both'):
                if self.use_gated_diff:
                    gate_va = torch.sigmoid(self.diff_gate_va(torch.cat([v, a], dim=-1)))
                    gate_vt = torch.sigmoid(self.diff_gate_vt(torch.cat([v, t], dim=-1)))
                    gate_at = torch.sigmoid(self.diff_gate_at(torch.cat([a, t], dim=-1)))
                    conflict_va = gate_va * torch.abs(v - a)
                    conflict_vt = gate_vt * torch.abs(v - t)
                    conflict_at = gate_at * torch.abs(a - t)
                else:
                    conflict_va = torch.abs(v - a)
                    conflict_vt = torch.abs(v - t)
                    conflict_at = torch.abs(a - t)
                tokens.extend([conflict_va, conflict_vt, conflict_at])

            if self.conflict_type in ('cosine', 'both'):
                cos_va = self.cos_va_proj(v * a)
                cos_vt = self.cos_vt_proj(v * t)
                cos_at = self.cos_at_proj(a * t)
                tokens.extend([cos_va, cos_vt, cos_at])

        # ── Full-fusion logit ─────────────────────────────────────────────
        if self.use_gated_fusion:
            stacked = torch.stack(tokens, dim=1)          # (B, N, D)
            weights = self.fusion_gate(stacked).squeeze(-1)  # (B, N)
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)  # (B, N, 1)
            fused = (weights * stacked).sum(dim=1)        # (B, D)
            full_logit = self.fusion_proj(fused)          # (B, 1)
        elif self.fusion_type == '6token':
            # Stack as num_tokens tokens (B, num_tokens, D); Transformer attends over tokens
            fusion = torch.stack(tokens, dim=1)        # (B, N, D)
            x = self.fusion_transformer(fusion)         # (B, N, D)
            x = x.flatten(1)                            # (B, N*D)
            full_logit = self.classifier(x)             # (B, 1)
        else:
            # Legacy: concat to (B, N*D), single token
            fusion = torch.cat(tokens, dim=-1)          # (B, N*D)
            x = self.temporal_transformer(fusion.unsqueeze(1))  # (B, 1, N*D)
            x = x.squeeze(1)  # (B, N*D)
            full_logit = self.classifier(x)             # (B, 1)

        return full_logit, text_logit
