from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput, logging

from .configuration_msp import MSPConfig

logger = logging.get_logger(__name__)


@dataclass
class MSPFusionOutput(ModelOutput):
    """Outputs produced by the modality-aware fusion network."""

    last_hidden_state: torch.FloatTensor | None = None
    padding_mask: torch.BoolTensor | None = None
    cross_attentions: tuple | None = None


class FeedForward(nn.Module):
    def __init__(self, config: MSPConfig):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.fusion_activation_dropout)

        self.intermediate_dense = nn.Linear(
            config.fusion_hidden_size, config.fusion_intermediate_size
        )
        if isinstance(config.fusion_hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.fusion_hidden_act]
        else:
            self.intermediate_act_fn = config.fusion_hidden_act

        self.output_dense = nn.Linear(
            config.fusion_intermediate_size, config.fusion_hidden_size
        )
        self.output_dropout = nn.Dropout(config.fusion_hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class MSPFusionModel(nn.Module):
    def __init__(self, config: MSPConfig):
        super().__init__()
        self.config = config

        self.audio_projection = nn.Sequential(
            nn.Identity()
            if config.audio_config.hidden_size == config.fusion_hidden_size
            else nn.Linear(config.audio_config.hidden_size, config.fusion_hidden_size),
            nn.LayerNorm(config.fusion_hidden_size, eps=config.fusion_layer_norm_eps),
        )

        self.visual_projection = nn.Sequential(
            nn.Identity()
            if config.visual_config.hidden_size == config.fusion_hidden_size
            else nn.Linear(config.visual_config.hidden_size, config.fusion_hidden_size),
            nn.LayerNorm(config.fusion_hidden_size, eps=config.fusion_layer_norm_eps),
        )

        self.av_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_size,
            num_heads=config.fusion_num_attention_heads,
            dropout=config.fusion_attention_dropout,
            batch_first=True,
        )
        self.va_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_size,
            num_heads=config.fusion_num_attention_heads,
            dropout=config.fusion_attention_dropout,
            batch_first=True,
        )
        self.av_layer_norm = nn.LayerNorm(
            config.fusion_hidden_size, eps=config.fusion_layer_norm_eps
        )
        self.va_layer_norm = nn.LayerNorm(
            config.fusion_hidden_size, eps=config.fusion_layer_norm_eps
        )

        self.fusion_layer = nn.Linear(
            config.fusion_hidden_size * 2, config.fusion_hidden_size
        )
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.fusion_hidden_size, eps=config.fusion_layer_norm_eps
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> MSPFusionOutput:

        if audio_features is None or video_features is None:
            raise ValueError(
                "MSP fusion requires both audio_features and video_features."
            )

        if audio_features.size(0) != video_features.size(0):
            raise ValueError("Audio and video batch sizes must match.")

        audio_only_encoder_out = self.audio_projection(audio_features)
        video_only_encoder_out = self.visual_projection(video_features)

        audio_mask = (
            torch.ones(
                audio_only_encoder_out.shape[:2],
                device=audio_only_encoder_out.device,
                dtype=torch.bool,
            )
            if audio_mask is None
            else audio_mask.to(device=audio_only_encoder_out.device, dtype=torch.bool)
        )
        video_mask = (
            torch.ones(
                video_only_encoder_out.shape[:2],
                device=video_only_encoder_out.device,
                dtype=torch.bool,
            )
            if video_mask is None
            else video_mask.to(device=video_only_encoder_out.device, dtype=torch.bool)
        )

        if (
            audio_mask.shape != audio_only_encoder_out.shape[:2]
            or video_mask.shape != video_only_encoder_out.shape[:2]
        ):
            raise ValueError(
                "Each padding mask must match its encoded feature sequence."
            )
        if (~audio_mask.any(dim=1)).any() or (~video_mask.any(dim=1)).any():
            raise ValueError(
                "Every sample must contain valid audio and video features."
            )

        # Cross Attention 1: Audio attends to Video
        av_attn_out, av_attn = self.av_attn.forward(
            query=audio_only_encoder_out,
            key=video_only_encoder_out,
            value=video_only_encoder_out,
            key_padding_mask=~video_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        av_encoder_out = audio_only_encoder_out + av_attn_out
        av_encoder_out = self.av_layer_norm(av_encoder_out)

        # Cross Attention 2: Video attends to Audio
        va_attn_out, va_attn = self.va_attn.forward(
            query=video_only_encoder_out,
            key=audio_only_encoder_out,
            value=audio_only_encoder_out,
            key_padding_mask=~audio_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        va_encoder_out = video_only_encoder_out + va_attn_out
        va_encoder_out = self.va_layer_norm(va_encoder_out)

        # Upsampling video_encoder_out to audio_encoder_output's time resolution
        # Correctly transposing before interpolating to operate on the time dimension
        va_encoder_out_perm = va_encoder_out.permute(0, 2, 1)
        va_encoder_out_upsampled = F.interpolate(
            va_encoder_out_perm,
            size=audio_only_encoder_out.shape[1],
            mode="nearest",
        )
        va_encoder_out = va_encoder_out_upsampled.permute(0, 2, 1)

        # Concatenate on feature dimension and project
        fused = self.fusion_layer(torch.cat((av_encoder_out, va_encoder_out), dim=2))

        # Feed-Forward Networks (FFN) for  Fusion Encoder Output
        fused = self.feed_forward(fused)
        output = self.final_layer_norm(fused)

        return MSPFusionOutput(
            last_hidden_state=output,
            padding_mask=audio_mask,
            cross_attentions=((av_attn, va_attn),),
        )
