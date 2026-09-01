from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import ModelOutput, logging

from .configuration_msp_visual import MSPVisualConfig
from .modeling_avhubert import AVHubertModel

logger = logging.get_logger(__name__)


@dataclass
class MSPVisualOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    padding_mask_videos: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None


class MSPVisualPreTrainedModel(PreTrainedModel):
    config_class = MSPVisualConfig
    base_model_prefix = "msp_visual"
    main_input_name = "pixel_values_videos"
    input_modalities = "video"
    supports_gradient_checkpointing = False
    all_tied_weights_keys = OrderedDict()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MSPVisualModel(MSPVisualPreTrainedModel, AVHubertModel):
    def __init__(self, config: MSPVisualConfig):
        super().__init__(config.visual_config)
        self.config = config.visual_config
        self.feature_extractor_audio.requires_grad_(False)

    @property
    def dummy_inputs(self) -> dict:
        return {
            "pixel_values_videos": torch.zeros(1, 1, 10, 88, 88, dtype=torch.float32),
            "padding_mask_videos": torch.ones(1, 10, dtype=torch.long),
        }

    def forward(
        self,
        pixel_values_videos: torch.Tensor | None = None,
        padding_mask_videos: torch.Tensor | None = None,
        **kwargs,
    ) -> MSPVisualOutput:
        # Public MSP masks consistently use 1/True for valid positions.
        if padding_mask_videos is not None:
            padding_mask_videos = padding_mask_videos.to(dtype=torch.bool)

        feature, padding_mask = self.extract_finetune(
            source={
                "video": pixel_values_videos,  # shape [batch_size, num_channels=1, num_frames, height, width]
                "audio": None,
            },
            padding_mask=padding_mask_videos,  # shape [batch_size, num_frames]
        )

        return MSPVisualOutput(
            last_hidden_state=feature,  # shape [batch_size, num_frames, hidden_size]
            padding_mask_videos=padding_mask,  # shape [batch_size, num_frames]
            hidden_states=None,
            attentions=None,
        )


class MSPVisualForCTC(MSPVisualPreTrainedModel):
    def __init__(self, config: MSPVisualConfig):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                "vocab_size must be set in MSPVisualConfig to instantiate MSPVisualForCTC."
            )

        self.msp_visual = MSPVisualModel(config)
        for param in self.msp_visual.feature_extractor_audio.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(config.final_dropout)
        output_hidden_size = (
            config.visual_config.adim
            if hasattr(config.visual_config, "adim")
            else config.visual_config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

    @property
    def dummy_inputs(self) -> dict:
        return {
            "pixel_values_videos": torch.zeros(1, 1, 10, 88, 88, dtype=torch.float32),
            "padding_mask_videos": torch.ones(1, 10, dtype=torch.long),
        }

    def freeze_feature_encoder(self) -> None:
        for param in self.msp_visual.feature_extractor_video.parameters():
            param.requires_grad = False
        for param in self.msp_visual.feature_extractor_audio.parameters():
            param.requires_grad = False

    def freeze_base_model(self) -> None:
        for param in self.msp_visual.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:

        if labels is not None:
            valid_labels = labels[labels >= 0]
            if (
                valid_labels.numel() > 0
                and valid_labels.max() >= self.config.vocab_size
            ):
                raise ValueError(
                    f"Label value {valid_labels.max()} >= "
                    f"vocab_size={self.config.vocab_size}."
                )

        outputs = self.msp_visual(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self.dropout(outputs.last_hidden_state)
        padding_mask_videos = outputs.padding_mask_videos

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if padding_mask_videos is not None:
                input_lengths = (
                    padding_mask_videos.sum(-1)
                    .to(torch.long)
                    .to(pixel_values_videos.device)
                )

            else:
                input_lengths = torch.full(
                    (pixel_values_videos.shape[0],),
                    pixel_values_videos.shape[2],
                    dtype=torch.long,
                    device=pixel_values_videos.device,
                )

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
