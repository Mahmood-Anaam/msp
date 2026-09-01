from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import ModelOutput, logging

from msp.models.msp_audio.modeling_msp_audio import MSPAudioForCTC
from msp.models.msp_visual.modeling_msp_visual import MSPVisualForCTC

from .configuration_msp import MSPConfig
from .fusion import MSPFusionModel

logger = logging.get_logger(__name__)


@dataclass
class MSPModelOutput(ModelOutput):
    """Encoder and modality-aware fusion representations."""

    last_hidden_state: torch.FloatTensor | None = None
    padding_mask: torch.BoolTensor | None = None
    audio_input_lengths: torch.LongTensor | None = None
    visual_input_lengths: torch.LongTensor | None = None
    audio_hidden_state: torch.FloatTensor | None = None
    visual_hidden_state: torch.FloatTensor | None = None
    cross_attentions: tuple | None = None


@dataclass
class MSPCTCOutput(CausalLMOutput):
    """MSP CTC output with optional training-only auxiliary losses."""

    pass


def _compute_ctc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    blank: int,
    reduction: str,
    zero_infinity: bool,
) -> torch.Tensor:
    labels_mask = labels >= 0
    target_lengths = labels_mask.sum(dim=-1)
    flattened_targets = labels.masked_select(labels_mask)
    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        return F.ctc_loss(
            log_probs,
            flattened_targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )


class MSPPreTrainedModel(PreTrainedModel):
    config_class = MSPConfig
    base_model_prefix = "msp"
    main_input_name = "input_values"
    input_modalities = ["audio", "video"]
    supports_gradient_checkpointing = False
    all_tied_weights_keys = OrderedDict()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def dummy_inputs(self) -> dict:
        return {
            "input_values": torch.zeros(1, 16000, dtype=torch.float32),
            "padding_mask": torch.ones(1, 16000, dtype=torch.long),
            "pixel_values_videos": torch.zeros(1, 1, 25, 88, 88, dtype=torch.float32),
            "padding_mask_videos": torch.ones(1, 25, dtype=torch.long),
            "labels": torch.ones(1, 5, dtype=torch.long),
        }


class MSPModel(MSPPreTrainedModel):
    def __init__(self, config: MSPConfig):
        super().__init__(config)
        self.config = config
        self.audio_model = MSPAudioForCTC(config=config.audio_config)
        self.visual_model = MSPVisualForCTC(config=config.visual_config)
        self.fusion_model = MSPFusionModel(config)
        self.post_init()

    def encode_audio(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values, dtype=torch.long)
        if padding_mask.ndim != 2 or padding_mask.shape != input_values.shape[:2]:
            raise ValueError(
                "padding_mask must match input_values on batch and time axes."
            )

        outputs = self.audio_model.msp_audio.forward(
            input_values=input_values,
            padding_mask=padding_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        feature_mask = self.audio_model._get_feature_vector_attention_mask(
            hidden_states.size(1), padding_mask
        ).to(device=hidden_states.device, dtype=torch.bool)
        input_lengths = feature_mask.sum(dim=-1).to(torch.long)

        return hidden_states, feature_mask, input_lengths

    def encode_visual(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, num_frames = pixel_values_videos.shape[:3]
        if padding_mask_videos is None:
            padding_mask_videos = torch.ones(
                batch_size,
                num_frames,
                dtype=torch.long,
                device=pixel_values_videos.device,
            )

        if padding_mask_videos.ndim != 2 or tuple(padding_mask_videos.shape) != (
            batch_size,
            num_frames,
        ):
            raise ValueError(
                "padding_mask_videos must match pixel_values_videos on batch "
                "and time axes."
            )

        outputs = self.visual_model.msp_visual.forward(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
        )
        hidden_states = outputs.last_hidden_state
        feature_mask = outputs.padding_mask_videos
        if feature_mask is None:
            feature_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        feature_mask = feature_mask.to(device=hidden_states.device, dtype=torch.bool)
        if tuple(feature_mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(
                "The visual encoder returned a padding mask that does not "
                "match its hidden states."
            )
        input_lengths = feature_mask.sum(dim=-1).to(torch.long)
        return hidden_states, feature_mask, input_lengths

    def forward(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        padding_mask_videos: torch.Tensor | None = None,
        **kwargs,
    ) -> MSPModelOutput:

        if input_values is None or pixel_values_videos is None:
            raise ValueError("MSP requires synchronized audio and video inputs.")
        audio_states, audio_mask, audio_lengths = self.encode_audio(
            input_values, padding_mask
        )
        visual_states, visual_mask, visual_lengths = self.encode_visual(
            pixel_values_videos, padding_mask_videos
        )

        fusion_output = self.fusion_model.forward(
            audio_features=audio_states,
            video_features=visual_states,
            audio_mask=audio_mask,
            video_mask=visual_mask,
            **kwargs,
        )

        return MSPModelOutput(
            last_hidden_state=fusion_output.last_hidden_state,
            padding_mask=fusion_output.padding_mask,
            audio_input_lengths=audio_lengths,
            visual_input_lengths=visual_lengths,
            audio_hidden_state=audio_states,
            visual_hidden_state=visual_states,
            cross_attentions=fusion_output.cross_attentions,
        )


class MSPForCTC(MSPPreTrainedModel):
    def __init__(self, config: MSPConfig):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                "vocab_size must be set in MSPConfig to instantiate MSPForCTC."
            )

        self.msp = MSPModel(config=config)
        # lm head
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.fusion_hidden_size, config.vocab_size)
        self.post_init()

    def freeze_feature_encoder(self):
        self.msp.audio_model.freeze_feature_encoder()
        self.msp.visual_model.freeze_feature_encoder()

    def freeze_feature_encoders(self):
        self.freeze_feature_encoder()

    def freeze_base_model(self):
        self.msp.audio_model.freeze_base_model()
        self.msp.visual_model.freeze_base_model()

    def freeze_audio_branch(self):
        for param in self.msp.audio_model.parameters():
            param.requires_grad = False

    def freeze_visual_branch(self):
        for param in self.msp.visual_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        padding_mask_videos: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> MSPCTCOutput:
        if not self.training:
            if input_values is not None and pixel_values_videos is None:
                outputs = self.msp.audio_model(
                    input_values=input_values, padding_mask=padding_mask, labels=labels
                )
                return MSPCTCOutput(loss=outputs.loss, logits=outputs.logits)

            if pixel_values_videos is not None and input_values is None:
                outputs = self.msp.visual_model(
                    pixel_values_videos=pixel_values_videos,
                    padding_mask_videos=padding_mask_videos,
                    labels=labels,
                )
                return MSPCTCOutput(loss=outputs.loss, logits=outputs.logits)

        if input_values is None or pixel_values_videos is None:
            raise ValueError("MSPForCTC training requires both audio and video inputs.")

        outputs = self.msp.forward(
            input_values=input_values,
            pixel_values_videos=pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
        )

        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.lm_head(hidden_states)
        input_lengths = outputs.padding_mask.sum(-1).long()

        loss = msp_loss = audio_loss = visual_loss = None
        audio_logits = visual_logits = None

        if labels is not None and outputs.audio_hidden_state is not None:
            audio_logits = self.msp.audio_model.lm_head(
                self.msp.audio_model.dropout(outputs.audio_hidden_state)
            )

        if labels is not None and outputs.visual_hidden_state is not None:
            visual_logits = self.msp.visual_model.lm_head(
                self.msp.visual_model.dropout(outputs.visual_hidden_state)
            )

        if labels is not None:
            valid_labels = labels[labels >= 0]
            if valid_labels.numel() and valid_labels.max() >= self.config.vocab_size:
                raise ValueError("A label id is outside the configured vocabulary.")
            msp_loss = _compute_ctc_loss(
                logits,
                labels,
                input_lengths,
                self.config.pad_token_id,
                self.config.ctc_loss_reduction,
                self.config.ctc_zero_infinity,
            )
            loss = self.config.ctc_loss_msp_weight * msp_loss
            # The pretrained branch heads are training-only deep supervision;
            # audiovisual inference decodes the shared MSP head above.
            if audio_logits is not None:
                audio_loss = _compute_ctc_loss(
                    audio_logits,
                    labels,
                    outputs.audio_input_lengths,
                    self.config.pad_token_id,
                    self.config.ctc_loss_reduction,
                    self.config.ctc_zero_infinity,
                )
                loss = loss + self.config.ctc_loss_audio_weight * audio_loss
            if visual_logits is not None:
                visual_loss = _compute_ctc_loss(
                    visual_logits,
                    labels,
                    outputs.visual_input_lengths,
                    self.config.pad_token_id,
                    self.config.ctc_loss_reduction,
                    self.config.ctc_zero_infinity,
                )
                loss = loss + self.config.ctc_loss_visual_weight * visual_loss

        return MSPCTCOutput(
            loss=loss,
            logits=logits,
        )
