from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_msp_audio import MSPAudioConfig

logger = logging.get_logger(__name__)


@dataclass
class MSPAudioOutput(ModelOutput):
    """
    Output for MSPAudioForCTC.

    Args:
        loss: CTC loss (if labels provided).
        logits: Raw per-frame log-softmax scores of shape (B, T, vocab_size).
        hidden_states: Optional tuple of encoder hidden states.
        attentions: Optional tuple of attention weights.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class MSPAudioPreTrainedModel(Wav2Vec2PreTrainedModel):
    config_class = MSPAudioConfig
    base_model_prefix = "msp_audio"
    main_input_name = "input_values"
    input_modalities = "audio"
    all_tied_weights_keys = OrderedDict()


class MSPAudioModel(MSPAudioPreTrainedModel, Wav2Vec2Model):
    """
    Wav2Vec2 encoder wrapped as MSPAudioModel.
    """

    def __init__(self, config: MSPAudioConfig):
        super().__init__(config)
        self.config = config

    @property
    def dummy_inputs(self) -> dict:
        return {
            "input_values": torch.zeros(1, 16000, dtype=torch.float32),
            "padding_mask": torch.ones(1, 16000, dtype=torch.long),
        }

    def forward(
        self,
        input_values: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            input_values: Raw waveform tensor of shape (B, T).
            padding_mask: Boolean mask of shape (B, T); 1 = valid, 0 = padded.
            output_attentions: Return attention weights if True.
            output_hidden_states: Return all hidden states if True.

        Returns:
            Wav2Vec2BaseModelOutput with last_hidden_state of shape (B, T', D).
        """
        return super().forward(
            input_values=input_values,
            attention_mask=padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class MSPAudioForCTC(MSPAudioPreTrainedModel):
    def __init__(self, config: MSPAudioConfig):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                "vocab_size must be set in the config to instantiate MSPAudioForCTC."
            )

        self.msp_audio = MSPAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        output_hidden_size = (
            config.output_hidden_size if config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

    @property
    def dummy_inputs(self) -> dict:
        """Minimal inputs for model tracing."""
        return {
            "input_values": torch.zeros(1, 16000, dtype=torch.float32),
            "padding_mask": torch.ones(1, 16000, dtype=torch.long),
        }

    def freeze_feature_encoder(self) -> None:
        self.msp_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self) -> None:
        for param in self.msp_audio.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> MSPAudioOutput:
        """
        Args:
            input_values: Raw waveform of shape (B, T).
            padding_mask: Boolean mask of shape (B, T); 1 = valid frame.
            output_attentions: Return attention weights.
            output_hidden_states: Return all hidden states.
            labels: Token ids of shape (B, L); -100 entries are ignored.

        Returns:
            MSPAudioOutput with loss (if labels given), logits, hidden_states,
            and attentions.
        """

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label value {labels.max()} exceeds vocab_size={self.config.vocab_size}."
            )

        outputs = self.msp_audio(
            input_values=input_values,
            padding_mask=padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self.dropout(outputs[0])
        logits = self.lm_head(hidden_states)  # (B, T', vocab_size)

        loss = None
        if labels is not None:
            # Compute CTC input lengths from padding mask
            padding_mask = (
                padding_mask
                if padding_mask is not None
                else torch.ones_like(
                    input_values, dtype=torch.long, device=input_values.device
                )
            )
            input_lengths = self._get_feat_extract_output_lengths(
                padding_mask.sum(-1)
            ).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)  # (T', B, vocab_size)

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

        return MSPAudioOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
