from typing import Literal

from transformers import PretrainedConfig
from transformers.utils import logging

from msp.models.msp_audio.configuration_msp_audio import MSPAudioConfig
from msp.models.msp_visual.configuration_msp_visual import MSPVisualConfig

logger = logging.get_logger(__name__)


class MSPConfig(PretrainedConfig):
    model_type = "msp"

    sub_configs = {
        "audio_config": MSPAudioConfig,
        "visual_config": MSPVisualConfig,
    }

    def __init__(
        self,
        audio_config: MSPAudioConfig | dict | None = None,
        visual_config: MSPVisualConfig | dict | None = None,
        fusion_hidden_size: int = 768,
        fusion_num_attention_heads: int = 12,
        fusion_intermediate_size: int = 3072,
        fusion_hidden_act: str = "gelu",
        fusion_activation_dropout: float = 0.1,
        fusion_attention_dropout: float = 0.1,
        fusion_hidden_dropout: float = 0.1,
        fusion_layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        final_dropout: float = 0.1,
        vocab_size: int = 32,
        ctc_loss_reduction: Literal["mean", "sum", "none"] = "mean",
        ctc_zero_infinity: bool = True,
        ctc_loss_audio_weight: float = 0.2,
        ctc_loss_visual_weight: float = 0.2,
        ctc_loss_msp_weight: float = 0.6,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # encoders configs
        if isinstance(audio_config, dict):
            audio_config = MSPAudioConfig(**audio_config)
        elif audio_config is None:
            audio_config = MSPAudioConfig()

        if isinstance(visual_config, dict):
            visual_config = MSPVisualConfig(**visual_config)
        elif visual_config is None:
            visual_config = MSPVisualConfig()

        self.audio_config = audio_config
        self.visual_config = visual_config

        # fusion configs
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_attention_heads = fusion_num_attention_heads
        self.fusion_intermediate_size = fusion_intermediate_size
        self.fusion_hidden_act = fusion_hidden_act
        self.fusion_activation_dropout = fusion_activation_dropout
        self.fusion_attention_dropout = fusion_attention_dropout
        self.fusion_hidden_dropout = fusion_hidden_dropout
        self.fusion_layer_norm_eps = fusion_layer_norm_eps
        self.initializer_range = initializer_range

        # lm heead configs
        self.final_dropout = final_dropout
        self.vocab_size = vocab_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_loss_audio_weight = ctc_loss_audio_weight
        self.ctc_loss_visual_weight = ctc_loss_visual_weight
        self.ctc_loss_msp_weight = ctc_loss_msp_weight
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
