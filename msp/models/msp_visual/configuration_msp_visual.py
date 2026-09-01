from transformers import PretrainedConfig
from transformers.utils import logging

from .configuration_avhubert import AVHubertConfig

logger = logging.get_logger(__name__)


class MSPVisualConfig(PretrainedConfig):
    model_type = "msp_visual"
    sub_configs = {"visual_config": AVHubertConfig}

    def __init__(
        self,
        visual_config: AVHubertConfig | None | dict = None,
        final_dropout: float = 0.1,
        vocab_size: int = 32,
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if visual_config is not None:
            if isinstance(visual_config, dict):
                self.visual_config = AVHubertConfig(**visual_config)
            elif isinstance(visual_config, AVHubertConfig):
                self.visual_config = visual_config
            else:
                raise ValueError("visual_config must be a dict or AVHubertConfig.")

        else:
            self.visual_config = AVHubertConfig()

        self.final_dropout = final_dropout
        self.vocab_size = vocab_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def hidden_size(self):
        return self.visual_config.adim
