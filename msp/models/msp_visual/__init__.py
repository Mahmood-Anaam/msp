from .configuration_avhubert import AVHubertConfig
from .configuration_msp_visual import MSPVisualConfig
from .modeling_avhubert import AVHubertModel
from .modeling_msp_visual import MSPVisualForCTC, MSPVisualModel
from .processing_msp_visual import MSPVisualProcessor
from .video_processing_msp_visual import MSPVisualVideoProcessor


def registers():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCTC,
        AutoProcessor,
        AutoVideoProcessor,
    )

    AutoConfig.register(MSPVisualConfig.model_type, MSPVisualConfig)
    AutoVideoProcessor.register(MSPVisualConfig, MSPVisualVideoProcessor)
    AutoProcessor.register(MSPVisualConfig, MSPVisualProcessor)
    AutoModel.register(MSPVisualConfig, MSPVisualModel)
    AutoModelForCTC.register(MSPVisualConfig, MSPVisualForCTC)

    MSPVisualConfig.register_for_auto_class("AutoConfig")
    MSPVisualVideoProcessor.register_for_auto_class("AutoVideoProcessor")
    MSPVisualProcessor.register_for_auto_class("AutoProcessor")
    MSPVisualModel.register_for_auto_class("AutoModel")
    MSPVisualForCTC.register_for_auto_class("AutoModelForCTC")


__all__ = [
    "AVHubertConfig",
    "AVHubertModel",
    "MSPVisualConfig",
    "MSPVisualVideoProcessor",
    "MSPVisualProcessor",
    "MSPVisualModel",
    "MSPVisualForCTC",
    "registers",
]
