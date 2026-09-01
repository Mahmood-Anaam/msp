from .configuration_msp_audio import MSPAudioConfig
from .feature_extraction_msp_audio import MSPAudioFeatureExtractor
from .modeling_msp_audio import MSPAudioForCTC, MSPAudioModel
from .processing_msp_audio import MSPAudioProcessor


def registers():
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForCTC,
        AutoProcessor,
    )

    AutoConfig.register(MSPAudioConfig.model_type, MSPAudioConfig)
    AutoFeatureExtractor.register(MSPAudioConfig, MSPAudioFeatureExtractor)
    AutoProcessor.register(MSPAudioConfig, MSPAudioProcessor)
    AutoModel.register(MSPAudioConfig, MSPAudioModel)
    AutoModelForCTC.register(MSPAudioConfig, MSPAudioForCTC)

    MSPAudioConfig.register_for_auto_class("AutoConfig")
    MSPAudioFeatureExtractor.register_for_auto_class("AutoFeatureExtractor")
    MSPAudioProcessor.register_for_auto_class("AutoProcessor")
    MSPAudioModel.register_for_auto_class("AutoModel")
    MSPAudioForCTC.register_for_auto_class("AutoModelForCTC")


__all__ = [
    "MSPAudioConfig",
    "MSPAudioFeatureExtractor",
    "MSPAudioProcessor",
    "MSPAudioModel",
    "MSPAudioForCTC",
    "registers",
]
