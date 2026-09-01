from .configuration_msp import MSPConfig
from .modeling_msp import MSPForCTC, MSPModel
from .processing_msp import MSPProcessor


def registers():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCTC,
        AutoProcessor,
    )

    AutoConfig.register(MSPConfig.model_type, MSPConfig)
    AutoProcessor.register(MSPConfig, MSPProcessor)
    AutoModel.register(MSPConfig, MSPModel)
    AutoModelForCTC.register(MSPConfig, MSPForCTC)
    MSPConfig.register_for_auto_class("AutoConfig")
    MSPProcessor.register_for_auto_class("AutoProcessor")
    MSPModel.register_for_auto_class("AutoModel")
    MSPForCTC.register_for_auto_class("AutoModelForCTC")


__all__ = [
    "MSPProcessor",
    "MSPConfig",
    "MSPModel",
    "MSPForCTC",
    "registers",
]
