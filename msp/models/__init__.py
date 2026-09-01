from .msp import (
    MSPConfig,
    MSPForCTC,
    MSPModel,
    MSPProcessor,
)
from .msp import (
    registers as msp_registers,
)
from .msp_audio import (
    MSPAudioConfig,
    MSPAudioFeatureExtractor,
    MSPAudioForCTC,
    MSPAudioModel,
    MSPAudioProcessor,
)
from .msp_audio import (
    registers as msp_audio_registers,
)
from .msp_visual import (
    AVHubertConfig,
    AVHubertModel,
    MSPVisualConfig,
    MSPVisualForCTC,
    MSPVisualModel,
    MSPVisualProcessor,
    MSPVisualVideoProcessor,
)
from .msp_visual import (
    registers as msp_visual_registers,
)


def registers():
    msp_audio_registers()
    msp_visual_registers()
    msp_registers()


__all__ = [
    # Audio
    "MSPAudioConfig",
    "MSPAudioFeatureExtractor",
    "MSPAudioProcessor",
    "MSPAudioModel",
    "MSPAudioForCTC",
    # Visual
    "AVHubertConfig",
    "AVHubertModel",
    "MSPVisualConfig",
    "MSPVisualVideoProcessor",
    "MSPVisualProcessor",
    "MSPVisualModel",
    "MSPVisualForCTC",
    # Multimodal
    "MSPConfig",
    "MSPProcessor",
    "MSPModel",
    "MSPForCTC",
    # Registration
    "registers",
]
