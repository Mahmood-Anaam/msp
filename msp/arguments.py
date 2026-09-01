import os
import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from transformers import TrainingArguments as HfTrainingArguments


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class TrainingArguments(HfTrainingArguments):
    pass


@dataclass
class ModelArguments:
    modality: str = field(
        default="msp_visual",
        metadata={
            "help": "The modality of the model. Options: 'msp_audio', 'msp_visual', 'msp'.",
            "choices": ["msp_audio", "msp_visual", "msp"],
        },
    )

    model_type: str = field(
        default="msp_visual",
        metadata={
            "help": "The type of model. Options: 'msp_audio', 'msp_visual', 'msp'.",
            "choices": ["msp_audio", "msp_visual", "msp"],
        },
    )

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    processor_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to pretrained processor. "
                "If not specified, uses model_name_or_path."
            )
        },
    )

    tokenizer_name_or_path: str | None = field(
        default="facebook/wav2vec2-base-960h",
        metadata={
            "help": "Path to pretrained tokenizer. If not specified, uses model_name_or_path."
        },
    )

    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )

    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the base model layers of the model."},
    )

    freeze_audio_branch: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the audio branch of the model."},
    )

    freeze_visual_branch: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the visual branch of the model."},
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code when loading datasets."},
    )

    ctc_loss_audio_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for the audio branch in the CTC loss."},
    )

    ctc_loss_visual_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for the visual branch in the CTC loss."},
    )

    ctc_loss_msp_weight: float = field(
        default=0.6,
        metadata={"help": "Weight for the MSP branch in the CTC loss."},
    )


@dataclass
class DataArguments:
    streaming_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming dataset."},
    )
    include_mcorec: bool = field(
        default=False,
        metadata={"help": "Whether to include mcorec dataset."},
    )

    data_cache_dir: str | None = field(
        default="/workspace/data/cache",
        metadata={"help": "Path to cache directory for datasets."},
    )


def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parse command-line arguments into (ModelArguments, DataArguments, TrainingArguments)."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


if __name__ == "__main__":
    model_args, data_args, training_args = parse_args()
    print(model_args)
    print(data_args)
    print(training_args)
