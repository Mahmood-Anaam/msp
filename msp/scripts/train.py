import os

import torch
from torchsummary import summary
from transformers import AutoModel, AutoModelForCTC, AutoProcessor, AutoTokenizer

from msp.arguments import parse_args
from msp.data import (
    AudioTransform,
    DataCollator,
    TextTransform,
    VideoTransform,
    load_avsr_dataset,
)
from msp.metrics import _metrics
from msp.models import (
    MSPAudioConfig,
    MSPAudioFeatureExtractor,
    MSPAudioForCTC,
    MSPAudioProcessor,
    MSPConfig,
    MSPForCTC,
    MSPProcessor,
    MSPVisualConfig,
    MSPVisualForCTC,
    MSPVisualProcessor,
    MSPVisualVideoProcessor,
    registers,
)
from msp.trainer import MSPTrainer

use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def main():
    registers()

    model_args, data_args, training_args = parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    # Load from pretrained checkpoint
    if model_args.model_name_or_path is not None:
        print("Loading pretrained model from", model_args.model_name_or_path)

        if model_args.model_type == "msp_audio":
            model = MSPAudioForCTC.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
            processor = MSPAudioProcessor.from_pretrained(
                model_args.processor_name_or_path or model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
        elif model_args.model_type == "msp_visual":
            model = MSPVisualForCTC.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
            processor = MSPVisualProcessor.from_pretrained(
                model_args.processor_name_or_path or model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
        elif model_args.model_type == "msp":
            model = MSPForCTC.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
            processor = MSPProcessor.from_pretrained(
                model_args.processor_name_or_path or model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
        else:
            raise ValueError(
                "Please specify a valid model type using --model_type. Valid options are: msp_audio, msp_visual, msp"
            )
    else:
        print("Training from scratch")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path, trust_remote_code=True
        )
        if model_args.model_type == "msp_audio":
            pretrained_name_or_path = "facebook/wav2vec2-large-robust-ft-libri-960h"

            pretrained_model = AutoModelForCTC.from_pretrained(
                pretrained_name_or_path,
            )
            pretrained_config = pretrained_model.config
            pretrained_processor = AutoProcessor.from_pretrained(
                pretrained_name_or_path
            )

            config = MSPAudioConfig(**pretrained_config.to_dict())
            config.update(
                {
                    "vocab_size": tokenizer.vocab_size,
                    "final_dropout": 0.1,
                    "ctc_loss_reduction": "mean",
                    "ctc_zero_infinity": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            )
            model = MSPAudioForCTC(config=config)
            processor = MSPAudioProcessor(
                feature_extractor=MSPAudioFeatureExtractor(
                    feature_size=pretrained_processor.feature_extractor.feature_size,
                    sampling_rate=pretrained_processor.feature_extractor.sampling_rate,
                    padding_value=pretrained_processor.feature_extractor.padding_value,
                    return_attention_mask=pretrained_processor.feature_extractor.return_attention_mask,
                    do_normalize=pretrained_processor.feature_extractor.do_normalize,
                ),
                tokenizer=tokenizer,
            )

            model.msp_audio.load_state_dict(pretrained_model.wav2vec2.state_dict())
            model.dropout.load_state_dict(pretrained_model.dropout.state_dict())
            model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

        elif model_args.model_type == "msp_visual":
            pretrained_name_or_path = f"{os.environ['HF_USERNAME']}/avhubert_encoder_large_noise_pt_noise_ft_433h"

            pretrained_model = AutoModel.from_pretrained(
                pretrained_name_or_path,
                trust_remote_code=True,
            )

            pretrained_config = pretrained_model.config
            config = MSPVisualConfig(
                visual_config=pretrained_config.to_dict(),
                vocab_size=tokenizer.vocab_size,
                final_dropout=0.1,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            model = MSPVisualForCTC(config=config)
            processor = MSPVisualProcessor(
                video_processor=MSPVisualVideoProcessor(), tokenizer=tokenizer
            )

            model.msp_visual.load_state_dict(pretrained_model.state_dict())

        elif model_args.model_type == "msp":
            hf_username = os.getenv("HF_USERNAME", "MahmoodAnaam")
            pretrained_audio_name_or_path = f"{hf_username}/MSP-ASR"
            pretrained_visual_name_or_path = f"{hf_username}/MSP-VSR"

            pretrained_audio_model = MSPAudioForCTC.from_pretrained(
                pretrained_audio_name_or_path,
                trust_remote_code=True,
            )
            pretrained_visual_model = MSPVisualForCTC.from_pretrained(
                pretrained_visual_name_or_path,
                trust_remote_code=True,
            )

            pretrained_audio_processor = MSPAudioProcessor.from_pretrained(
                pretrained_audio_name_or_path,
                trust_remote_code=True,
            )
            pretrained_visual_processor = MSPVisualProcessor.from_pretrained(
                pretrained_visual_name_or_path,
                trust_remote_code=True,
            )

            config = MSPConfig(
                audio_config=pretrained_audio_model.config.to_dict(),
                visual_config=pretrained_visual_model.config.to_dict(),
                vocab_size=tokenizer.vocab_size,
                ctc_loss_audio_weight=model_args.ctc_loss_audio_weight,
                ctc_loss_visual_weight=model_args.ctc_loss_visual_weight,
                ctc_loss_msp_weight=model_args.ctc_loss_msp_weight,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            model = MSPForCTC(config=config)
            processor = MSPProcessor(
                feature_extractor=pretrained_audio_processor.feature_extractor,
                video_processor=pretrained_visual_processor.video_processor,
                tokenizer=tokenizer,
            )

            model.msp.audio_model.load_state_dict(pretrained_audio_model.state_dict())
            model.msp.visual_model.load_state_dict(pretrained_visual_model.state_dict())

        else:
            raise ValueError(
                "Please specify a valid model type using --model_type. Valid options are: msp_audio, msp_visual, msp"
            )

    if model_args.freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
    if model_args.freeze_base_model:
        if hasattr(model, "freeze_base_model"):
            model.freeze_base_model()
    if model_args.freeze_audio_branch:
        if hasattr(model, "freeze_audio_branch"):
            model.freeze_audio_branch()
    if model_args.freeze_visual_branch:
        if hasattr(model, "freeze_visual_branch"):
            model.freeze_visual_branch()

    # Load dataset
    train_dataset, valid_dataset, interference_dataset = load_avsr_dataset(
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming_dataset,
        include_mcorec=data_args.include_mcorec,
    )

    train_av_data_collator = DataCollator(
        text_transform=TextTransform(tokenizer=processor.tokenizer),
        audio_transform=AudioTransform(
            subset="train", speech_dataset=interference_dataset
        ),
        video_transform=VideoTransform(subset="train"),
        modality=model_args.modality,
    )

    valid_av_data_collator = DataCollator(
        text_transform=TextTransform(tokenizer=processor.tokenizer),
        audio_transform=AudioTransform(subset="val"),
        video_transform=VideoTransform(subset="val"),
        modality=model_args.modality,
    )

    print("train_dataset\n", train_dataset)
    print("valid_dataset\n", valid_dataset)
    summary(model)

    training_args.bf16 = use_bf16
    training_args.fp16 = not use_bf16 and torch.cuda.is_available()

    print("Training arguments:\n", training_args)

    trainer = MSPTrainer(
        model=model,
        data_collator=train_av_data_collator,
        valid_data_collator=valid_av_data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=_metrics(processor.tokenizer),
        processing_class=processor,
    )

    if not training_args.resume_from_checkpoint:
        trainer.train()
    else:
        print("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    print("Evaluating the model on the validation dataset...")
    trainer.evaluate()

    print("Pushing the model to the HuggingFace Hub...")
    trainer.push_to_hub(commit_message="End of training")


if __name__ == "__main__":
    main()
