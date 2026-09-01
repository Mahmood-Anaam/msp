from transformers.processing_utils import ProcessorMixin


class MSPProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "video_processor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, feature_extractor=None, video_processor=None, tokenizer=None, **kwargs
    ):
        super().__init__(feature_extractor, video_processor, tokenizer, **kwargs)
        self.feature_extractor = feature_extractor
        self.video_processor = video_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        audio=None,
        videos=None,
        text=None,
        **kwargs,
    ):
        if audio is None and videos is None and text is None:
            raise ValueError("Provide at least one of audio, videos, or text.")

        inputs = super().__call__(
            images=None, audio=audio, videos=videos, text=text, **kwargs
        )

        if "input_ids" in inputs:
            inputs["labels"] = inputs.pop("input_ids")

        if "attention_mask" in inputs:
            inputs.pop("attention_mask")

        return inputs

    @property
    def model_input_names(self) -> list[str]:
        return [
            "input_values",
            "padding_mask",
            "pixel_values_videos",
            "padding_mask_videos",
            "labels",
        ]
