from transformers.processing_utils import ProcessorMixin


class MSPVisualProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, video_processor=None, tokenizer=None, **kwargs):
        super().__init__(video_processor, tokenizer, **kwargs)
        self.video_processor = video_processor
        self.tokenizer = tokenizer

    def __call__(self, videos=None, text=None, **kwargs):
        if videos is None and text is None:
            raise ValueError("Provide at least one of videos or text.")
        inputs = super().__call__(
            images=None, audio=None, videos=videos, text=text, **kwargs
        )

        if "input_ids" in inputs:
            inputs["labels"] = inputs.pop("input_ids")

        return inputs

    @property
    def model_input_names(self) -> list[str]:
        return ["pixel_values_videos", "padding_mask_videos", "labels"]
