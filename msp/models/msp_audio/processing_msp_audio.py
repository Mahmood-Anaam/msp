from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import (
    AudioInput,
    PreTokenizedInput,
    TextInput,
)


class MSPAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class MSPAudioProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: AudioInput | None = None,
        text: str | list[str] | TextInput | PreTokenizedInput | None = None,
        **kwargs: Unpack[MSPAudioProcessorKwargs],
    ):
        if audio is None and text is None:
            raise ValueError("Provide at least one of audio or text.")

        output_kwargs = self._merge_kwargs(
            MSPAudioProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        inputs = None
        encodings = None

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        if audio is None:
            return encodings

        inputs["labels"] = encodings["input_ids"]
        return inputs

    def pad(self, *args, **kwargs):
        """
        Pad a batch of features and/or labels.

        Forwards audio batches to feature_extractor.pad and label batches
        to tokenizer.pad.
        """
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if args:
            input_features = args[0]
            args = args[1:]

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        if input_features is None:
            return labels

        input_features["labels"] = labels["input_ids"]
        return input_features

    @property
    def model_input_names(self) -> list[str]:
        return self.feature_extractor.model_input_names + ["labels"]
