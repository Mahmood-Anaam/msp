from pathlib import Path

import numpy as np
import torch
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import PaddingStrategy, TensorType, logging

logger = logging.get_logger(__name__)


class MSPAudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: list[np.ndarray],
        attention_mask: list[np.ndarray] | None,
        padding_value: float = 0.0,
    ) -> list[np.ndarray]:
        """Normalize each sequence to zero mean and unit variance."""
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, dtype=np.int32)
            normed = []
            for vec, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vec - vec[:length].mean()) / np.sqrt(
                    vec[:length].var() + 1e-7
                )
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value
                normed.append(normed_slice)
        else:
            normed = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]
        return normed

    def _load_audio(
        self,
        src: str | Path | bytes | torch.Tensor,
        start_seconds: float = 0.0,
        stop_seconds: float | None = None,
    ) -> np.ndarray:
        """Load audio waveform from file path or bytes as a 1-D numpy array."""
        from torchcodec.decoders import AudioDecoder

        audio_decoder = AudioDecoder(source=src, sample_rate=self.sampling_rate)
        if stop_seconds is None:
            stop_seconds = audio_decoder.metadata.duration_seconds_from_header
        waveform = audio_decoder.get_samples_played_in_range(
            start_seconds, stop_seconds
        ).data.numpy()
        return waveform.squeeze()  # shape: (T,)

    def __call__(
        self,
        raw_speech: (
            str
            | Path
            | bytes
            | np.ndarray
            | list[str]
            | list[Path]
            | list[bytes]
            | list[float]
            | list[np.ndarray]
            | list[list[float]]
        ),
        padding: bool | str | PaddingStrategy = False,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Featurize and pad one or several audio sequences.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Sampling rate mismatch: expected {self.sampling_rate}, "
                f"got {sampling_rate}."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1
        if is_batched_numpy and raw_speech.ndim > 2:
            raise ValueError("Only mono-channel audio is supported.")

        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple))
            and isinstance(raw_speech[0], (str, Path, bytes, np.ndarray, list, tuple))
        )

        if not is_batched:
            raw_speech = [raw_speech]

        # Load from file paths or bytes
        if isinstance(raw_speech[0], (str, Path, bytes)):
            raw_speech = [self._load_audio(src) for src in raw_speech]

        encoded = BatchFeature({"input_values": raw_speech})

        padded = self.pad(
            encoded,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # Ensure float32
        vals = padded["input_values"]
        if not isinstance(vals[0], np.ndarray):
            padded["input_values"] = [np.asarray(a, dtype=np.float32) for a in vals]
        elif isinstance(vals[0], np.ndarray) and vals[0].dtype == np.float64:
            padded["input_values"] = [a.astype(np.float32) for a in vals]

        # Normalize
        attn = padded.get("attention_mask")
        if attn is not None:
            padded["attention_mask"] = [np.asarray(a, dtype=np.int32) for a in attn]

        if self.do_normalize:
            norm_attn = (
                attn
                if self._get_padding_strategies(padding, max_length=max_length)
                is not PaddingStrategy.DO_NOT_PAD
                else None
            )
            padded["input_values"] = self.zero_mean_unit_var_norm(
                padded["input_values"],
                attention_mask=norm_attn,
                padding_value=self.padding_value,
            )

        # Rename attention_mask -> padding_mask
        if "attention_mask" in padded:
            padded["padding_mask"] = padded.pop("attention_mask")

        if return_tensors is not None:
            padded = padded.convert_to_tensors(return_tensors)

        return padded
