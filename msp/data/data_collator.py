from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch

from .transforms import (
    AudioTransform,
    TextTransform,
    VideoTransform,
    load_audio,
    load_audio_with_av,
    load_video,
    load_video_with_av,
)


def _pad_sequence(
    samples: list[torch.Tensor], pad_value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length tensors on the time axis."""
    lengths = torch.tensor([sample.size(0) for sample in samples], dtype=torch.long)
    max_length = int(lengths.max().item())
    shape = list(samples[0].shape[1:])
    batch = samples[0].new_full([len(samples), max_length] + shape, pad_value)
    for index, sample in enumerate(samples):
        batch[index, : sample.size(0)] = sample
    return batch, lengths


def _length_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create a valid-position mask from sequence lengths."""
    return torch.arange(int(lengths.max().item()))[None, :] < lengths[:, None]


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


@dataclass
class DataCollator:
    modality: Literal["msp_audio", "msp_visual", "msp"] = "msp"
    audio_transform: Optional[AudioTransform] = None
    video_transform: Optional[VideoTransform] = None
    text_transform: Optional[TextTransform] = None

    def _load_modalities(self, feature: Dict) -> tuple[torch.Tensor, torch.Tensor]:
        start_time = feature.get("start_time", 0)
        end_time = feature.get("end_time", None)
        source = feature.get("video")

        audio, video = None, None
        if self.modality in {"msp_audio", "msp"}:
            subset = self.audio_transform.subset if self.audio_transform else None
            if subset and subset in ["train", "val"]:
                audio_loader = load_audio_with_av
            else:
                audio_loader = load_audio

            audio = audio_loader(source, start_time, end_time)

        if self.modality in {"msp_visual", "msp"}:
            subset = self.video_transform.subset if self.video_transform else None
            if subset and subset in ["train", "val"]:
                video_loader = load_video_with_av
            else:
                video_loader = load_video

            video = video_loader(source, start_time, end_time)

        if audio is not None and self.audio_transform is not None:
            audio = self.audio_transform(audio).reshape((-1,))

        if video is not None and self.video_transform is not None:
            video = self.video_transform(video)

        return audio, video

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        audios, videos, labels = [], [], []

        for feature in features:
            audio, video = self._load_modalities(feature)
            if audio is not None:
                audios.append(audio)
            if video is not None:
                videos.append(video)

            label = feature.get("label")
            if label is not None:
                labels.append(self.text_transform.tokenize(label))

        batch: Dict[str, torch.Tensor] = {}
        if audios:
            input_values, audio_lengths = _pad_sequence(audios, 0.0)
            batch["input_values"] = input_values
            batch["padding_mask"] = _length_mask(audio_lengths).long()

            normalized_input_values = zero_mean_unit_var_norm(
                input_values.cpu().numpy(),
                batch["padding_mask"].cpu().numpy()
                if batch["padding_mask"] is not None
                else None,
                padding_value=0.0,
            )
            batch["input_values"] = torch.from_numpy(
                np.stack(normalized_input_values, axis=0).astype(np.float32, copy=False)
            )
        if videos:
            pixel_values, video_lengths = _pad_sequence(videos, 0.0)
            batch["pixel_values_videos"] = pixel_values.permute(
                0, 2, 1, 3, 4
            )  # [B, C, T, H, W]
            batch["padding_mask_videos"] = _length_mask(video_lengths).long()

        if labels:
            label_batch, _ = _pad_sequence(labels, -100)
            batch["labels"] = (
                label_batch.squeeze(-1) if label_batch.dim() == 3 else label_batch
            )

        return batch
