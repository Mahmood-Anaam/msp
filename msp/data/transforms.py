import io
import random
import re

import av
import cv2
import numpy as np
import torch
import torchaudio
import torchvision
from decord import VideoReader
from transformers import AutoTokenizer


def load_audio(path, start_time=0, end_time=None):
    from torchcodec.decoders import AudioDecoder

    if AudioDecoder is not None:
        audio_decoder = AudioDecoder(path)
        if end_time is None:
            end_time = audio_decoder.metadata.duration_seconds_from_header
        waveform = audio_decoder.get_samples_played_in_range(start_time, end_time).data
    else:
        if start_time == 0 and end_time is None:
            frame_offset = 0
            num_frames = -1
        else:
            frame_offset = int(start_time * 16000)
            num_frames = int((end_time - start_time) * 16000)
        waveform, sample_rate = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames, normalize=True
        )
        assert sample_rate == 16000
    return waveform.transpose(1, 0)  # T x 1


def load_video(path, start_time=0, end_time=None):
    """
    Loads a video file and returns it as a tensor.
    rtype: torch, T x C x H x W
    """
    from torchcodec.decoders import VideoDecoder

    video_decoder = VideoDecoder(path, dimension_order="NHWC")
    if end_time is None:
        end_time = video_decoder.metadata.duration_seconds
    vid_rgb = video_decoder.get_frames_played_in_range(start_time, end_time).data
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in vid_rgb.numpy()]
    vid = torch.from_numpy(np.stack(frames)).unsqueeze(1)
    return vid


def load_video_with_av(path: str | bytes, start_time=0, end_time=None):
    """
    Loads a video file and returns it as a tensor.
    rtype: torch, T x C x H x W
    """
    file_object = io.BytesIO(path)
    vr = VideoReader(file_object)
    frames_rgb = vr.get_batch(range(len(vr)))
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames_rgb.asnumpy()]
    video = torch.from_numpy(np.stack(frames)).unsqueeze(1)
    return video  # T x 1 x H x W


def load_audio_with_av(path: str | bytes, start_time=0, end_time=None):
    """
    Loads audio without requiring an external ffmpeg installation.
    """
    # 1. Open the source (file path string or in-memory bytes)
    if isinstance(path, bytes):
        file_obj = io.BytesIO(path)
        container = av.open(file_obj)
    else:
        container = av.open(path)

    # 2. Select the audio stream and configure the resampler to 16000Hz mono
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

    audio_frames = []

    # 3. Decode chunks and extract timeline segments
    for frame in container.decode(stream):
        # Calculate current timestamp in seconds
        frame_time = frame.pts * stream.time_base if frame.pts is not None else 0

        if end_time is not None and frame_time > end_time:
            break
        if frame_time < start_time:
            continue

        # Resample frame to 16kHz Mono Float Planar
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            # Convert planar float data directly to numpy
            audio_frames.append(rf.to_ndarray().flatten())

    container.close()

    # 4. Handle edge-case for empty data
    if not audio_frames:
        return torch.zeros((0, 1), dtype=torch.float32)

    # 5. Stack chunks, convert to torch, and shape to T x 1
    audio_np = np.concatenate(audio_frames)
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(1)

    return audio_tensor


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AddMultiSpk(torch.nn.Module):
    def __init__(
        self,
        speech_dataset=None,
        snr_target=None,
        interferer_spk=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20]
        self.interferer_spk = [interferer_spk] if interferer_spk else [0, 0, 1, 2]
        self.speech_dataset = speech_dataset

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.speech_dataset is None:
            return speech
        speech_length = speech.size(0) / 16000
        if speech_length < 2:
            return speech

        num_interferer = random.choice(self.interferer_spk)
        interferer_signal = None
        for _ in range(num_interferer):
            interferer = load_audio_with_av(random.choice(self.speech_dataset)["video"])
            interferer_length = interferer.size(0) / 16000
            # print(interferer, interferer_length)
            if 2 <= interferer_length <= 10:
                interferer = cut_or_pad(interferer, len(speech))
                if interferer_signal is None:
                    interferer_signal = interferer
                else:
                    snr_level = torch.tensor([random.choice([-5, 0, 5, 10, 15])])
                    interferer_signal = torchaudio.functional.add_noise(
                        interferer_signal.t(), interferer.t(), snr_level
                    ).t()

        if interferer_signal is None:
            return speech
        # print(f"Adding {num_interferer} interferer(s) to speech with length {speech_length:.2f}s")
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        speech = torchaudio.functional.add_noise(
            speech.t(), interferer_signal.t(), snr_level
        ).t()

        return speech


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=None,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        if noise_filename is None:
            self.noise = None
        else:
            self.noise, sample_rate = torchaudio.load(noise_filename)
            assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise is None:
            return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


class VideoTransform:
    def __init__(self, subset):
        self.subset = subset
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, speech_dataset=None, snr_target=None):
        self.subset = subset
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddMultiSpk(speech_dataset=speech_dataset),
                AddNoise(),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)


TOKENIZER = AutoTokenizer.from_pretrained(
    "facebook/wav2vec2-large-robust-ft-libri-960h"
)


class TextTransform:
    """Normalize CTC targets and remove AVYT non-speech markers.

    ``<unk>`` in AVYT/AVYT-mix marks a visible face with no target speech.  It is
    not a lexical token and must therefore be removed *before* punctuation is
    stripped.  Otherwise the old normalization converted it to the word
    ``UNK`` and trained the character CTC heads to emit U-N-K during silence.
    """

    def __init__(self, tokenizer=TOKENIZER):
        self.tokenizer = tokenizer
        if tokenizer is None:
            ValueError("Tokenizer must be provided for TextTransform.")

    def tokenize(self, text):
        text = self.norm_string(text)
        encoding = self.tokenizer(text)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        return input_ids

    def post_process(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        return text

    def norm_string(self, text):
        text = re.sub(r"(?i)<unk>", " ", text)
        text = re.sub(r"[^a-zA-Z' ]", "", text)
        return re.sub(r"\s+", " ", text).strip().upper()

    def __call__(self, sample):

        return self.norm_string(sample)
