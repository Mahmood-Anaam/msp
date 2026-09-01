# Multimodal Speech Perception (MSP)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](pyproject.toml)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow.svg)](https://huggingface.co/MahmoodAnaam)

MSP is a Transformers-compatible speech-recognition stack for audio (ASR), video (VSR), and synchronized audio-video (AVSR). It combines Wav2Vec2 and AV-HuBERT encoders with bidirectional cross-attention, CTC training, and optional 3-gram language-model decoding.

## Model family

| Model | Modality | Checkpoint | Demo |
|---|---|---|---|
| MSP-ASR | Audio |  [Model](https://huggingface.co/MahmoodAnaam/MSP-ASR) | [MSP-ASR](https://huggingface.co/spaces/MahmoodAnaam/MSP-Audio) |
| MSP-VSR | Visual  |  [Model](https://huggingface.co/MahmoodAnaam/MSP-VSR) | [MSP-VSR](https://huggingface.co/spaces/MahmoodAnaam/MSP-Visual) |
| MSP-AVSR | Audio + Visual | [Model](https://huggingface.co/MahmoodAnaam/MSP-AVSR)  | [MSP-AVSR](https://huggingface.co/spaces/MahmoodAnaam/MSP-AVSR) |
| MSP Processor with LM | CTC logits | [Processor](https://huggingface.co/MahmoodAnaam/MSP-Processor-With-LM) | — |

## Architecture

1. **Audio encoder:** MSP-ASR converts normalized 16 kHz waveforms into contextual Wav2Vec2 representations.
2. **Visual encoder:** MSP-VSR converts video to normalized grayscale mouth-region frames and encodes them with AV-HuBERT.
3. **Shared space:** both sequences are independently projected and layer-normalized to the 768-dimensional fusion width.
4. **Bidirectional cross-attention:** one 12-head module uses audio queries with visual keys/values; a second uses visual queries with audio keys/values. Both apply padding masks, residual connections, and layer normalization.
5. **Temporal fusion:** the video-query sequence is upsampled to the audio time resolution, concatenated with the audio-query sequence, projected, and processed by a 3,072-dimensional GELU feed-forward layer.
6. **CTC decoding:** a shared head predicts token logits. Training adds auxiliary audio and visual CTC losses to the fused objective with published weights `0.5 / 0.5 / 1.0`.

MSP-AVSR can also route audio-only or video-only inputs through its pretrained branches during evaluation.

## Benchmarks

These tables reproduce the checked-in LaTeX results under `docs/tables/`. Scores use `MSP-Processor-With-LM` beam search. WER is a percentage; lower is better. `—` means not reported, and ★ marks MSP models.

### LRS2: clean and aggregate

| ID | Model | Modality | Training data | Clean WER | Average WER |
|---|---|---|---|---:|---:|
| A1 | MSP-ASR ★ | Audio | LRS2, Vox2, AVYT | 6.30 | 46.34 |
| V1 | MSP-VSR ★ | Visual | LRS2, Vox2, AVYT | 27.37 | — |
| AV1 | MSP-AVSR ★ | Audio + visual | LRS2, Vox2, AVYT | 3.93 | 22.80 |
| AV2 | Auto-AVSR | Audio + visual | LRS2, Vox2, LRS3, AVSpeech | 1.70 | 21.70 |
| AV3 | Whisper-Flamingo | Audio + visual | LRS3, Vox2 | 6.10 | 40.10 |

The average covers the clean split and eight noisy conditions where complete results are available.

### LRS2: acoustic interference

| ID | Model | Interferer | -5 dB | 0 dB | 5 dB | 10 dB |
|---|---|---:|---:|---:|---:|---:|
| A1 | MSP-ASR ★ | 1 | 90.22 | 42.76 | 47.40 | 22.52 |
| A1 | MSP-ASR ★ | 2 | 92.99 | 42.86 | 48.91 | 23.09 |
| AV1 | MSP-AVSR ★ | 1 | 48.85 | 20.90 | 22.41 | 12.33 |
| AV1 | MSP-AVSR ★ | 2 | 40.92 | 19.91 | 22.86 | 13.08 |
| AV2 | Auto-AVSR | 1 | 56.60 | 16.60 | 10.30 | 4.20 |
| AV2 | Auto-AVSR | 2 | 69.60 | 21.80 | 11.70 | 3.50 |
| AV3 | Whisper-Flamingo | 1 | 96.90 | 37.40 | 26.20 | 12.10 |
| AV3 | Whisper-Flamingo | 2 | 99.60 | 38.60 | 30.60 | 13.40 |

### AVCocktail

AVCocktail values are word-weighted across 51 videos.

| ID | Model | ASD chunks | Fixed 10 s | Gold chunks |
|---|---|---:|---:|---:|
| A1 | MSP-ASR ★ | 77.62 | 127.36 | 74.37 |
| V1 | MSP-VSR ★ | 58.09 | 74.77 | 58.06 |
| AV1 | MSP-AVSR ★ | 67.73 | 107.46 | 63.64 |
| AV2 | Auto-AVSR | 74.60 | 133.20 | 67.80 |
| AV3 | Whisper-Flamingo | 70.80 | 133.30 | 58.30 |

WER may exceed 100% when insertions are numerous. Results depend on the checkpoint, preprocessing, segmentation, dataset revision, and decoder configuration.

## Quick start

Requires Python 3.12+, a compatible PyTorch build, and media support for TorchCodec.

```bash
git clone https://github.com/Mahmood-Anaam/msp.git
cd msp
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

```python
import torch
from transformers import AutoModelForCTC, AutoProcessor

model_id = "MahmoodAnaam/MSP-AVSR"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True).eval()

inputs = processor(audio="sample.mp4", videos="sample.mp4", return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits

text = processor.tokenizer.batch_decode(logits.argmax(dim=-1))[0]
print(text)
```

Audio and video must cover the same time interval. Use MSP-ASR with `processor(audio=...)` or MSP-VSR with `processor(videos=...)` for unimodal inference. Pin Hub revisions when loading custom code in controlled environments.

For 3-gram beam search, install `pyctcdecode` and `kenlm`, then decode the logits with the companion processor:

```python
lm_processor = AutoProcessor.from_pretrained(
    "MahmoodAnaam/MSP-Processor-With-LM",
    trust_remote_code=True,
)
text = lm_processor.batch_decode(logits.cpu().numpy()).text[0]
```

## Train and evaluate

```bash
train configs/train_msp_asr.json
train configs/train_msp_vsr.json
train configs/train_msp_avsr.json

evaluate \
  --model_type msp \
  --model_name_or_path MahmoodAnaam/MSP-AVSR \
  --processor_name_or_path MahmoodAnaam/MSP-Processor-With-LM \
  --dataset_name lrs2 \
  --streaming_dataset true \
  --set_id test \
  --device cuda
```

Set `HF_TOKEN` for Hub uploads and the relevant `WANDB_*` variables for Weights & Biases. See the [AMD MI300X guide](docs/amd_droplet.md) for the complete LRS2 and AVCocktail evaluation matrix. The workflows reference [AVYT](https://huggingface.co/datasets/nguyenvulebinh/AVYT) and [AVCocktail](https://huggingface.co/datasets/nguyenvulebinh/AVCocktail); review their terms before use.

## Citation

```text
Mahmood Anaam. Multimodal Speech Perception (MSP).
https://github.com/Mahmood-Anaam/msp
```
