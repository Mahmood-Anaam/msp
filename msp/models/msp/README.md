---
library_name: transformers
license: apache-2.0
language:
- en
base_model:
- MahmoodAnaam/MSP-ASR
- MahmoodAnaam/MSP-VSR
datasets:
- nguyenvulebinh/AVYT
- nguyenvulebinh/AVCocktail
metrics:
- wer
tags:
- custom_code
- msp
- ctc
- audio-visual-speech-recognition
- AVSR
- VSR
- ASR
---

# MSP-AVSR

MSP-AVSR is the multimodal model in [Multimodal Speech Perception (MSP)](https://github.com/Mahmood-Anaam/msp). It combines Wav2Vec2 audio representations and AV-HuBERT visual representations through bidirectional cross-attention, then predicts English text with CTC.

| Property | Value |
|---|---|
| Inputs | Synchronized mono 16 kHz audio and visible-speaker video |
| Audio / visual encoders | MSP-ASR / MSP-VSR |
| Fusion | Mask-aware, 12-head bidirectional cross-attention |
| Fusion / feed-forward width | 768 / 3,072 |
| Demo | [MSP-AVSR](https://huggingface.co/spaces/MahmoodAnaam/MSP-AVSR) |

## Architecture

1. MSP-ASR and MSP-VSR encode audio and mouth-region video independently.
2. Separate projections and layer normalization map both sequences to a shared 768-dimensional space.
3. Two cross-attention paths model both directions: audio queries attend to visual keys/values, while visual queries attend to audio keys/values. Each path uses padding masks, a residual connection, and layer normalization.
4. The visual-query output is upsampled to the audio frame rate, concatenated with the audio-query output, projected, and processed by a 3,072-dimensional GELU feed-forward block.
5. A shared CTC head predicts the transcript. Training adds auxiliary audio and visual CTC losses to the fused loss with weights `0.5 / 0.5 / 1.0`.

In evaluation mode, the checkpoint can also process audio-only or video-only inputs through the corresponding pretrained branch.

## Usage

```python
import torch
from transformers import AutoModelForCTC, AutoProcessor

model_id = "MahmoodAnaam/MSP-AVSR"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True).eval()

inputs = processor(
    audio="sample.mp4",
    videos="sample.mp4",
    return_tensors="pt",
)
with torch.inference_mode():
    logits = model(**inputs).logits

text = processor.tokenizer.batch_decode(logits.argmax(dim=-1))[0]
print(text)
```

Audio and video must represent the same time interval. For 3-gram beam search, install `pyctcdecode` and `kenlm`, then decode with `MahmoodAnaam/MSP-Processor-With-LM`. Pin Hub revisions when loading custom code in controlled environments.

## Training

| Setting | Value |
|---|---:|
| Maximum steps | 30,000 |
| Learning rate | 1e-4 |
| Train / eval batch size per device | 32 / 32 |
| Scheduler | Cosine; 1,000 warmup steps |
| Precision | bfloat16 |
| Audio / visual / fused CTC weights | 0.5 / 0.5 / 1.0 |

The selected validation checkpoint recorded loss **1.1193** and WER **17.36%**. The recorded stack was Transformers 5.10.2, PyTorch 2.10.0 with ROCm 7.2.4, Datasets 4.0.0, and Tokenizers 0.22.2.

## Evaluation

The following tables report **MSP-AVSR only**. Results use `MSP-Processor-With-LM`; WER is a percentage and lower is better.

### LRS2

| Evaluation | WER |
|---|---:|
| Clean test | 3.93 |
| Clean + eight noisy conditions, macro average | 22.80 |

| Interferer | -5 dB | 0 dB | 5 dB | 10 dB |
|---:|---:|---:|---:|---:|
| 1 | 48.85 | 20.90 | 22.41 | 12.33 |
| 2 | 40.92 | 19.91 | 22.86 | 13.08 |

### AVCocktail

| Segmentation | WER |
|---|---:|
| Active-speaker-detection chunks | 67.73 |
| Fixed 10-second chunks | 107.46 |
| Gold chunks | 63.64 |

AVCocktail values are word-weighted across 51 videos. WER can exceed 100% when insertions are numerous. Scores depend on preprocessing, synchronization, face tracking, dataset revision, checkpoint, segmentation, and decoder settings.


## Resources

- [Source and training configuration](https://github.com/Mahmood-Anaam/msp)
- [MSP-AVSR demo](https://huggingface.co/spaces/MahmoodAnaam/MSP-AVSR)
- [3-gram LM processor](https://huggingface.co/MahmoodAnaam/MSP-Processor-With-LM)
