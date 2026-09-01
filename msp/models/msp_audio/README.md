---
library_name: transformers
license: apache-2.0
language:
- en
base_model: facebook/wav2vec2-large-robust-ft-libri-960h
datasets:
- nguyenvulebinh/AVYT
- nguyenvulebinh/AVCocktail
metrics:
- wer
tags:
- custom_code
- msp
- msp_audio
- audio
- ctc
- speech-recognition
- ASR
---

# MSP-ASR

MSP-ASR is the audio model in [Multimodal Speech Perception (MSP)](https://github.com/Mahmood-Anaam/msp). It fine-tunes a Wav2Vec2 encoder with a CTC head for English speech recognition and supports greedy decoding or the companion 3-gram language model.

| Property | Value |
|---|---|
| Input | Mono, 16 kHz audio |
| Architecture | Wav2Vec2 encoder + CTC head |
| Base checkpoint | `facebook/wav2vec2-large-robust-ft-libri-960h` |
| Demo | [MSP-ASR](https://huggingface.co/spaces/MahmoodAnaam/MSP-ASR) |

MSP-ASR also supplies the audio encoder to MSP-AVSR. There, its representations participate in both directions of the bidirectional cross-attention fusion block; the standalone checkpoint remains audio-only.

## Usage

```python
import torch
from transformers import AutoModelForCTC, AutoProcessor

model_id = "MahmoodAnaam/MSP-ASR"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True).eval()

inputs = processor(audio="sample.wav", return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits

text = processor.tokenizer.batch_decode(logits.argmax(dim=-1))[0]
print(text)
```

For beam search, install `pyctcdecode` and `kenlm`, then load `MahmoodAnaam/MSP-Processor-With-LM` and call `batch_decode(logits.cpu().numpy())`. Pin Hub revisions when loading custom code in controlled environments.

## Training

| Setting | Value |
|---|---:|
| Maximum steps | 40,000 |
| Learning rate | 1e-4 |
| Train / eval batch size per device | 16 / 16 |
| Gradient accumulation | 2 |
| Scheduler | Cosine; 1,000 warmup steps |
| Precision | bfloat16 |

The selected validation checkpoint recorded loss **0.3481** and WER **20.40%**. The recorded stack was Transformers 5.10.2, PyTorch 2.10.0 with ROCm 7.2.4, Datasets 4.0.0, and Tokenizers 0.22.2.

## Evaluation

The following tables report **MSP-ASR only**. Results use `MSP-Processor-With-LM`; WER is a percentage and lower is better.

### LRS2

| Evaluation | WER |
|---|---:|
| Clean test | 6.30 |
| Clean + eight noisy conditions, macro average | 46.34 |

| Interferer | -5 dB | 0 dB | 5 dB | 10 dB |
|---:|---:|---:|---:|---:|
| 1 | 90.22 | 42.76 | 47.40 | 22.52 |
| 2 | 92.99 | 42.86 | 48.91 | 23.09 |

### AVCocktail

| Segmentation | WER |
|---|---:|
| Active-speaker-detection chunks | 77.62 |
| Fixed 10-second chunks | 127.36 |
| Gold chunks | 74.37 |


## Resources

- [Source and training configuration](https://github.com/Mahmood-Anaam/msp)
- [MSP-ASR demo](https://huggingface.co/spaces/MahmoodAnaam/MSP-ASR)
- [3-gram LM processor](https://huggingface.co/MahmoodAnaam/MSP-Processor-With-LM)
