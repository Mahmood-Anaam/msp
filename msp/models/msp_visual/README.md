---
library_name: transformers
license: apache-2.0
language:
- en
base_model: MahmoodAnaam/avhubert_encoder_large_noise_pt_noise_ft_433h
datasets:
- nguyenvulebinh/AVYT
- nguyenvulebinh/AVCocktail
metrics:
- wer
tags:
- custom_code
- msp_visual
- visual-speech-recognition
- lip-reading
- ctc
- VSR
---

# MSP-VSR

MSP-VSR is the visual speech-recognition model in [Multimodal Speech Perception (MSP)](https://github.com/Mahmood-Anaam/msp). It uses an AV-HuBERT encoder and a CTC head to transcribe English speech from silent mouth-region video.

| Property | Value |
|---|---|
| Input | Visible-speaker video |
| Architecture | AV-HuBERT visual encoder + CTC head |
| Base checkpoint | `MahmoodAnaam/avhubert_encoder_large_noise_pt_noise_ft_433h` |
| Demo | [MSP-VSR](https://huggingface.co/spaces/MahmoodAnaam/MSP-VSR) |

The processor converts frames to grayscale, rescales and normalizes them, resizes to 96 pixels, and applies an 88-pixel crop. MSP-VSR also supplies the visual encoder to MSP-AVSR, where its representations participate in both directions of the bidirectional cross-attention fusion block.

## Usage

```python
import torch
from transformers import AutoModelForCTC, AutoProcessor

model_id = "MahmoodAnaam/MSP-VSR"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True).eval()

inputs = processor(videos="sample.mp4", return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits

text = processor.tokenizer.batch_decode(logits.argmax(dim=-1))[0]
print(text)
```

Use footage with a visible, trackable speaking face. For 3-gram beam search, install `pyctcdecode` and `kenlm`, then decode with `MahmoodAnaam/MSP-Processor-With-LM`. Pin Hub revisions when loading custom code in controlled environments.

## Training

| Setting | Value |
|---|---:|
| Maximum steps | 50,000 |
| Learning rate | 1e-4 |
| Train / eval batch size per device | 32 / 32 |
| Scheduler | Cosine; 1,000 warmup steps |
| Precision | bfloat16 |

The selected validation checkpoint recorded loss **1.2109** and WER **61.98%**. The recorded stack was Transformers 5.10.2, PyTorch 2.10.0 with ROCm 7.2.4, Datasets 4.0.0, and Tokenizers 0.22.2.

## Evaluation

The following tables report **MSP-VSR only**. Results use `MSP-Processor-With-LM`; WER is a percentage and lower is better.

### LRS2

| Evaluation | WER |
|---|---:|
| Clean test | 27.37 |

### AVCocktail

| Segmentation | WER |
|---|---:|
| Active-speaker-detection chunks | 58.09 |
| Fixed 10-second chunks | 74.77 |
| Gold chunks | 58.06 |

## Resources

- [Source and training configuration](https://github.com/Mahmood-Anaam/msp)
- [MSP-VSR demo](https://huggingface.co/spaces/MahmoodAnaam/MSP-VSR)
- [3-gram LM processor](https://huggingface.co/MahmoodAnaam/MSP-Processor-With-LM)
