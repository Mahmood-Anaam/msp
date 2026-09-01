---
library_name: transformers
license: apache-2.0
language:
- en
pipeline_tag: automatic-speech-recognition
datasets:
- MahmoodAnaam/LRS2-Text
tags:
- processor
- custom_code
- msp
- beam-search
- language-model
- 3-gram
- pyctcdecode
- kenlm
---

# MSP Processor with KenLM 3-gram Decoder

`MSP-Processor-With-LM` adds beam-search decoding to the MSP model family. It combines the MSP tokenizer, `pyctcdecode`, and an English KenLM 3-gram language model; it does not generate logits and is not a standalone speech-recognition model.

The language model was built exclusively from transcripts in the `train` and `pretrain` splits of [`MahmoodAnaam/LRS2-Text`](https://huggingface.co/datasets/MahmoodAnaam/LRS2-Text). Validation and test transcripts were not used to build it.

The processor is compatible with CTC logits from:

- [`MahmoodAnaam/MSP-ASR`](https://huggingface.co/MahmoodAnaam/MSP-ASR)
- [`MahmoodAnaam/MSP-VSR`](https://huggingface.co/MahmoodAnaam/MSP-VSR)
- [`MahmoodAnaam/MSP-AVSR`](https://huggingface.co/MahmoodAnaam/MSP-AVSR)

## Usage

Install the decoder dependencies:

```bash
python -m pip install pyctcdecode
python -m pip install "https://github.com/kpu/kenlm/archive/master.zip"
```

### Decode existing logits

```python
from transformers import AutoProcessor

lm_processor = AutoProcessor.from_pretrained(
    "MahmoodAnaam/MSP-Processor-With-LM",
    trust_remote_code=True,
)

# logits: NumPy array with shape (batch, time, vocabulary)
result = lm_processor.batch_decode(logits)
print(result.text[0])
```

### End-to-end MSP-AVSR example

```python
import torch
from transformers import AutoModelForCTC, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "MahmoodAnaam/MSP-AVSR"
model_processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)
lm_processor = AutoProcessor.from_pretrained(
    "MahmoodAnaam/MSP-Processor-With-LM",
    trust_remote_code=True,
)
model = AutoModelForCTC.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval().to(device)

inputs = model_processor(
    audio="sample.mp4",
    videos="sample.mp4",
    return_tensors="pt",
).to(device)

# OR
inputs = lm_processor(
    audio="sample.mp4",
    videos="sample.mp4",
    return_tensors="pt",
).to(device) 

with torch.inference_mode():
    logits = model(**inputs).logits

greedy_text = model_processor.tokenizer.batch_decode(
    logits.argmax(dim=-1)
)[0]
beam_text = lm_processor.batch_decode(
    logits.cpu().detach().numpy()
).text[0]

print("Greedy:", greedy_text)
print("Beam search:", beam_text)
```

Use the corresponding model processor to prepare audio or video inputs, and use this LM processor to decode the resulting logits. For MSP-ASR or MSP-VSR, change `model_id` and provide only the supported input modality.

`batch_decode` supports controls such as `beam_width`, `beam_prune_logp`, `token_min_logp`, `hotwords`, `hotword_weight`, `alpha`, `beta`, `n_best`, and `output_word_offsets`. Tune these values on a validation set for the target domain; increasing language-model weight does not guarantee a lower WER.


## Resources

- [MSP source repository](https://github.com/Mahmood-Anaam/msp)
- [LRS2 text dataset](https://huggingface.co/datasets/MahmoodAnaam/LRS2-Text)
- [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)
- [KenLM](https://github.com/kpu/kenlm)
