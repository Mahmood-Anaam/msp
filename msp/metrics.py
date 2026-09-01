import evaluate
import numpy as np


def _metrics(tokenizer):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_text = tokenizer.batch_decode(pred_ids)
        label_text = tokenizer.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_text, references=label_text)}

    return compute_metrics
