import argparse
import math
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import datasets
import torch
import torchaudio
import webvtt
from jiwer import wer
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
)

from msp.data import AudioTransform, DataCollator, TextTransform, VideoTransform

ModelType = Literal["msp_audio", "msp_visual", "msp"]
DatasetName = Literal["lrs2", "AVCocktail"]


LRS2_EVAL_SPLITS = [
    "test",
    "test_snr_n5_interferer_1",
    "test_snr_n5_interferer_2",
    "test_snr_0_interferer_1",
    "test_snr_0_interferer_2",
    "test_snr_5_interferer_1",
    "test_snr_5_interferer_2",
    "test_snr_10_interferer_1",
    "test_snr_10_interferer_2",
]


AVCOCKTAIL_VIDEO_IDS = [f"video_{i}" for i in range(0, 51)]
AVCOCKTAIL_CHUNK_TYPES = ["asd_chunk", "fixed_chunk", "gold_chunk"]

text_transform = TextTransform()


def _decode_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def normalize_text_for_wer(text: str) -> str:
    """
    Normalize text before WER calculation.

    This uses the project's TextTransform normalization to keep evaluation
    deterministic. If you have the exact AVSRCocktail `norm_string`, replace
    this function with that implementation for strict paper-level parity.
    """
    return text_transform.norm_string(text)


@dataclass
class MSPInferenceEngine:
    model_type: ModelType
    model_name_or_path: str
    processor_name_or_path: str | None = None
    cache_dir: str = "/workspace/cache/huggingface"
    device: str = "cuda"
    modality: ModelType = None

    def __post_init__(self) -> None:

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.model = None
        self.processor = None
        self.data_collator = None

    def load_model(self) -> None:
        print(f"Loading MSP model: {self.model_name_or_path}")
        print(f"Model type: {self.model_type}")

        processor_path = self.processor_name_or_path or self.model_name_or_path
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        self.model.eval().to(self.device)

        self.data_collator = DataCollator(
            text_transform=text_transform,
            audio_transform=AudioTransform(subset="test"),
            video_transform=VideoTransform(subset="test"),
            modality=self.model_type if self.modality is None else self.modality,
        )

        print("Model loaded successfully.")

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key != "labels"
        }

    def _decode_logits(self, logits: torch.Tensor) -> str:
        if hasattr(self.processor, "decoder"):
            logits = logits.cpu().detach().numpy()
            text = self.processor.batch_decode(logits).text[0]
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            if pred_ids.ndim == 1:
                pred_ids = pred_ids.unsqueeze(0)
            text = self.processor.tokenizer.batch_decode(pred_ids)[0]

        return text

    @torch.inference_mode()
    def infer_processed_sample(self, video: Any) -> str:
        """
        Run inference on one already-loaded dataset sample.

        `video` can be bytes, path, tensor-compatible source, or file-like object
        supported by torchcodec decoders.
        """
        sample = {"video": video}
        batch = self.data_collator([sample])
        model_inputs = self._move_batch_to_device(batch)
        outputs = self.model(**model_inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        return self._decode_logits(logits)

    def chunk_video(
        self,
        video_path: str,
        max_length: int = 15,
    ) -> List[tuple[float, float]]:
        """
        Split a video path into fixed chunks.

        This is kept for future file-based inference. Dataset evaluation below
        uses the same processed/chunked samples as AVSRCocktail.
        """
        audio, rate = torchaudio.load(video_path)
        duration = audio.shape[1] / rate

        num_chunks = math.ceil(duration / max_length)
        chunk_size = math.ceil(duration / num_chunks)

        segments: list[tuple[float, float]] = []
        steps = int(duration * 100)
        step_size = int(chunk_size * 100)

        for index in range(0, steps, step_size):
            start_time = index / 100
            end_time = min((index + step_size) / 100, duration)
            segments.append((start_time, end_time))

        return segments


def eval_lrs2(
    engine: MSPInferenceEngine,
    dataset: datasets.IterableDataset,
    max_samples: int | None = None,
) -> dict[str, float]:
    output_list: list[str] = []
    label_list: list[str] = []

    for index, sample in enumerate(tqdm(dataset, desc="Evaluating LRS2")):
        if max_samples is not None and index >= max_samples:
            break

        label = _decode_bytes(sample["label"])
        label_norm = normalize_text_for_wer(label)

        output = engine.infer_processed_sample(sample["video"])
        output_norm = normalize_text_for_wer(output)

        output_list.append(output_norm)
        label_list.append(label_norm)
    result = {
        "wer": wer(reference=label_list, hypothesis=output_list),
    }

    return result


def _parse_vtt_label(label_bytes: bytes) -> tuple[str, float, float]:
    label_list: list[str] = []
    label_start_times: list[float] = []

    global_start_time: float | None = None
    global_end_time: float | None = None

    with tempfile.NamedTemporaryFile(suffix=".vtt") as temp_file:
        with open(temp_file.name, "w") as file:
            file.write(_decode_bytes(label_bytes))

        for caption in webvtt.read(temp_file.name):
            if caption.text == "":
                continue

            start_time = (
                caption.start_time.hours * 3600
                + caption.start_time.minutes * 60
                + caption.start_time.seconds
                + caption.start_time.milliseconds / 1000
            )
            end_time = (
                caption.end_time.hours * 3600
                + caption.end_time.minutes * 60
                + caption.end_time.seconds
                + caption.end_time.milliseconds / 1000
            )

            label_list.append(caption.text)
            label_start_times.append(start_time)

            if global_start_time is None or start_time < global_start_time:
                global_start_time = start_time

            if global_end_time is None or end_time > global_end_time:
                global_end_time = end_time

    sorted_labels = [label for _, label in sorted(zip(label_start_times, label_list))]

    label_text = normalize_text_for_wer(" ".join(sorted_labels))

    if global_start_time is None or global_end_time is None:
        raise ValueError("Empty VTT label file.")

    return label_text, global_start_time, global_end_time


def eval_avcocktail(
    engine: MSPInferenceEngine,
    video_dataset: datasets.DatasetDict,
    label_dataset: datasets.Dataset,
    set_name: str | None = None,
    max_samples_per_chunk: int | None = None,
) -> tuple[dict[str, float], int]:
    wer_scores: dict[str, float] = {}

    label_text, label_start_time, label_end_time = _parse_vtt_label(
        label_dataset["label"][0]
    )
    num_words = len(label_text.split())

    for chunk_type in AVCOCKTAIL_CHUNK_TYPES:
        output_list: list[str] = []
        output_start_times: list[float] = []
        split_dataset = video_dataset[chunk_type]

        for index, sample in enumerate(
            tqdm(
                split_dataset,
                desc=f"Evaluating {set_name or ''}/{chunk_type}".strip(),
                total=len(split_dataset),
            )
        ):
            if max_samples_per_chunk is not None and index >= max_samples_per_chunk:
                break

            seg_start_time = float(_decode_bytes(sample["start_time"]))
            seg_end_time = float(_decode_bytes(sample["end_time"]))

            if (
                seg_start_time + 1 < label_start_time
                or seg_end_time - 1 > label_end_time
            ):
                continue

            output = engine.infer_processed_sample(sample["video"])

            output_list.append(output)
            output_start_times.append(seg_start_time)

        sorted_outputs = [
            output for _, output in sorted(zip(output_start_times, output_list))
        ]

        output_text = normalize_text_for_wer(" ".join(sorted_outputs))
        wer_scores[chunk_type] = wer(reference=label_text, hypothesis=output_text)

    return wer_scores, num_words


def evaluate_lrs2(args: argparse.Namespace, engine: MSPInferenceEngine) -> None:
    if args.set_id == "*":
        wer_scores: list[float] = []

        for split in LRS2_EVAL_SPLITS:
            print(f"\nEvaluating LRS2/{split}")
            dataset = datasets.load_dataset(
                "nguyenvulebinh/AVYT",
                "lrs2",
                split=split,
                streaming=args.streaming_dataset,
                cache_dir=args.dataset_cache_dir,
            )

            score = eval_lrs2(engine, dataset, max_samples=args.max_samples)
            wer_scores.append(score["wer"])
            print(f"WER {split}: {score['wer']:.4f}")

        print(f"\nAverage WER: {sum(wer_scores) / len(wer_scores):.4f}")

    else:
        if args.set_id not in LRS2_EVAL_SPLITS:
            raise ValueError(f"Invalid LRS2 set_id={args.set_id}")

        dataset = datasets.load_dataset(
            "nguyenvulebinh/AVYT",
            "lrs2",
            split=args.set_id,
            streaming=args.streaming_dataset,
            cache_dir=args.dataset_cache_dir,
        )

        score = eval_lrs2(engine, dataset, max_samples=args.max_samples)
        print(f"WER {args.set_id}: {score['wer']:.4f}")


def evaluate_avcocktail(args: argparse.Namespace, engine: MSPInferenceEngine) -> None:
    if args.set_id == "*":
        weighted_wer_scores: dict[str, list[float]] = {}

        for set_id in AVCOCKTAIL_VIDEO_IDS:
            print(f"\nEvaluating AVCocktail/{set_id}")

            video_dataset = datasets.load_dataset(
                "nguyenvulebinh/AVCocktail",
                set_id,
                cache_dir=args.dataset_cache_dir,
            )

            label_dataset = datasets.load_dataset(
                "nguyenvulebinh/AVCocktail",
                "labels",
                cache_dir=args.dataset_cache_dir,
            )[set_id]

            wer_scores, num_words = eval_avcocktail(
                engine,
                video_dataset,
                label_dataset,
                set_name=set_id,
                max_samples_per_chunk=args.max_samples_per_chunk,
            )

            for chunk_type, score in wer_scores.items():
                weighted_wer_scores.setdefault(chunk_type, [])
                weighted_wer_scores[chunk_type].extend([score] * num_words)
                print(f"WER {set_id} {chunk_type}: {score:.4f}")

        print("\nFinal AVCocktail weighted WER:")
        for chunk_type, scores in weighted_wer_scores.items():
            print(f"Average WER {chunk_type}: {sum(scores) / len(scores):.4f}")

    else:
        if args.set_id not in AVCOCKTAIL_VIDEO_IDS:
            raise ValueError(f"Invalid AVCocktail set_id={args.set_id}")

        video_dataset = datasets.load_dataset(
            "nguyenvulebinh/AVCocktail",
            args.set_id,
            cache_dir=args.dataset_cache_dir,
        )

        label_dataset = datasets.load_dataset(
            "nguyenvulebinh/AVCocktail",
            "labels",
            cache_dir=args.dataset_cache_dir,
        )[args.set_id]

        wer_scores, num_words = eval_avcocktail(
            engine,
            video_dataset,
            label_dataset,
            set_name=args.set_id,
            max_samples_per_chunk=args.max_samples_per_chunk,
        )

        for chunk_type, score in wer_scores.items():
            print(f"WER {chunk_type}: {score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MSP models using the AVSRCocktail-compatible protocol."
    )

    parser.add_argument(
        "--streaming_dataset",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=False,
    )

    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["msp_audio", "msp_visual", "msp"],
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["msp_audio", "msp_visual", "msp"],
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--processor_name_or_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["lrs2", "AVCocktail"],
    )

    parser.add_argument(
        "--set_id",
        type=str,
        default="*",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/workspace/cache/huggingface",
    )

    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default="/workspace/data/cache",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_samples_per_chunk",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = MSPInferenceEngine(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        processor_name_or_path=args.processor_name_or_path,
        cache_dir=args.cache_dir,
        device=args.device,
        modality=args.modality,
    )
    engine.load_model()

    if args.dataset_name == "lrs2":
        evaluate_lrs2(args, engine)

    elif args.dataset_name == "AVCocktail":
        evaluate_avcocktail(args, engine)

    else:
        raise ValueError(f"Unsupported dataset_name={args.dataset_name}")


if __name__ == "__main__":
    main()
