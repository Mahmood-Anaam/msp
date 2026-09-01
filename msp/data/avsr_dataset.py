import os
import time

import datasets

os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"


def load_avsr_dataset(
    cache_dir="/workspace/data/cache", include_mcorec=False, streaming=False
):
    # streaming=True to avoid downloading all dataset at once, but it can be crash if network is unstable
    # streaming=False to download all dataset at once, it take time and around 1.5TB disk space. More stable.

    def format_sample(sample):
        sample["label"] = str(sample["label"], encoding="utf-8")
        sample["length"] = int(sample["length"])
        sample["sample_id"] = str(sample["sample_id"], encoding="utf-8")
        return sample

    # Load dataset
    finished_loading = False
    try_times = 0
    max_try_times = 5

    while not finished_loading:
        try:
            # Load dataset. It's quite bigdataset and sometime downloading can break. You can simple retry.
            lrs2 = datasets.load_dataset(
                "nguyenvulebinh/AVYT", "lrs2", streaming=streaming, cache_dir=cache_dir
            ).remove_columns(["__key__", "__url__"])
            vox2 = datasets.load_dataset(
                "nguyenvulebinh/AVYT", "vox2", streaming=streaming, cache_dir=cache_dir
            ).remove_columns(["__key__", "__url__"])
            avyt = datasets.load_dataset(
                "nguyenvulebinh/AVYT", "avyt", streaming=streaming, cache_dir=cache_dir
            ).remove_columns(["__key__", "__url__"])
            avyt_mix = datasets.load_dataset(
                "nguyenvulebinh/AVYT",
                "avyt-mix",
                streaming=streaming,
                cache_dir=cache_dir,
            ).remove_columns(["__key__", "__url__"])
            # Load mcorec dataset. Ensure you have permission to use this dataset.
            if include_mcorec:
                print("Loading MCoRec dataset")
                mcorec_dataset = datasets.load_dataset(
                    "MCoRecChallenge/MCoRec", streaming=streaming, cache_dir=cache_dir
                ).remove_columns(["__key__", "__url__"])
            finished_loading = True
        except Exception as e:
            try_times += 1
            if try_times >= max_try_times:
                raise e
            time.sleep(10)

    if not streaming:
        # That mean above datasets are already downloaded and cached
        list_datasets = [lrs2, vox2, avyt, avyt_mix]
        if include_mcorec:
            list_datasets.append(mcorec_dataset)
        for ds in list_datasets:
            for split in ds.keys():
                split_size = len(ds[split])
                if split_size > 10000:
                    num_shards = max(20, split_size // 10000)
                else:
                    num_shards = 1
                ds[split] = ds[split].to_iterable_dataset(num_shards=num_shards)
                print(
                    f"Split {split} has {split_size} samples and {ds[split].num_shards} shards"
                )

    if include_mcorec:
        map_dataset_probabilities = {
            "lrs2": 0.25,
            "vox2": 0.10,
            "avyt": 0.20,
            "avyt-mix": 0.25,
            "mcorec": 0.2,
        }
    else:
        map_dataset_probabilities = {
            "lrs2": 0.3,
            "vox2": 0.2,
            "avyt": 0.25,
            "avyt-mix": 0.25,
        }

    map_datasets = {
        "lrs2": {
            "probabilities": map_dataset_probabilities["lrs2"],
            "dataset": {
                "train": datasets.concatenate_datasets(
                    [lrs2["train"], lrs2["pretrain"]]
                ),
                "valid": datasets.concatenate_datasets(
                    [lrs2["valid"], lrs2["test_snr_0_interferer_2"]]
                )
                if not include_mcorec
                else None,
            },
        },
        "vox2": {
            "probabilities": map_dataset_probabilities["vox2"],
            "dataset": {
                "train": vox2["dev"],
                "valid": None,
            },
        },
        "avyt": {
            "probabilities": map_dataset_probabilities["avyt"],
            "dataset": {
                "train": datasets.concatenate_datasets(
                    [avyt["talking"], avyt["silent"]]
                ),
                "valid": None,
            },
        },
        "avyt-mix": {
            "probabilities": map_dataset_probabilities["avyt-mix"],
            "dataset": {
                "train": avyt_mix["train"],
                "valid": avyt_mix["test"] if not include_mcorec else None,
            },
        },
        "mcorec": {
            "probabilities": map_dataset_probabilities["mcorec"]
            if include_mcorec
            else 0,
            "dataset": {
                "train": mcorec_dataset["train"] if include_mcorec else None,
                "valid": mcorec_dataset["valid"] if include_mcorec else None,
            },
        },
    }
    print("map_datasets\n", map_datasets)

    train_dataset = datasets.interleave_datasets(
        [
            item["dataset"]["train"]
            for item in map_datasets.values()
            if item["dataset"]["train"] is not None
        ],
        seed=11,
        probabilities=[
            item["probabilities"]
            for item in map_datasets.values()
            if item["dataset"]["train"] is not None
        ],
        stopping_strategy="all_exhausted",
    )
    valid_dataset = datasets.interleave_datasets(
        [
            item["dataset"]["valid"]
            for item in map_datasets.values()
            if item["dataset"]["valid"] is not None
        ],
        stopping_strategy="first_exhausted",
    )

    train_dataset = train_dataset.map(format_sample)
    valid_dataset = valid_dataset.map(format_sample)

    # load lrs2 for interference speech
    # interference_speech = None
    print(
        "Loading interference speech dataset. Actual file around 10GB need to download. This may take a while..."
    )
    interference_speech = datasets.load_dataset(
        "nguyenvulebinh/AVYT",
        "lrs2",
        cache_dir=cache_dir,
        data_files="lrs2/lrs2-train-*.tar",
    ).remove_columns(["__key__", "__url__"])["train"]
    return train_dataset, valid_dataset, interference_speech
