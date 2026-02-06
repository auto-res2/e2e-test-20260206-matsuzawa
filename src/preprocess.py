import os
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


CACHE_DIR = ".cache/"


def set_cache_dirs(cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(cache_dir, "sentence_transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cache_dir


def _process_split(
    dataset: Any,
    field_q: str,
    field_a: str,
    lowercase: bool,
) -> List[Dict[str, str]]:
    processed: List[Dict[str, str]] = []
    for ex in dataset:
        question = ex.get(field_q)
        answer = ex.get(field_a)
        if question is None or answer is None:
            continue
        question = str(question)
        answer = str(answer)
        if lowercase:
            question = question.lower()
            answer = answer.lower()
        processed.append({"question": question, "answer": answer})
    return processed


def load_dataset_splits(dataset_cfg: Any, cache_dir: str = CACHE_DIR) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    config_name = dataset_cfg.get("config_name") if hasattr(dataset_cfg, "get") else None
    load_kwargs = {"cache_dir": cache_dir}
    if config_name:
        load_kwargs["name"] = config_name

    train_split = dataset_cfg.splits.train
    test_split = dataset_cfg.splits.test

    train_ds = load_dataset(dataset_cfg.name, split=train_split, **load_kwargs)
    test_ds = load_dataset(dataset_cfg.name, split=test_split, **load_kwargs)

    field_q = dataset_cfg.fields.question
    field_a = dataset_cfg.fields.answer
    if field_q not in train_ds.column_names or field_a not in train_ds.column_names:
        raise ValueError(f"Dataset missing required fields: {field_q}, {field_a}")

    lowercase = bool(dataset_cfg.preprocessing.get("lowercase", False))
    train_examples = _process_split(train_ds, field_q, field_a, lowercase)
    test_examples = _process_split(test_ds, field_q, field_a, lowercase)

    if len(train_examples) == 0 or len(test_examples) == 0:
        raise ValueError("Loaded dataset splits are empty after preprocessing")

    return train_examples, test_examples
