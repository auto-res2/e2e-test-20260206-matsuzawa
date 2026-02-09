import re
from typing import Any, Dict

from datasets import load_dataset


def normalize_answer(raw_answer: str, normalized: str | None = None) -> str:
    if normalized is not None:
        return normalized
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip()
    return raw_answer.strip()


def extract_final_number(text: str, pattern: str = r"-?\d+\.?\d*", normalize_commas: bool = True) -> str | None:
    if normalize_commas:
        text = text.replace(",", "")
    nums = re.findall(pattern, text)
    if not nums:
        return None
    return nums[-1]


def compare_numbers(pred: str, gold: str, tol: float = 1e-6) -> bool:
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        pred_int = pred_val.is_integer()
        gold_int = gold_val.is_integer()
        if pred_int and gold_int:
            return int(pred_val) == int(gold_val)
        return abs(pred_val - gold_val) <= tol
    except Exception:
        return False


def load_dataset_split(dataset_cfg) -> Any:
    ds = load_dataset(
        dataset_cfg.name,
        dataset_cfg.subset,
        split=dataset_cfg.split,
        cache_dir=".cache/",
    )
    return ds


def build_prompts(question: str) -> Dict[str, str]:
    direct = f"Answer with only the final numeric answer.\nQuestion: {question}\nAnswer:"
    brief = (
        "Solve the problem. Think step by step briefly (<= 4 steps). "
        "End with: Therefore, the answer is <number>.\n"
        f"Question: {question}\n"
    )
    full = (
        "Solve the problem. Let’s think step by step. "
        "End with: Therefore, the answer is <number>.\n"
        f"Question: {question}\n"
    )
    return {"direct": direct, "brief": brief, "full": full}
