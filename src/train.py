import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .model import CAMCoTController, CRACoTController, GenerationHelper
from .preprocess import (
    build_prompts,
    compare_numbers,
    extract_final_number,
    load_dataset_split,
    normalize_answer,
)


def _set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalResult:
    accuracy: float
    avg_gen_tokens: float
    coverage_direct: float
    coverage_brief: float
    coverage_full: float
    direct_error: Optional[float]
    brief_error: Optional[float]
    selective_efficiency: float
    tokens_per_correct: float
    ood_guard_escalation_rate: float
    forced_full_rate: float


def _wandb_init(cfg):
    if cfg.wandb.mode == "disabled":
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run.run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    print(run.get_url())
    return run


def _log_step_metrics(step: int, metrics: Dict[str, Any]) -> None:
    metrics["step"] = step
    if wandb.run is not None:
        wandb.log(metrics)


@torch.no_grad()
def evaluate_split(
    examples: List[Dict[str, Any]],
    solver_fn,
    tokenizer,
    step_offset: int = 0,
) -> Tuple[EvalResult, List[Dict[str, Any]]]:
    correct = 0
    tokens = 0
    n = 0
    exits = {"direct": 0, "brief": 0, "full": 0}
    fast_err = {"direct": 0, "brief": 0}
    forced = 0
    forced_full = 0
    history_rows: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        q = ex["question"]
        gold = normalize_answer(ex["answer"], ex.get("normalized_answer"))
        out = solver_fn(q)
        pred = extract_final_number(out["text"], pattern=ex.get("extract_answer_regex"), normalize_commas=ex.get("normalize_commas", True))
        ok = pred is not None and compare_numbers(pred, gold)

        correct += int(ok)
        tokens += int(out["tokens"])
        n += 1
        exits[out["route"]] = exits.get(out["route"], 0) + 1
        if out["route"] in fast_err:
            fast_err[out["route"]] += int(not ok)
        if out.get("forced", False):
            forced += 1
            if out["route"] == "full":
                forced_full += 1

        # Batch-start assertion (step 0)
        if idx == 0:
            prompt = build_prompts(q)["direct"]
            inputs = tokenizer(prompt, return_tensors="pt")
            labels = tokenizer(gold, return_tensors="pt")
            assert inputs["input_ids"].ndim == labels["input_ids"].ndim == 2, "Input/label dims must match"
            assert inputs["input_ids"].shape[0] == labels["input_ids"].shape[0] == 1, "Batch size mismatch"

        running_acc = correct / max(n, 1)
        running_tokens = tokens / max(n, 1)

        step_metrics = {
            "eval_accuracy": float(ok),
            "eval_avg_gen_tokens": float(out["tokens"]),
            "eval_route_direct": 1.0 if out["route"] == "direct" else 0.0,
            "eval_route_brief": 1.0 if out["route"] == "brief" else 0.0,
            "eval_route_full": 1.0 if out["route"] == "full" else 0.0,
            "eval_direct_error_indicator": float((not ok) and out["route"] == "direct"),
            "eval_brief_error_indicator": float((not ok) and out["route"] == "brief"),
            "eval_forced": 1.0 if out.get("forced", False) else 0.0,
            "eval_score_s1": float(out.get("s1", math.nan)),
            "eval_score_s2": float(out.get("s2", math.nan)),
            "eval_cum_accuracy": running_acc,
            "eval_cum_avg_gen_tokens": running_tokens,
        }
        _log_step_metrics(step_offset + idx, step_metrics)
        history_rows.append(step_metrics)

    accuracy = correct / max(n, 1)
    avg_tokens = tokens / max(n, 1)
    coverage_direct = exits["direct"] / max(n, 1)
    coverage_brief = exits["brief"] / max(n, 1)
    coverage_full = exits["full"] / max(n, 1)
    direct_error = (fast_err["direct"] / exits["direct"]) if exits["direct"] else None
    brief_error = (fast_err["brief"] / exits["brief"]) if exits["brief"] else None
    selective_efficiency = accuracy / max(avg_tokens, 1e-8)
    tokens_per_correct = avg_tokens / max(accuracy, 1e-8)
    ood_guard_escalation_rate = forced / max(n, 1)
    forced_full_rate = forced_full / max(forced, 1) if forced else 0.0

    return (
        EvalResult(
            accuracy=accuracy,
            avg_gen_tokens=avg_tokens,
            coverage_direct=coverage_direct,
            coverage_brief=coverage_brief,
            coverage_full=coverage_full,
            direct_error=direct_error,
            brief_error=brief_error,
            selective_efficiency=selective_efficiency,
            tokens_per_correct=tokens_per_correct,
            ood_guard_escalation_rate=ood_guard_escalation_rate,
            forced_full_rate=forced_full_rate,
        ),
        history_rows,
    )


def _build_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache/")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"

    model = None
    last_error = None
    for cls in (AutoModelForSeq2SeqLM, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(
                cfg.model.name,
                torch_dtype=getattr(torch, cfg.model.dtype),
                device_map=cfg.model.device_map,
                cache_dir=".cache/",
            )
            break
        except Exception as e:
            last_error = e
    if model is None:
        raise RuntimeError(f"Failed to load model: {last_error}")

    model.eval()
    assert hasattr(model.config, "vocab_size") and model.config.vocab_size > 0, "Model vocab_size invalid"
    return model, tokenizer


def _apply_mode_overrides(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.dataset.calibration_size = min(cfg.dataset.calibration_size, 2)
        cfg.dataset.evaluation_size = min(cfg.dataset.evaluation_size, 2)
        cfg.dataset.max_examples = cfg.dataset.calibration_size + cfg.dataset.evaluation_size
        if hasattr(cfg.controller, "threshold_grid"):
            cfg.controller.threshold_grid.quantiles = min(int(cfg.controller.threshold_grid.quantiles), 5)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"


def train(cfg) -> None:
    _set_seed(42)
    _apply_mode_overrides(cfg)
    os.makedirs(cfg.results_dir, exist_ok=True)

    model, tokenizer = _build_model(cfg)

    # Post-init assertion
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set after init"

    gen_helper = GenerationHelper(model, tokenizer, cfg)

    dataset = load_dataset_split(cfg.dataset)
    assert len(dataset) >= cfg.dataset.max_examples, "Dataset size smaller than max_examples"
    examples = dataset.select(range(cfg.dataset.max_examples))

    cal_end = cfg.dataset.calibration_size
    eval_end = cal_end + cfg.dataset.evaluation_size
    assert eval_end <= len(examples), "Calibration + evaluation exceeds dataset size"

    cal = examples.select(range(cal_end))
    ev = examples.select(range(cal_end, eval_end))

    if "CAM-CoT" in cfg.method:
        controller = CAMCoTController(cfg, gen_helper)
    else:
        controller = CRACoTController(cfg, gen_helper)

    # Calibration
    calib = controller.calibrate(cal)
    assert calib is not None and "cal_scores" in calib, "Calibration failed"
    assert len(calib["cal_scores"]) == cfg.dataset.calibration_size, "Calibration scores size mismatch"

    run = _wandb_init(cfg)

    # Evaluation with per-example logging
    res, history_rows = evaluate_split(ev, controller.solve, tokenizer, step_offset=0)

    # Summary
    if wandb.run is not None:
        wandb.summary["final_accuracy"] = res.accuracy
        wandb.summary["final_avg_gen_tokens"] = res.avg_gen_tokens
        wandb.summary["final_selective_efficiency"] = res.selective_efficiency
        wandb.summary["final_tokens_per_correct"] = res.tokens_per_correct
        wandb.summary["final_coverage_direct"] = res.coverage_direct
        wandb.summary["final_coverage_brief"] = res.coverage_brief
        wandb.summary["final_coverage_full"] = res.coverage_full
        wandb.summary["final_direct_error"] = res.direct_error
        wandb.summary["final_brief_error"] = res.brief_error
        wandb.summary["final_ood_guard_escalation_rate"] = res.ood_guard_escalation_rate
        wandb.summary["final_forced_full_rate"] = res.forced_full_rate
        wandb.summary["cal_U1"] = calib.get("U1")
        wandb.summary["cal_U2"] = calib.get("U2")
        wandb.summary["cal_n1"] = calib.get("n1")
        wandb.summary["cal_n2"] = calib.get("n2")
        wandb.summary["cal_tau1"] = calib.get("tau1")
        wandb.summary["cal_tau2"] = calib.get("tau2")
        wandb.summary["cal_tau"] = calib.get("tau")
        wandb.summary["cal_cov1"] = calib.get("cov1")
        wandb.summary["cal_cov2"] = calib.get("cov2")

    if run is not None:
        run.finish()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
