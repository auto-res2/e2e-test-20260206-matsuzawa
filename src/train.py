import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.cluster import KMeans

from .model import (
    C3AutoCoTBuilder,
    EmbeddingWrapper,
    LLMWrapper,
    PIRAutoCoTBuilder,
    assign_clusters,
    compute_grounding_utility_correlation,
    evaluate_accuracy,
    set_seed,
    update_cfg_from_optuna_params,
)
from .preprocess import load_dataset_splits, set_cache_dirs


def _apply_mode_overrides(cfg: Any) -> Any:
    OmegaConf.set_struct(cfg, False)
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if "optuna" in cfg:
            cfg.optuna.n_trials = 0
        if "run" in cfg:
            cfg.run.optuna.n_trials = 0
            cfg.run.dataset.splits.train = "train[:50]"
            cfg.run.dataset.splits.test = "test[:20]"
            cfg.run.method_config.k_clusters = int(min(2, cfg.run.method_config.k_clusters))
            cfg.run.method_config.max_candidates_per_cluster = int(
                min(2, cfg.run.method_config.max_candidates_per_cluster)
            )
            cfg.run.method_config.m_self_consistency = int(
                min(2, cfg.run.method_config.m_self_consistency)
            )
            cfg.run.method_config.p_paraphrases = int(min(1, cfg.run.method_config.p_paraphrases))
            cfg.run.training.epochs = 1
            cfg.run.training.batch_size = 1
            cfg.run.training.eval_batch_size = 1
            cfg.run.training.gradient_accumulation_steps = 1
            cfg.run.training.max_train_batches = 2
            cfg.run.training.max_eval_batches = 2
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    return cfg


def _gradient_sanity_check(device: torch.device) -> None:
    dummy_param = torch.nn.Parameter(torch.tensor(1.0, device=device))
    optimizer = torch.optim.SGD([dummy_param], lr=0.1)
    loss = (dummy_param * 3.0).sum()
    grad = torch.autograd.grad(loss, dummy_param, create_graph=False)[0]
    assert grad is not None and torch.any(grad != 0), "Gradient sanity check failed"
    dummy_param.grad = grad
    assert dummy_param.grad is not None and torch.any(dummy_param.grad != 0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _post_init_asserts(llm: LLMWrapper) -> None:
    assert llm.tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"
    assert getattr(llm.model.config, "vocab_size", 0) > 0, "Model vocab size invalid"
    assert hasattr(llm.model, "generate"), "Model must implement generate()"
    assert llm.device is not None, "Model device not resolved"


def _get_builder(method_name: str, llm: LLMWrapper, embedder: EmbeddingWrapper, run_cfg: Any):
    if "C3" in method_name:
        return C3AutoCoTBuilder(llm=llm, embedder=embedder, run_cfg=run_cfg)
    if "PIR" in method_name:
        return PIRAutoCoTBuilder(llm=llm, embedder=embedder, run_cfg=run_cfg)
    raise ValueError(f"Unsupported method: {method_name}")


def _get_diag_flags(run_cfg: Any) -> Dict[str, bool]:
    diag_cfg = getattr(run_cfg.method_config, "diagnostics", None)
    return {
        "track_acceptance_rate": bool(getattr(diag_cfg, "track_acceptance_rate", False))
        if diag_cfg is not None
        else False,
        "track_reliability_components": bool(getattr(diag_cfg, "track_reliability_components", False))
        if diag_cfg is not None
        else False,
        "track_grounding_utility_correlation": bool(
            getattr(diag_cfg, "track_grounding_utility_correlation", False)
        )
        if diag_cfg is not None
        else False,
    }


def _run_optuna(
    cfg: Any,
    llm: LLMWrapper,
    embedder: EmbeddingWrapper,
    train_questions: List[str],
    train_embeddings: np.ndarray,
    kmeans: KMeans,
    test_examples: List[Dict[str, str]],
) -> Dict[str, Any]:
    optuna_cfg = cfg.optuna
    if optuna_cfg.n_trials <= 0:
        return {}
    search_spaces = optuna_cfg.search_spaces
    seed = int(cfg.run.training.seed)

    def objective(trial: optuna.Trial) -> float:
        trial_params: Dict[str, Any] = {}
        for space in search_spaces:
            if space.distribution_type == "uniform":
                trial_params[space.param_name] = trial.suggest_float(
                    space.param_name, float(space.low), float(space.high)
                )
            elif space.distribution_type == "categorical":
                trial_params[space.param_name] = trial.suggest_categorical(
                    space.param_name, list(space.choices)
                )
            else:
                raise ValueError(f"Unknown distribution_type: {space.distribution_type}")

        trial_cfg = update_cfg_from_optuna_params(cfg.run, trial_params)
        builder = _get_builder(trial_cfg.method, llm, embedder, trial_cfg)
        demos, _, _ = builder.build_demos(
            questions=train_questions,
            embeddings=train_embeddings,
            kmeans=kmeans,
            log_fn=None,
            log_step_start=0,
        )
        eval_limit = min(50, len(test_examples))
        greedy_eval = bool(trial_cfg.model.generation.greedy_eval)
        accuracy, _, _, _ = evaluate_accuracy(
            llm=llm,
            demos=demos,
            test_examples=test_examples[:eval_limit],
            max_new_tokens=trial_cfg.model.max_new_tokens,
            max_length=trial_cfg.dataset.preprocessing.max_length,
            do_sample=not greedy_eval,
            temperature=float(trial_cfg.model.generation.sample_temperature_sc),
            top_p=float(trial_cfg.model.generation.sample_top_p),
            log_fn=None,
            log_step_start=0,
            assert_batch_shapes=False,
        )
        return float(accuracy)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(optuna_cfg.n_trials))
    return dict(study.best_params)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: Any) -> None:
    cfg = _apply_mode_overrides(cfg)
    cfg.optuna = cfg.optuna if "optuna" in cfg else cfg.run.optuna
    run_cfg = cfg.run

    cache_dir = set_cache_dirs(".cache/")
    results_dir = Path(cfg.results_dir) / run_cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(run_cfg.training.seed))

    train_examples, test_examples = load_dataset_splits(run_cfg.dataset, cache_dir=cache_dir)
    train_questions = [ex["question"] for ex in train_examples]

    embedder = EmbeddingWrapper(run_cfg.method_config.embedder_name, cache_dir=cache_dir)
    train_embeddings = embedder.encode(train_questions, normalize=True)

    kmeans = KMeans(
        n_clusters=int(run_cfg.method_config.k_clusters),
        random_state=int(run_cfg.training.seed),
        n_init=10,
    ).fit(train_embeddings)

    llm = LLMWrapper(run_cfg.model, cache_dir=cache_dir)
    _post_init_asserts(llm)
    _gradient_sanity_check(llm.device)

    best_params: Dict[str, Any] = {}
    if int(cfg.optuna.n_trials) > 0:
        best_params = _run_optuna(
            cfg=cfg,
            llm=llm,
            embedder=embedder,
            train_questions=train_questions,
            train_embeddings=train_embeddings,
            kmeans=kmeans,
            test_examples=test_examples,
        )
        if best_params:
            run_cfg = update_cfg_from_optuna_params(run_cfg, best_params)
            cfg.run = run_cfg

    wandb_run = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(wandb_run.get_url())

    global_step = 0
    log_fn = None
    if wandb_run is not None:

        def _log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        log_fn = _log

    builder = _get_builder(run_cfg.method, llm, embedder, run_cfg)
    demos, demo_stats, global_step = builder.build_demos(
        questions=train_questions,
        embeddings=train_embeddings,
        kmeans=kmeans,
        log_fn=log_fn,
        log_step_start=global_step,
    )

    diag_flags = _get_diag_flags(run_cfg)
    accepted_flags = [s["accepted"] for s in demo_stats]
    acceptance_rate = float(np.mean(accepted_flags)) if accepted_flags else 0.0
    mean_candidates_tried = float(np.mean([s["candidates_tried"] for s in demo_stats])) if demo_stats else 0.0

    r_sc_vals = [s["r_sc"] for s in demo_stats if s["accepted"]]
    r_pi_vals = [s["r_pi"] for s in demo_stats if s["accepted"]]
    r_cc_vals = [s["r_cc"] for s in demo_stats if s["accepted"] and s["r_cc"] is not None]
    r_vals = [s["r"] for s in demo_stats if s["accepted"]]

    mean_r_sc = float(np.mean(r_sc_vals)) if r_sc_vals else 0.0
    mean_r_pi = float(np.mean(r_pi_vals)) if r_pi_vals else 0.0
    mean_r_cc = float(np.mean(r_cc_vals)) if r_cc_vals else 0.0
    mean_r = float(np.mean(r_vals)) if r_vals else 0.0

    para_gen = sum(s["n_paraphrases_generated"] for s in demo_stats)
    para_ok = sum(s["n_paraphrases_kept"] for s in demo_stats)
    paraphrase_filter_reject_rate = 1.0 - float(para_ok / max(1, para_gen))

    if log_fn is not None:
        metrics_payload: Dict[str, Any] = {}
        if diag_flags["track_acceptance_rate"]:
            metrics_payload.update(
                {
                    "demo_acceptance_rate": acceptance_rate,
                    "mean_candidates_tried": mean_candidates_tried,
                }
            )
        if diag_flags["track_reliability_components"]:
            metrics_payload.update(
                {
                    "mean_r_sc": mean_r_sc,
                    "mean_r_pi": mean_r_pi,
                    "mean_r_cc": mean_r_cc,
                    "mean_r": mean_r,
                    "paraphrase_filter_reject_rate": paraphrase_filter_reject_rate,
                }
            )
        if metrics_payload:
            log_fn(metrics_payload, step=global_step)

    eval_batch_size = int(run_cfg.training.eval_batch_size)
    max_eval_batches = getattr(run_cfg.training, "max_eval_batches", None)
    eval_limit = len(test_examples)
    if max_eval_batches is not None:
        eval_limit = min(eval_limit, int(max_eval_batches) * eval_batch_size)
    eval_examples = test_examples[:eval_limit]

    greedy_eval = bool(run_cfg.model.generation.greedy_eval)
    accuracy, preds, golds, corrects = evaluate_accuracy(
        llm=llm,
        demos=demos,
        test_examples=eval_examples,
        max_new_tokens=run_cfg.model.max_new_tokens,
        max_length=run_cfg.dataset.preprocessing.max_length,
        do_sample=not greedy_eval,
        temperature=float(run_cfg.model.generation.sample_temperature_sc),
        top_p=float(run_cfg.model.generation.sample_top_p),
        log_fn=log_fn,
        log_step_start=global_step,
        assert_batch_shapes=True,
    )

    grounding_corr: Dict[str, float] = {}
    if diag_flags["track_grounding_utility_correlation"]:
        test_questions = [ex["question"] for ex in eval_examples]
        test_embeddings = embedder.encode(test_questions, normalize=True)
        test_assignments = assign_clusters(test_embeddings, kmeans)
        grounding_corr = compute_grounding_utility_correlation(
            demo_stats=demo_stats,
            test_assignments=test_assignments,
            corrects=corrects,
            n_clusters=int(run_cfg.method_config.k_clusters),
        )

    if log_fn is not None:
        log_fn(
            {
                "accuracy": accuracy,
            },
            step=global_step + len(eval_examples),
        )
        if diag_flags["track_grounding_utility_correlation"]:
            log_fn(
                {
                    "grounding_utility_correlation": grounding_corr.get(
                        "grounding_utility_correlation", 0.0
                    ),
                    "grounding_utility_spearman": grounding_corr.get("grounding_utility_spearman", 0.0),
                },
                step=global_step + len(eval_examples) + 1,
            )
        wandb_run.summary["accuracy"] = accuracy
        if diag_flags["track_acceptance_rate"]:
            wandb_run.summary["demo_acceptance_rate"] = acceptance_rate
            wandb_run.summary["mean_candidates_tried"] = mean_candidates_tried
        if diag_flags["track_reliability_components"]:
            wandb_run.summary["mean_r_sc"] = mean_r_sc
            wandb_run.summary["mean_r_pi"] = mean_r_pi
            wandb_run.summary["mean_r_cc"] = mean_r_cc
            wandb_run.summary["mean_r"] = mean_r
            wandb_run.summary["paraphrase_filter_reject_rate"] = paraphrase_filter_reject_rate
        if diag_flags["track_grounding_utility_correlation"]:
            wandb_run.summary["grounding_utility_correlation"] = grounding_corr.get(
                "grounding_utility_correlation", 0.0
            )
            wandb_run.summary["grounding_utility_spearman"] = grounding_corr.get(
                "grounding_utility_spearman", 0.0
            )
        if best_params:
            wandb_run.summary["optuna_best_params"] = best_params
        wandb.finish()


if __name__ == "__main__":
    main()
