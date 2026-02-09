import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf


def _load_cfg():
    return OmegaConf.load("config/config.yaml")


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: Path) -> List[str]:
    paths = []
    if "eval_cum_accuracy" in history.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["eval_cum_accuracy"], label="cumulative accuracy")
        ax.set_title(f"{run_id} cumulative accuracy")
        ax.set_xlabel("step")
        ax.set_ylabel("accuracy")
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"{run_id}_learning_curve.pdf"
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _plot_route_fractions(summary: Dict, run_id: str, out_dir: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["direct", "brief", "full"]
    vals = [
        summary.get("final_coverage_direct", 0.0),
        summary.get("final_coverage_brief", 0.0),
        summary.get("final_coverage_full", 0.0),
    ]
    ax.bar(labels, vals)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{run_id} route fractions")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p = out_dir / f"{run_id}_route_fractions.pdf"
    fig.savefig(p)
    plt.close(fig)
    return [str(p)]


def _plot_accuracy_tokens(history: pd.DataFrame, run_id: str, out_dir: Path) -> List[str]:
    if "eval_accuracy" not in history.columns or "eval_avg_gen_tokens" not in history.columns:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(history["eval_avg_gen_tokens"], history["eval_accuracy"], alpha=0.6)
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("accuracy (0/1)")
    ax.set_title(f"{run_id} accuracy vs tokens")
    fig.tight_layout()
    p = out_dir / f"{run_id}_accuracy_tokens_scatter.pdf"
    fig.savefig(p)
    plt.close(fig)
    return [str(p)]


def _plot_stagewise_error(history: pd.DataFrame, summary: Dict, run_id: str, out_dir: Path) -> List[str]:
    direct = history["eval_route_direct"] == 1.0
    brief = history["eval_route_brief"] == 1.0
    direct_err = history.loc[direct, "eval_accuracy"].apply(lambda x: 1 - x).sum()
    brief_err = history.loc[brief, "eval_accuracy"].apply(lambda x: 1 - x).sum()
    direct_n = int(direct.sum())
    brief_n = int(brief.sum())
    direct_rate = (direct_err / direct_n) if direct_n else 0.0
    brief_rate = (brief_err / brief_n) if brief_n else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["direct", "brief"]
    rates = [direct_rate, brief_rate]
    ax.bar(labels, rates, color=["#4C72B0", "#55A868"])
    for i, v in enumerate(rates):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("error rate")
    ax.set_title(f"{run_id} stagewise error")
    fig.tight_layout()
    p = out_dir / f"{run_id}_stagewise_error.pdf"
    fig.savefig(p)
    plt.close(fig)
    return [str(p)]


def _plot_summary_bars(summary: Dict, run_id: str, out_dir: Path) -> List[str]:
    keys = ["final_accuracy", "final_avg_gen_tokens", "final_selective_efficiency", "final_tokens_per_correct"]
    values = [summary.get(k, np.nan) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keys, values)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{run_id} summary metrics")
    fig.tight_layout()
    p = out_dir / f"{run_id}_summary_bar.pdf"
    fig.savefig(p)
    plt.close(fig)
    return [str(p)]


def _comparison_bar(metric: str, data: Dict[str, float], out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    run_ids = list(data.keys())
    vals = [data[r] for r in run_ids]
    ax.bar(run_ids, vals)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = out_dir / f"comparison_{metric}_bar.pdf"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def _comparison_table(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> str:
    df = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(cellText=np.round(df.values, 4), colLabels=df.columns, rowLabels=df.index, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()
    p = out_dir / "comparison_metrics_table.pdf"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def _permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 1000) -> float:
    observed = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        new_a = combined[: len(a)]
        new_b = combined[len(a) :]
        if abs(np.mean(new_a) - np.mean(new_b)) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def _is_minimization_metric(metric: str) -> bool:
    lower_better = ["loss", "perplexity", "error", "avg_gen_tokens", "tokens_per_correct"]
    return any(k in metric for k in lower_better)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    cfg = _load_cfg()
    api = wandb.Api()
    run_ids = json.loads(args.run_ids)

    all_metrics: Dict[str, Dict[str, float]] = {}
    per_run_paths: List[str] = []
    run_histories: Dict[str, pd.DataFrame] = {}

    for run_id in run_ids:
        run = api.run(f"{cfg.wandb.entity}/{cfg.wandb.project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        out_dir = Path(args.results_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_json(out_dir / "metrics.json", {
            "history": history.to_dict(orient="list"),
            "summary": summary,
            "config": config,
        })

        paths: List[str] = []
        paths += _plot_learning_curve(history, run_id, out_dir)
        paths += _plot_route_fractions(summary, run_id, out_dir)
        paths += _plot_accuracy_tokens(history, run_id, out_dir)
        paths += _plot_stagewise_error(history, summary, run_id, out_dir)
        paths += _plot_summary_bars(summary, run_id, out_dir)
        per_run_paths += paths

        run_histories[run_id] = history

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, {})[run_id] = float(v)

    comp_dir = Path(args.results_dir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    primary_metric_key = "final_accuracy"
    metrics_out = {
        "primary_metric": "accuracy",
        "metrics": all_metrics,
        "best_proposed": {"run_id": None, "value": None},
        "best_baseline": {"run_id": None, "value": None},
        "gap": None,
    }

    if primary_metric_key in all_metrics:
        proposed = {k: v for k, v in all_metrics[primary_metric_key].items() if "proposed" in k}
        baseline = {k: v for k, v in all_metrics[primary_metric_key].items() if "comparative" in k or "baseline" in k}
        if proposed:
            best_p = max(proposed.items(), key=lambda x: x[1])
            metrics_out["best_proposed"] = {"run_id": best_p[0], "value": best_p[1]}
        if baseline:
            best_b = max(baseline.items(), key=lambda x: x[1])
            metrics_out["best_baseline"] = {"run_id": best_b[0], "value": best_b[1]}
        if metrics_out["best_proposed"]["value"] is not None and metrics_out["best_baseline"]["value"] is not None:
            bp = metrics_out["best_proposed"]["value"]
            bb = metrics_out["best_baseline"]["value"]
            gap = (bp - bb) / max(bb, 1e-8) * 100
            metrics_out["gap"] = gap

    _save_json(comp_dir / "aggregated_metrics.json", metrics_out)

    comp_paths = []
    for metric, data in all_metrics.items():
        if metric in ["final_accuracy", "final_avg_gen_tokens", "final_selective_efficiency", "final_tokens_per_correct"]:
            comp_paths.append(_comparison_bar(metric, data, comp_dir))

    if all_metrics:
        comp_paths.append(_comparison_table(all_metrics, comp_dir))

    # Statistical significance test for accuracy
    if metrics_out["best_proposed"]["run_id"] and metrics_out["best_baseline"]["run_id"]:
        rp = metrics_out["best_proposed"]["run_id"]
        rb = metrics_out["best_baseline"]["run_id"]
        hp = run_histories[rp]
        hb = run_histories[rb]
        if "eval_accuracy" in hp.columns and "eval_accuracy" in hb.columns:
            pval = _permutation_test(hp["eval_accuracy"].values, hb["eval_accuracy"].values)
            metrics_out["significance"] = {"accuracy_pvalue": float(pval)}
            _save_json(comp_dir / "aggregated_metrics.json", metrics_out)

    for p in per_run_paths + comp_paths:
        print(p)


if __name__ == "__main__":
    main()
