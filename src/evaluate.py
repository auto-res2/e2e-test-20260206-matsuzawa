import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix


def _load_wandb_config() -> Dict[str, str]:
    cfg = OmegaConf.load("config/config.yaml")
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _history_to_records(history: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for rec in history.to_dict(orient="records"):
        clean = {k: _sanitize_value(v) for k, v in rec.items()}
        records.append(clean)
    return records


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000) -> List[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(np.mean(sample))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def _plot_learning_curve(history: pd.DataFrame, out_path: Path, run_id: str) -> None:
    if "eval_accuracy_running" not in history.columns or "eval_step" not in history.columns:
        return
    data = history.dropna(subset=["eval_accuracy_running", "eval_step"])
    if data.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(data["eval_step"], data["eval_accuracy_running"], label="Running Accuracy")
    plt.scatter(data["eval_step"].iloc[-1], data["eval_accuracy_running"].iloc[-1])
    plt.title(f"{run_id} Learning Curve")
    plt.xlabel("Eval Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_demo_reliability(history: pd.DataFrame, out_path: Path, run_id: str) -> None:
    cols = [c for c in ["r_sc", "r_pi", "r_cc", "r"] if c in history.columns]
    if "demo_cluster" not in history.columns or not cols:
        return
    data = history.dropna(subset=["demo_cluster"])
    if data.empty:
        return
    data = data.drop_duplicates(subset=["demo_cluster"])
    plt.figure(figsize=(7, 4))
    for col in cols:
        plt.plot(data["demo_cluster"], data[col], marker="o", label=col)
    plt.title(f"{run_id} Demo Reliability Components")
    plt.xlabel("Cluster")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_error_histogram(history: pd.DataFrame, out_path: Path, run_id: str) -> None:
    if "eval_error" not in history.columns:
        return
    data = history.dropna(subset=["eval_error"])
    if data.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(data["eval_error"], bins=20, kde=True)
    plt.title(f"{run_id} Error Histogram")
    plt.xlabel("Prediction Error (pred - gold)")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _canonicalize_num(val: float) -> float:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return float("nan")
    if abs(val - round(val)) < 1e-6:
        return float(int(round(val)))
    return float(np.round(val, 3))


def _plot_confusion_matrix(history: pd.DataFrame, out_path: Path, run_id: str, top_k: int = 10) -> None:
    if "eval_pred" not in history.columns or "eval_gold" not in history.columns:
        return
    preds = pd.to_numeric(history["eval_pred"], errors="coerce")
    golds = pd.to_numeric(history["eval_gold"], errors="coerce")
    valid = ~(preds.isna() | golds.isna())
    preds = preds[valid]
    golds = golds[valid]
    if len(preds) == 0:
        return

    gold_can = golds.apply(_canonicalize_num)
    pred_can = preds.apply(_canonicalize_num)

    top_values = gold_can.value_counts().head(top_k).index.tolist()
    top_values = [v for v in top_values if not math.isnan(v)]

    def _bucket(val: float) -> str:
        return f"{val:g}" if val in top_values else "other"

    y_true = gold_can.apply(_bucket)
    y_pred = pred_can.apply(_bucket)
    labels = [f"{v:g}" for v in top_values] + ["other"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"{run_id} Confusion Matrix (Top {top_k} Answers)")
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_accuracy_boxplot(eval_correct_by_run: Dict[str, List[int]], out_path: Path) -> None:
    rows = []
    for run_id, corrects in eval_correct_by_run.items():
        for val in corrects:
            rows.append({"run_id": run_id, "correct": val})
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="run_id", y="correct", data=df)
    plt.ylabel("Correct (0/1)")
    plt.title("Comparison of Correctness Distributions")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_metrics_table(table_df: pd.DataFrame, out_path: Path) -> None:
    if table_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1 + table_df.shape[1]), 1 + 0.5 * table_df.shape[0]))
    ax.axis("off")
    table = ax.table(
        cellText=np.round(table_df.values, 4),
        rowLabels=table_df.index.tolist(),
        colLabels=table_df.columns.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _metric_is_lower_better(metric_name: str) -> bool:
    lower = metric_name.lower()
    return any(x in lower for x in ["loss", "error", "perplexity"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    wandb_cfg = _load_wandb_config()
    api = wandb.Api()
    run_ids = json.loads(args.run_ids)

    results_dir = Path(args.results_dir)
    metrics_by_name: Dict[str, Dict[str, float]] = defaultdict(dict)
    run_summaries: Dict[str, Dict[str, Any]] = {}
    run_histories: Dict[str, pd.DataFrame] = {}
    eval_correct_by_run: Dict[str, List[int]] = {}
    processed_run_ids: List[str] = []

    for run_id in run_ids:
        try:
            run = api.run(f"{wandb_cfg['entity']}/{wandb_cfg['project']}/{run_id}")
        except wandb.errors.CommError:
            print(f"Skipping missing run {run_id}")
            continue

        config = dict(run.config)
        mode = config.get("mode")
        wandb_mode = config.get("wandb.mode")
        if isinstance(config.get("wandb"), dict):
            wandb_mode = config.get("wandb", {}).get("mode", wandb_mode)

        if mode == "trial" or wandb_mode == "disabled":
            print(f"Skipping trial/disabled run {run_id}")
            continue

        history = run.history(pandas=True, samples=100000)
        summary = run.summary._json_dict

        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "summary": summary,
            "config": config,
            "history": _history_to_records(history),
        }
        metrics_path = run_dir / "metrics.json"
        _save_json(metrics_path, metrics)
        print(metrics_path)

        run_histories[run_id] = history
        run_summaries[run_id] = summary
        processed_run_ids.append(run_id)

        _plot_learning_curve(history, run_dir / f"{run_id}_learning_curve.pdf", run_id)
        _plot_demo_reliability(history, run_dir / f"{run_id}_demo_reliability.pdf", run_id)
        _plot_error_histogram(history, run_dir / f"{run_id}_error_histogram.pdf", run_id)
        _plot_confusion_matrix(history, run_dir / f"{run_id}_confusion_matrix.pdf", run_id)

        if "eval_correct" in history.columns:
            corrects = pd.to_numeric(history["eval_correct"], errors="coerce").dropna().astype(int).tolist()
            eval_correct_by_run[run_id] = corrects

        numeric_summary = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        numeric_cols = [
            col
            for col in history.columns
            if not col.startswith("_") and pd.to_numeric(history[col], errors="coerce").notna().any()
        ]
        metric_names = set(numeric_summary.keys()) | set(numeric_cols)
        for name in metric_names:
            if name in numeric_summary:
                metrics_by_name[name][run_id] = float(numeric_summary[name])
            else:
                series = pd.to_numeric(history[name], errors="coerce").dropna()
                if not series.empty:
                    metrics_by_name[name][run_id] = float(series.iloc[-1])

    if not processed_run_ids:
        print("No valid runs found for evaluation.")
        return

    primary_metric = "accuracy"
    primary_values: Dict[str, float] = {}
    if primary_metric in metrics_by_name:
        primary_values = dict(metrics_by_name[primary_metric])

    if not primary_values:
        for run_id in processed_run_ids:
            history = run_histories[run_id]
            if "eval_accuracy_running" in history.columns:
                valid = history.dropna(subset=["eval_accuracy_running"])
                if not valid.empty:
                    primary_values[run_id] = float(valid["eval_accuracy_running"].iloc[-1])

    best_proposed = {"run_id": None, "value": None}
    best_baseline = {"run_id": None, "value": None}
    for run_id, value in primary_values.items():
        if "proposed" in run_id:
            if best_proposed["value"] is None or value > best_proposed["value"]:
                best_proposed = {"run_id": run_id, "value": value}
        if "comparative" in run_id or "baseline" in run_id:
            if best_baseline["value"] is None or value > best_baseline["value"]:
                best_baseline = {"run_id": run_id, "value": value}

    gap = None
    if best_proposed["value"] is not None and best_baseline["value"] is not None:
        raw_gap = (best_proposed["value"] - best_baseline["value"]) / best_baseline["value"] * 100
        if _metric_is_lower_better(primary_metric):
            raw_gap = -raw_gap
        gap = float(raw_gap)

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics_by_name,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }

    if best_proposed["run_id"] and best_baseline["run_id"]:
        hist_p = run_histories[best_proposed["run_id"]]
        hist_b = run_histories[best_baseline["run_id"]]
        if "eval_correct" in hist_p.columns and "eval_correct" in hist_b.columns:
            corr_p = pd.to_numeric(hist_p["eval_correct"], errors="coerce").dropna().values
            corr_b = pd.to_numeric(hist_b["eval_correct"], errors="coerce").dropna().values
            if len(corr_p) > 1 and len(corr_b) > 1:
                t_stat, p_val = ttest_ind(corr_p, corr_b, equal_var=False)
                aggregated["significance"] = {"t_stat": float(t_stat), "p_value": float(p_val)}

    aggregated_path = comparison_dir / "aggregated_metrics.json"
    _save_json(aggregated_path, aggregated)
    print(aggregated_path)

    if primary_values:
        labels = list(primary_values.keys())
        values = np.array([primary_values[k] for k in labels])
        cis = []
        for run_id in labels:
            corr = eval_correct_by_run.get(run_id, [])
            cis.append(_bootstrap_ci(np.array(corr, dtype=float)) if corr else [np.nan, np.nan])
        lower = np.array([v - ci[0] if not np.isnan(ci[0]) else 0 for v, ci in zip(values, cis)])
        upper = np.array([ci[1] - v if not np.isnan(ci[1]) else 0 for v, ci in zip(values, cis)])
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values, yerr=[lower, upper], capsize=4)
        plt.ylabel(primary_metric.capitalize())
        plt.title("Comparison of Accuracy")
        plt.xticks(rotation=20, ha="right")
        for idx, val in enumerate(values):
            plt.text(idx, val + 0.01, f"{val:.3f}", ha="center")
        plt.tight_layout()
        out_path = comparison_dir / "comparison_accuracy_bar_chart.pdf"
        plt.savefig(out_path)
        plt.close()
        print(out_path)

    _plot_accuracy_boxplot(eval_correct_by_run, comparison_dir / "comparison_accuracy_box_plot.pdf")

    table_metrics = [
        primary_metric,
        "demo_acceptance_rate",
        "mean_r_sc",
        "mean_r_pi",
        "mean_r_cc",
        "mean_r",
        "paraphrase_filter_reject_rate",
        "grounding_utility_correlation",
    ]
    table_rows = []
    for run_id in processed_run_ids:
        row = {"run_id": run_id}
        for metric in table_metrics:
            row[metric] = metrics_by_name.get(metric, {}).get(run_id, float("nan"))
        table_rows.append(row)
    table_df = pd.DataFrame(table_rows).set_index("run_id")
    _plot_metrics_table(table_df, comparison_dir / "comparison_metrics_table.pdf")


if __name__ == "__main__":
    main()
