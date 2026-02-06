import subprocess
import sys
from pathlib import Path
from typing import Any, List

import hydra
from omegaconf import OmegaConf


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


def _build_overrides(cfg: Any) -> List[str]:
    overrides = [
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    return overrides


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: Any) -> None:
    cfg = _apply_mode_overrides(cfg)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    overrides = _build_overrides(cfg)
    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
