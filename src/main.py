import os
import subprocess
import sys

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    run_id = cfg.run
    run_cfg_path = os.path.join("config", "runs", f"{run_id}.yaml")
    assert os.path.exists(run_cfg_path), f"Run config not found: {run_cfg_path}"

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        f"wandb.mode={cfg.wandb.mode}",
        f"optuna.n_trials={cfg.optuna.n_trials}",
    ]
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
