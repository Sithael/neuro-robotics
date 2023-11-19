import logging

import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class HistoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, experiment_dir, csv_path, n_evals, device, verbose=0) -> None:
        super().__init__(verbose)
        self.experiment_dir = experiment_dir
        self.csv_path = csv_path
        self.n_evals = n_evals
        self.device = device

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        best_model_pth = self.experiment_dir / "best_model" / "best_model.zip"
        if best_model_pth.exists():
            best_model = DDPG.load(
                best_model_pth, self.training_env, device=self.device
            )
            mean_reward, std_reward = evaluate_policy(
                best_model,
                self.training_env,
                n_eval_episodes=self.n_evals,
                deterministic=True,
            )
            frame = {
                "model": [best_model_pth],
                "mean_reward": [mean_reward],
                "std_reward": [std_reward],
            }
            df = pd.DataFrame.from_dict(frame)
            with open(self.csv_path, "a") as f:
                df.to_csv(f, mode="a", header=f.tell() == 0, index=False)

        else:
            logging.info("Best model was not registered")
