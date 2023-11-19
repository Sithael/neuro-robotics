import logging
from pathlib import Path

import gym
import pandas as pd
import wandb
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder
from utils.common import constants
from wandb.integration.sb3 import WandbCallback

import neuro_robotics
from neuro_robotics.algorithm.callbacks import HistoryCallback


class BaselineCore:
    def __init__(self, baseline_configuration: dict, model):
        self.baseline_configuration = baseline_configuration
        self.baseline_model = model
        self.device = self.baseline_configuration["inference"]["device"]

    def _register_experiment(self, experiment):
        checkpoint_dir = constants.DATA_SAVE_DIRECTORY_PATH
        root_save_directory = checkpoint_dir / experiment
        try:
            root_save_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise SystemError(f"Could not register experiment: {e}")
        return root_save_directory

    def _check_env_implementation(self, env):
        check_env(env)

    def _instantiate_env(self):
        env_identifier = self.baseline_configuration["baseline"]["env"]
        env = gym.make(env_identifier)
        env = Monitor(env)
        return env

    def _instantiate_wandb(self, wandb_configuration, datapoint):
        wandb.login()
        run = wandb.init(
            entity=wandb_configuration["entity"],
            project=wandb_configuration["project"],
            config=wandb_configuration["config"],
            dir=datapoint,
            sync_tensorboard=wandb_configuration["sync_tensorboard"],
            monitor_gym=wandb_configuration["monitor_gym"],
            save_code=wandb_configuration["save_code"],
        )
        wandb.run.name = datapoint.name
        return run

    def _chain_callbacks(self, eval_env, datapoint):
        callback_list = []
        checkpoint_callback_settings = self.baseline_configuration["callback"]["eval"]
        performance_callback_settings = self.baseline_configuration["callback"][
            "performance"
        ]
        history_callback_settings = self.baseline_configuration["callback"]["history"][
            "configuration"
        ]

        if checkpoint_callback_settings["use"]:
            checkpoint_callback = EvalCallback(
                eval_env,
                best_model_save_path=datapoint
                / checkpoint_callback_settings["checkpoint"],
                log_path=datapoint / checkpoint_callback_settings["log"],
                eval_freq=checkpoint_callback_settings["frequency"],
                deterministic=checkpoint_callback_settings["deterministic"],
                render=checkpoint_callback_settings["render"],
            )
            callback_list.append(checkpoint_callback)

        if performance_callback_settings["use"]:
            performance_callback = WandbCallback(
                gradient_save_freq=performance_callback_settings["grad_save_freq"],
                model_save_path=datapoint / performance_callback_settings["checkpoint"],
                verbose=performance_callback_settings["verbose"],
            )
            callback_list.append(performance_callback)

        if history_callback_settings["use"]:
            history_callback = HistoryCallback(
                experiment_dir=datapoint,
                csv_path=constants.DATA_SAVE_DIRECTORY_PATH.parent
                / f'{history_callback_settings["csv_file"]}.csv',
                n_evals=history_callback_settings["n_evals"],
                device=self.device,
            )
            callback_list.append(history_callback)

        callbacks = CallbackList(callback_list)
        return callbacks

    def _instantiate_model(self, env, policy, verbose, tensorboard_log):
        model = self.baseline_model(
            policy=policy,
            env=env,
            replay_buffer_class=HerReplayBuffer,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=self.device,
        )
        return model

    def _load_pretrained_model(self, path_to_model: Path, env):
        return self.baseline_model.load(path_to_model, env, device=self.device)

    def _fetch_best_model_from_csv(self, csv_file: Path, score_thr) -> Path:
        best_model = None
        csv_exists = csv_file.exists()
        if csv_exists:
            df = pd.read_csv(csv_file)
            if df.empty:
                logging.info(f"csv file is empty: {csv_file.name}")
            else:
                sorted_df = df.sort_values(
                    ["mean_reward", "std_reward"], ascending=False
                )
                if sorted_df.iloc[0]["mean_reward"] >= score_thr:
                    best_model = sorted_df.iloc[0]["model"]
        return best_model

    def _update_csv_file(self, model, csv_f):
        frame = {
            "model": [model],
            "mean_reward": ["pretrained"],
            "std_reward": ["pretrained"],
        }
        df = pd.DataFrame.from_dict(frame)
        with open(csv_f, "a") as f:
            df.to_csv(f, mode="a", header=f.tell() == 0, index=False)

    def train_model(self):
        date = neuro_robotics.date_of_instantiation
        experiment_identifier = self._register_experiment(date)

        if self.baseline_configuration["baseline"]["record"]:
            env = DummyVecEnv([self._instantiate_env])
            video_save_pth = str(
                experiment_identifier
                / self.baseline_configuration["baseline"]["video_path"]
            )
            record_frequency = self.baseline_configuration["baseline"][
                "record_frequency"
            ]
            env = VecVideoRecorder(
                env,
                video_save_pth,
                record_video_trigger=lambda x: not (x % record_frequency),
                video_length=200,
            )
        else:
            env = self._instantiate_env()

        tensorboard_log = (
            experiment_identifier
            / self.baseline_configuration["baseline"]["tensorboard_log"]
        )

        datapoint = None
        if self.baseline_configuration["callback"]["performance"]["use"]:
            wandb_configuration = self.baseline_configuration["wandb"]
            datapoint = self._instantiate_wandb(
                wandb_configuration, experiment_identifier
            )

        # TODO: refactor model choosing mechanism ( works but messy )
        policy = self.baseline_configuration["baseline"]["policy_type"]
        verbose = self.baseline_configuration["baseline"]["verbose"]
        online_settings = self.baseline_configuration["online"]
        if online_settings["use"]:
            csv_file = (
                constants.DATA_SAVE_DIRECTORY_PATH.parent
                / f'{online_settings["csv_file"]}.csv'
            )
            if self.baseline_configuration["inference"]["pretrained"]:
                """
                #TODO: move logic to _fetch_best_model method
                pretrained_used = 0
                pretrained_model_pth = constants.PRETRAINED_MODEL_PATH
                csv_exists = csv_file.exists()
                if csv_exists:
                    df = pd.read_csv(csv_file)
                    pretrained_already_loaded = df[df['model'] == pretrained_model_pth]
                    if pretrained_already_loaded:
                        pretrained_used = 1
                    else:
                        model = self._load_pretrained_model(pretrained_model_pth, env)
                        self._update_csv_file(pretrained_model_pth, csv_file)
                else:
                    model = self._load_pretrained_model(pretrained_model_pth, env)
                    self._update_csv_file(pretrained_model_pth, csv_file)
                """
                raise SystemError(
                    "Can not initiate online settings with pretrained model"
                )
            score_thresold = online_settings["score_threshold"]
            best_model_path = self._fetch_best_model_from_csv(csv_file, score_thresold)
            if best_model_path is not None:
                model = self._load_pretrained_model(best_model_path, env)
            else:
                model = self._instantiate_model(env, policy, verbose, tensorboard_log)
        elif self.baseline_configuration["inference"]["pretrained"]:
            model_pth = constants.PRETRAINED_MODEL_PATH
            model = self._load_pretrained_model(model_pth, env)
        else:
            model = self._instantiate_model(env, policy, verbose, tensorboard_log)

        callback_chain = self._chain_callbacks(env, experiment_identifier)

        if datapoint is not None:
            with datapoint:
                model.learn(
                    total_timesteps=self.baseline_configuration["baseline"][
                        "total_timesteps"
                    ],
                    callback=callback_chain,
                )
        else:
            model.learn(
                total_timesteps=self.baseline_configuration["baseline"][
                    "total_timesteps"
                ],
                callback=callback_chain,
            )

    def evaluate_model(self):
        eval_settings = self.baseline_configuration["evaluator"]
        path_to_model = (
            constants.DATA_SAVE_DIRECTORY_PATH
            / eval_settings["dir"]
            / eval_settings["checkpoint"]
        )
        env = self._instantiate_env()
        model = self.baseline_model.load(path_to_model, env, device=self.device)
        n_episodes = eval_settings["n_episodes"]
        deterministic = eval_settings["deterministic"]
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_episodes, deterministic=deterministic
        )
