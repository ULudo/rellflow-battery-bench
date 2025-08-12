import gymnasium as gym
import numpy as np
import os
import csv
import time
import uuid
from datetime import datetime
from typing import Dict, Tuple

from .building_env import BuildingEnv
from .consts_and_types import BuildingEnvObservation


class RewardScalingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        soc_penalty: bool = False,
    ):
        super().__init__(env)
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.soc_penalty = soc_penalty
        if self.reward_std <= 0.0:
            raise ValueError("reward_std must be positive")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.soc_penalty:
            reward += -((self.env.battery.soc - 0.5) ** 2)
        scaled_reward = (reward - self.reward_mean) / self.reward_std
        info["scaled_reward"] = scaled_reward
        return observation, scaled_reward, terminated, truncated, info


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length=10):
        super().__init__(env)
        self.history = None
        self.history_length = history_length
        self.single_obs_shape = env.observation_space.shape
        self.single_obs_space = env.observation_space
        low = np.stack([self.single_obs_space.low for _ in range(self.history_length)])
        high = np.stack([self.single_obs_space.high for _ in range(self.history_length)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.single_obs_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history = np.zeros((self.history_length, *self.single_obs_shape), dtype=self.single_obs_space.dtype)
        self.history[-1] = obs
        return self.history, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs
        return self.history, reward, terminated, truncated, info


class CSVWrapper(gym.Wrapper):
    def __init__(self, env, log_dir="logs"):
        super().__init__(env)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        self.filename = os.path.join(log_dir, f"env_log_{timestamp}_{unique_id}.csv")
        self.obs_dim = np.prod(env.observation_space.shape)
        self.action_dim = np.prod(env.action_space.shape) if hasattr(env.action_space, "shape") else 1
        self.csv_file = open(self.filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["step", "timestamp", "event"]
        if self.action_dim == 1:
            header.append("action")
        else:
            header.extend([f"action_{i}" for i in range(self.action_dim)])
        header.extend(["reward", "terminated"])
        header.extend([f"obs_{i}" for i in range(self.obs_dim)])
        self.csv_writer.writerow(header)
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self._log_data("reset", "N/A", obs, 0.0, False)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self._log_data("step", action, obs, reward, terminated)
        return obs, reward, terminated, truncated, info

    def _log_data(self, event, action, observation, reward, terminated):
        flattened_obs = observation.flatten()
        row_data = [self.step_count, time.time(), event]
        if action == "N/A":
            if self.action_dim == 1:
                row_data.append(action)
            else:
                row_data.extend(["N/A"] * self.action_dim)
        else:
            try:
                if hasattr(action, "flatten"):
                    action_values = action.flatten()
                elif hasattr(action, "__iter__") and not isinstance(action, str):
                    action_values = list(action)
                else:
                    action_values = [action]
                row_data.extend(action_values)
            except Exception:
                if self.action_dim == 1:
                    row_data.append(str(action))
                else:
                    row_data.extend(["ERR"] * self.action_dim)
        row_data.extend([reward, terminated])
        row_data.extend(flattened_obs)
        self.csv_writer.writerow(row_data)
        if self.step_count % 100 == 0:
            self.csv_file.flush()

    def close(self):
        if hasattr(self, "csv_file") and self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        return self.env.close()


