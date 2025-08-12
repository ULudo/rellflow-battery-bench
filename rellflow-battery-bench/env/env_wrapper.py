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
    """Wrapper that scales rewards by mean and standard deviation."""

    def __init__(
        self,
        env,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        soc_penalty: bool = False,
    ):
        """
        Initialize the wrapper.

        Args:
            env: The environment to wrap
            reward_mean: Mean value to subtract from rewards
            reward_std: Standard deviation to divide rewards by
        """
        super().__init__(env)
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.soc_penalty = soc_penalty

        if self.reward_std <= 0.0:
            raise ValueError("reward_std must be positive")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Pull battery SoC to the middle of the range
        if self.soc_penalty:
            reward += -((self.env.battery.soc - 0.5) ** 2)

        # Scale the reward by subtracting mean and dividing by std
        scaled_reward = (reward - self.reward_mean) / self.reward_std

        # Store both original and scaled rewards in info
        info["scaled_reward"] = scaled_reward

        return observation, scaled_reward, terminated, truncated, info


class HistoryWrapper(gym.Wrapper):
    """Wrapper that maintains k-order hostory of past observations."""

    def __init__(self, env, history_length=10):
        super().__init__(env)
        self.history = None
        self.history_length = history_length

        # Original observation shape
        self.single_obs_shape = env.observation_space.shape
        self.single_obs_space = env.observation_space

        # Modify observation space to include history
        low = np.stack([self.single_obs_space.low for _ in range(self.history_length)])
        high = np.stack(
            [self.single_obs_space.high for _ in range(self.history_length)]
        )
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.single_obs_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize history array - sequence dimension first
        self.history = np.zeros(
            (self.history_length, *self.single_obs_shape),
            dtype=self.single_obs_space.dtype,
        )

        # Set the current observation to the last position
        self.history[-1] = obs

        return self.history, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Shift history one time step up (dropping the oldest)
        self.history[:-1] = self.history[1:]

        # Set the newest observation in the last position
        self.history[-1] = obs

        return self.history, reward, terminated, truncated, info


class CSVWrapper(gym.Wrapper):
    """Wrapper that logs environment data to a CSV file."""

    def __init__(self, env, log_dir="logs"):
        super().__init__(env)

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Generate a unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        self.filename = os.path.join(log_dir, f"env_log_{timestamp}_{unique_id}.csv")

        # Determine observation dimension and action dimension
        self.obs_dim = np.prod(env.observation_space.shape)
        self.action_dim = (
            np.prod(env.action_space.shape) if hasattr(env.action_space, "shape") else 1
        )

        # Set up CSV file and writer
        self.csv_file = open(self.filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write header row with observation feature indices
        header = ["step", "timestamp", "event"]
        # Add action columns
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

        # Log reset event with N/A for action
        self.step_count = 0
        self._log_data("reset", "N/A", obs, 0.0, False)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Log step event with the action
        self.step_count += 1
        self._log_data("step", action, obs, reward, terminated)

        return obs, reward, terminated, truncated, info

    def _log_data(self, event, action, observation, reward, terminated):
        # Flatten the observation array to ensure all values are logged
        flattened_obs = observation.flatten()

        # Prepare the basic row data
        row_data = [
            self.step_count,
            time.time(),
            event,
        ]

        # Process and add action data
        if action == "N/A":
            # Handle reset case
            if self.action_dim == 1:
                row_data.append(action)
            else:
                row_data.extend(["N/A"] * self.action_dim)
        else:
            # Handle step case - flatten action if it's an array
            try:
                if hasattr(action, "flatten"):
                    # For numpy arrays
                    action_values = action.flatten()
                elif hasattr(action, "__iter__") and not isinstance(action, str):
                    # For other iterables (lists, tuples)
                    action_values = list(action)
                else:
                    # For scalars
                    action_values = [action]

                row_data.extend(action_values)
            except Exception:
                # Fallback if action processing fails
                if self.action_dim == 1:
                    row_data.append(str(action))
                else:
                    row_data.extend(["ERR"] * self.action_dim)

        # Add reward and termination status
        row_data.extend([reward, terminated])

        # Add observation values
        row_data.extend(flattened_obs)

        # Write data to CSV
        self.csv_writer.writerow(row_data)

        # Flush periodically to ensure data is written to disk
        # but not too frequently to maintain performance
        if self.step_count % 100 == 0:
            self.csv_file.flush()

    def close(self):
        if hasattr(self, "csv_file") and self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        return self.env.close()
