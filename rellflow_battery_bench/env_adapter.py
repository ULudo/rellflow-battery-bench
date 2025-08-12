from typing import Any, Dict, Tuple, List

import gymnasium as gym

from .env import BuildingEnv, BuildingDataManager
from .env.consts_and_types import FormatType
from .control.controller import ControllerInfo


def make_env(cfg: Dict[str, Any]) -> gym.Env:
    data_cfg = cfg.get("data", {})
    datasets_list: List[Dict[str, Any]] = data_cfg.get("datasets", [])
    datasets = {d["name"]: d["house_data_file"] for d in datasets_list}
    price_data_file = data_cfg.get("price_data_file")
    data_format = data_cfg.get("data_format", "csv").lower()
    fmt = FormatType.CSV if data_format == "csv" else FormatType.FEATHER

    BuildingDataManager.reset_datasets()
    BuildingDataManager.load_datasets(datasets, price_data_file, fmt)

    env_cfg = cfg.get("environment", {})
    env = BuildingEnv(
        dataset_args=env_cfg.get("dataset_args"),
        battery_efficiency=env_cfg.get("battery", {}).get("efficiency", 0.95),
        battery_max_power=env_cfg.get("battery", {}).get("max_power", 8000),
        battery_capacity=env_cfg.get("battery", {}).get("capacity", 20000),
        use_time_features=True,
        random_time_init=env_cfg.get("random_time_init", False),
        random_soc_init=False,
        upper_time_bound=None,
        init_soc=0.5,
        init_time=env_cfg.get("init_time"),
        episode_length=env_cfg.get("episode_length"),
        tax=env_cfg.get("tax", 0.0),
        apply_deadband=False,
        scaling_method=env_cfg.get("scaling_method", "none"),
        load_stats=None,
        price_stats=None,
        pv_stats=None,
        prediction_horizon=0,
        action_space_type=env_cfg.get("action_space_type", "discrete"),
    )
    return env


def run_episode(env: gym.Env, controller, info_meta: Dict[str, Any]) -> Tuple[Dict, list[Dict]]:
    obs, info = env.reset()
    # Construct initial ControllerInfo dataclass
    ctrl_info = ControllerInfo(
        config={},
        metadata={**info_meta, "clean_obs": info.get("clean_obs"), "episode": info.get("episode")},
    )
    if hasattr(controller, "reset"):
        controller.reset(obs, ctrl_info)  # type: ignore[arg-type]
    logs: list[Dict] = []
    done = False
    total_reward = 0.0
    while not done:
        action = controller.act(obs, ctrl_info)  # type: ignore[arg-type]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        row = dict(info)
        row["reward"] = float(reward)
        logs.append(row)
        # Update ctrl_info for next decision
        ctrl_info = ControllerInfo(
            config=ctrl_info.config,
            metadata={**ctrl_info.metadata, "clean_obs": info.get("clean_obs"), "episode": info.get("episode")},
        )
    summary = {"total_reward": total_reward}
    return summary, logs


