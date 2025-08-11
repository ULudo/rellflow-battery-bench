from typing import Dict, List

import numpy as np


def summarize_rewards(trajectory: List[Dict]) -> Dict[str, float]:
    rewards = np.array([row.get("reward", 0.0) for row in trajectory], dtype=float)
    return {
        "total_reward": float(rewards.sum()),
        "mean_reward": float(rewards.mean() if rewards.size else 0.0),
        "std_reward": float(rewards.std() if rewards.size else 0.0),
        "num_steps": int(rewards.size),
    }


def final_soc(trajectory: List[Dict]) -> float:
    if not trajectory:
        return 0.0
    last = trajectory[-1]
    clean_obs = last.get("clean_obs", {})
    if isinstance(clean_obs, dict):
        arr = clean_obs.get("soc", [0.0])
        try:
            return float(arr[0])
        except Exception:
            return 0.0
    return 0.0
