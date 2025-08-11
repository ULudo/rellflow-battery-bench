from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_reward_series(trajectory: List[Dict], out_png: str | None = None):
    rewards = np.array([row.get("reward", 0.0) for row in trajectory], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rewards, label="reward")
    ax.set_title("Reward per step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig
