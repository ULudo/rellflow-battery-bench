from pathlib import Path

import yaml

from rellflow-battery-bench.baselines import NoBatteryController
from rellflow-battery-bench.env_adapter import make_env, run_episode


def test_tiny_run():
    cfg_path = Path("./res/configs/base_controller.yml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["environment"]["episode_length"] = 24 * 3600

    env = make_env(cfg)
    ctrl = NoBatteryController()
    summary, trajectory = run_episode(env, ctrl, info_meta={})

    assert isinstance(summary, dict)
    assert len(trajectory) > 0
