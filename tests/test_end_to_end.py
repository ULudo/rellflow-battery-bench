from pathlib import Path

import yaml

from rellflow_battery_bench.control import NoBatteryController
from rellflow_battery_bench.env_adapter import make_env, run_episode


def test_tiny_run():
    cfg_path = Path("./configs/rule_based.yml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["environment"]["episode_length"] = 24 * 3600

    env = make_env(cfg)
    ctrl = NoBatteryController()
    summary, trajectory = run_episode(env, ctrl, info_meta={})

    assert isinstance(summary, dict)
    assert len(trajectory) > 0
