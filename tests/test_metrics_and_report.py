from rellflow_battery_bench.metrics import summarize_rewards, final_soc
from rellflow_battery_bench.report import generate_all


def test_metrics_and_report(tmp_path):
    trajectory = [
        {"reward": 1.0, "clean_obs": {"soc": [0.5]}},
        {"reward": -0.5, "clean_obs": {"soc": [0.6]}},
    ]
    summary = summarize_rewards(trajectory)
    assert summary["total_reward"] == 0.5
    assert summary["num_steps"] == 2
    assert final_soc(trajectory) == 0.6

    cfg = {"name": "test"}
    out = generate_all(str(tmp_path), cfg, trajectory)
    assert (tmp_path / "trajectory.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "report.html").exists()
