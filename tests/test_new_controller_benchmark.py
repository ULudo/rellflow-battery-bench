from pathlib import Path

from rellflow_battery_bench.runner import main as cli_main


def test_user_controller_benchmark(tmp_path, monkeypatch):
    # Use rule_based config (already points to benchmark datasets)
    out_dir = tmp_path / "eval"
    args = [
        "bench",
        "--config",
        "./configs/rule_based.yml",
        "--user-controller",
        "scripts.test_controller:PVGreedyController",
        "--out",
        str(out_dir),
    ]

    # Run CLI entrypoint
    cli_main(args)

    # Check HTML index and comparison plots
    assert (out_dir / "index.html").exists()
    for name in ["summer", "autumn", "spring"]:
        assert (out_dir / f"comparison_{name}.png").exists()
        # Check per-dataset baseline outputs exist
        for ctrl in ["no_battery", "rule_based", "user"]:
            ds_dir = out_dir / name / ctrl
            assert (ds_dir / "metrics.csv").exists()
            assert (ds_dir / "trajectory.csv").exists()
            assert (ds_dir / "report.html").exists()


