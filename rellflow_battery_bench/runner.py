import sys
from pathlib import Path
import importlib
import json
from argparse import ArgumentParser
from pathlib import Path as _Path
from typing import Any, Dict

import yaml

from .env_adapter import make_env, run_episode
from .report import generate_all
from .control import (
    NoBatteryController,
    SimpleRuleBasedController,
    MPCOptimizer,
    PerfectMPController,
)
from .plotting import plot_reward_series


def load_controller(dotted_path: str, kwargs: Dict[str, Any] | None = None):
    if ":" in dotted_path:
        mod_name, cls_name = dotted_path.split(":", 1)
    else:
        parts = dotted_path.split(".")
        mod_name, cls_name = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    return cls(**(kwargs or {}))


def run(cfg: Dict[str, Any], controller_path: str, controller_kwargs: Dict[str, Any] | None, out_dir: str) -> Dict[str, Any]:
    env = make_env(cfg)
    controller = load_controller(controller_path, controller_kwargs)
    summary, trajectory = run_episode(env, controller, info_meta={})
    report = generate_all(out_dir, cfg, trajectory)
    result = {"summary": report["summary"], "html": report["html"], "out_dir": out_dir}
    return result


def main(argv: list[str] | None = None) -> None:
    p = ArgumentParser(prog="battery-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a single benchmark experiment")
    run_p.add_argument("--config", required=True, help="YAML config path")
    run_p.add_argument("--controller", required=True, help="Controller class path, e.g., pkg.mod:Class")
    run_p.add_argument("--controller-kwargs", default="{}", help="JSON string of controller kwargs")
    run_p.add_argument("--out", required=True, help="Output directory under reports/")

    bench_p = sub.add_parser("bench", help="Benchmark baselines and optional user controller across datasets in config")
    bench_p.add_argument("--config", required=True, help="YAML config path (must list datasets)")
    bench_p.add_argument("--user-controller", default=None, help="Optional user Controller class path, e.g., pkg.mod:Class")
    bench_p.add_argument("--user-controller-kwargs", default="{}", help="JSON string of user controller kwargs")
    bench_p.add_argument("--out", required=True, help="Output directory under reports/ (e.g., reports/eval)")

    args = p.parse_args(argv)

    if args.cmd == "run":
        cfg_path = _Path(args.config)
        cfg = yaml.safe_load(cfg_path.read_text())
        out_dir = str(_Path(args.out))
        ctrl_kwargs = json.loads(args.controller_kwargs)
        result = run(cfg, args.controller, ctrl_kwargs, out_dir)
        print(json.dumps(result, indent=2))
    elif args.cmd == "bench":
        cfg_path = _Path(args.config)
        cfg = yaml.safe_load(cfg_path.read_text())
        out_root = Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)

        # Extract dataset names from config data.datasets
        ds = cfg.get("data", {}).get("datasets", [])
        dataset_names = [d.get("name") for d in ds]
        dataset_map = {d.get("name"): d.get("house_data_file") for d in ds}

        # Prepare controllers: baselines and optional user
        controllers: list[tuple[str, Any]] = [
            ("no_battery", NoBatteryController()),
            ("rule_based", SimpleRuleBasedController()),
        ]
        # Try to add MPC perfect if solver available; fallback skip on error when running
        try:
            mpc_opt = MPCOptimizer(
                n_predictions=96,
                bat_efficiency=cfg.get("environment", {}).get("battery", {}).get("efficiency", 0.95),
                bat_capacity=cfg.get("environment", {}).get("battery", {}).get("capacity", 20000),
                bat_max_power=cfg.get("environment", {}).get("battery", {}).get("max_power", 8000),
                tax=cfg.get("environment", {}).get("tax", 0.0),
            )
            mpc_ctrl = PerfectMPController(mpc_opt)
            controllers.append(("mpc_perfect", mpc_ctrl))
        except Exception:
            pass

        if args.user_controller:
            user_kwargs = json.loads(args.user_controller_kwargs)
            user_ctrl = load_controller(args.user_controller, user_kwargs)
            controllers.append(("user", user_ctrl))

        # Evaluate per dataset, per controller
        index_rows = []
        for dataset_name in dataset_names:
            # Update env dataset selection
            cfg_ds = json.loads(json.dumps(cfg))  # deep copy via JSON
            env_cfg = cfg_ds.setdefault("environment", {})
            env_cfg["dataset_args"] = {"building": dataset_name, "price": dataset_name}
            # Make env once per dataset
            env = make_env(cfg_ds)

            # per-dataset aggregation for comparison plot
            comparison_curves = {}

            for ctrl_name, controller in controllers:
                out_dir = out_root / dataset_name / ctrl_name
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    summary, trajectory = run_episode(env, controller, info_meta={"action_space": cfg_ds.get("environment", {}).get("action_space_type", "continuous")})
                except Exception as e:
                    # Skip controller on failure (e.g., missing solver)
                    (out_dir / "ERROR.txt").write_text(str(e))
                    continue
                # Save report
                rep = generate_all(str(out_dir), cfg_ds | {"controller": ctrl_name}, trajectory)
                # Save rewards plot
                plot_reward_series(trajectory, out_png=str(out_dir / "rewards.png"))
                comparison_curves[ctrl_name] = trajectory
                index_rows.append({
                    "dataset": dataset_name,
                    "controller": ctrl_name,
                    **rep["summary"],
                    "report": str(out_dir / "report.html"),
                })

            # Per-dataset comparison plot
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots(figsize=(10, 4))
            for ctrl_name, traj in comparison_curves.items():
                rewards = np.array([row.get("reward", 0.0) for row in traj], dtype=float)
                ax.plot(rewards, label=ctrl_name)
            ax.set_title(f"Reward per step — {dataset_name}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_root / f"comparison_{dataset_name}.png", dpi=150)

        # Compute improvements vs no_battery
        # Reshape index_rows into dict by dataset -> controller -> summary
        by_ds: dict[str, dict[str, dict]] = {}
        for r in index_rows:
            by_ds.setdefault(r["dataset"], {})[r["controller"]] = r

        # Write simple HTML index
        index_html = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'><title>Battery Bench Evaluation</title>",
            "<style>body{font-family:Inter,Arial,sans-serif;margin:2rem}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px}</style>",
            "</head><body>",
            "<h1>Battery Bench Evaluation</h1>",
        ]
        for dataset_name in dataset_names:
            img = f"comparison_{dataset_name}.png"
            index_html.append(f"<h2>Dataset: {dataset_name}</h2>")
            if (out_root / img).exists():
                index_html.append(f"<img src='{img}' alt='{dataset_name}' style='max-width:100%'>")
            # table
            rows = [r for r in index_rows if r["dataset"] == dataset_name]
            if rows:
                index_html.append("<table><tr><th>Controller</th><th>Total</th><th>Mean</th><th>Std</th><th>Steps</th><th>Improvement vs no_controller</th><th>Report</th></tr>")
                base = by_ds.get(dataset_name, {}).get("no_battery")
                for r in rows:
                    improvement = "-"
                    if base and base.get("total_reward") not in (0, None) and r["controller"] != "no_battery":
                        try:
                            base_val = float(base["total_reward"])
                            current_val = float(r["total_reward"])
                            # higher is better (less negative cost), percentage vs absolute baseline magnitude
                            denom = abs(base_val) if abs(base_val) > 1e-9 else 1.0
                            improvement_val = 100.0 * (current_val - base_val) / denom
                            improvement = f"{improvement_val:+.1f}%"
                        except Exception:
                            improvement = "-"
                    index_html.append(
                        f"<tr><td>{r['controller']}</td><td>{r['total_reward']:.3f}</td>"
                        f"<td>{r['mean_reward']:.3f}</td><td>{r['std_reward']:.3f}</td><td>{r['num_steps']}</td>"
                        f"<td>{improvement}</td><td><a href='{Path(r['report']).relative_to(out_root)}'>report</a></td></tr>"
                    )
                index_html.append("</table>")
        index_html.append("</body></html>")
        (out_root / "index.html").write_text("\n".join(index_html))
        print(json.dumps({"out": str(out_root), "index": str(out_root / 'index.html')}, indent=2))


if __name__ == "__main__":
    main()


