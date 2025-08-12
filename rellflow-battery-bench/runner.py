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

    args = p.parse_args(argv)

    if args.cmd == "run":
        cfg_path = _Path(args.config)
        cfg = yaml.safe_load(cfg_path.read_text())
        out_dir = str(_Path(args.out))
        ctrl_kwargs = json.loads(args.controller_kwargs)
        result = run(cfg, args.controller, ctrl_kwargs, out_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
