from typing import Any

import numpy as np

from rellflow_battery_bench.control import ControllerInterface, ControllerInfo


class PVGreedyController(ControllerInterface):
    """Charges when PV exceeds load, otherwise idle; simple demo controller."""

    def reset(self, obs: Any, info: ControllerInfo) -> None:  # noqa: ANN401
        pass

    def act(self, obs: Any, info: ControllerInfo):  # noqa: ANN401
        # Prefer metadata if provided
        meta = info.metadata or {}
        clean = meta.get("clean_obs") if isinstance(meta, dict) else None
        if isinstance(clean, dict):
            load = float(clean.get("loads", [0.0])[0])
            pv = float(clean.get("gens", [0.0])[0])
        else:
            arr = np.array(obs).reshape(-1)
            # Heuristic: SOC, E_bat, load0, price0, pv0, ...
            load = float(arr[2]) if arr.size >= 3 else 0.0
            pv = float(arr[4]) if arr.size >= 5 else 0.0
        if pv > load:
            return 1  # charge
        return 0  # idle


