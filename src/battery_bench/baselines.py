from typing import Any

import numpy as np

from .controller import ControllerInterface, ControllerInfo


class NoBatteryController(ControllerInterface):
    def reset(self, obs: Any, info: ControllerInfo) -> None:  # noqa: ANN401
        return None

    def act(self, obs: Any, info: ControllerInfo):  # noqa: ANN401
        return 0


class SimpleRuleBasedController(ControllerInterface):
    def __init__(self, window: int = 24, low_q: float = 0.4, high_q: float = 0.6):
        self.window = window
        self.low_q = low_q
        self.high_q = high_q
        self.price_hist: list[float] = []

    def reset(self, obs: Any, info: ControllerInfo) -> None:  # noqa: ANN401
        self.price_hist.clear()

    def act(self, obs: Any, info: ControllerInfo):  # noqa: ANN401
        obs_arr = np.array(obs).reshape(-1)
        meta = info.metadata or {}
        clean_obs = meta.get("clean_obs") if isinstance(meta, dict) else None
        if isinstance(clean_obs, dict):
            load = float(clean_obs.get("loads", [0.0])[0])
            price = float(clean_obs.get("prices", [0.0])[0])
            pv = float(clean_obs.get("gens", [0.0])[0])
        else:
            load, price, pv = float(obs_arr[2]), float(obs_arr[3]), float(obs_arr[4])

        self.price_hist.append(price)
        if len(self.price_hist) > self.window:
            self.price_hist.pop(0)

        if pv > load:
            return 1
        if len(self.price_hist) >= max(3, int(0.5 * self.window)):
            low_thr = float(np.quantile(self.price_hist, self.low_q))
            high_thr = float(np.quantile(self.price_hist, self.high_q))
            if price <= low_thr:
                return 1
            if price >= high_thr:
                return 2
        return 0
