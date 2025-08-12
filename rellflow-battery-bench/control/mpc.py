from .controller import ControllerInterface

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model

from ..env.consts_and_types import DATA_FREQUENCY
from ..env.functions import s_to_hour


class MPCOptimizer:

    def __init__(
        self,
        n_predictions: int,
        bat_efficiency: float,
        bat_capacity: float,  # in Wh
        bat_max_power: float,  # in W
        tax: float = 0.0,
    ) -> None:
        self.dt = s_to_hour(DATA_FREQUENCY)
        self.n_predictions = n_predictions
        self.bat_efficiency = bat_efficiency
        self.bat_capacity = bat_capacity / 1e3  # Wh → kWh
        self.bat_max_power = bat_max_power / 1e3  #  W → kW
        self.tax = tax
        self.n_threads = 0  # 0 implies no limit

    def set_n_threads(self, n_threads: int) -> None:
        """
        Set the number of threads for the Gurobi solver.
        :param n_threads: Number of threads to use for Gurobi solver.
        """
        if n_threads < 0:
            raise ValueError("Number of threads must be non-negative.")
        self.n_threads = n_threads

    def _get_adaptive_horizon(self, episode_info: Optional[Dict]) -> int:
        """
        Calculate adaptive prediction horizon based on remaining episode steps from environment.
        Gradually reduces horizon towards episode end.

        Args:
            episode_info: Episode information from environment containing:
                - remaining_steps: Steps remaining in episode
                - total_steps: Total steps in episode (optional, for fallback)

        Returns:
            int: Adjusted prediction horizon (between 1 and n_predictions)
        """
        if episode_info is None:
            return self.n_predictions
        remaining_steps = episode_info.get("remaining_steps", self.n_predictions)
        return min(self.n_predictions, max(1, remaining_steps))

    def optimize(
        self,
        soc_initial: float,
        load_ts: np.ndarray,  # in kWh
        price_ts: np.ndarray,  # in ct/kWh
        pv_ts: np.ndarray,  # in kWh
        episode_info: Optional[Dict] = None,
        ret_full: bool = False,
    ) -> Union[
        Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        adaptive_horizon = self._get_adaptive_horizon(episode_info)
        time_steps = adaptive_horizon + 1  # Include current time step

        # Create a linopy model
        m = Model()

        # Define time indices
        time = pd.Index(range(time_steps), name="time")
        # Defined for time steps t = 0 to N (total N+1 steps) because you need to know the SOC after the last action at time t = N-1.
        # SOC[0] -> p_charge/p_discharge[0] -> SOC[1] -> p_charge/p_discharge[1] -> ... -> SOC[N-1] -> p_charge/p_discharge[N-1] -> SOC[N]
        time_soc = pd.Index(range(time_steps + 1), name="time")
        # Define coords for variables
        coords = {"time": time}
        coords_soc = {"time": time_soc}
        # Define variables
        soc = m.add_variables(
            name="soc", dims=["time"], coords=coords_soc, lower=0, upper=1
        )
        p_charge = m.add_variables(
            name="p_charge",
            dims=["time"],
            coords=coords,
            lower=0,
            upper=self.bat_max_power,
        )  # in kW
        p_discharge = m.add_variables(
            name="p_discharge",
            dims=["time"],
            coords=coords,
            lower=0,
            upper=self.bat_max_power,
        )  # in kW

        # Convert prediction arrays to xarrays with time coordinate for the use with linopy
        load_da = xr.DataArray(
            load_ts[:time_steps], dims=["time"], coords=coords
        )  # in kWh
        price_da = xr.DataArray(
            price_ts[:time_steps], dims=["time"], coords=coords
        )  # in ct/kWh
        pv_da = xr.DataArray(pv_ts[:time_steps], dims=["time"], coords=coords)  # in kWh

        # Convert p_charge and p_discharge
        p_charge_kwh = p_charge * self.dt
        p_discharge_kwh = p_discharge * self.dt

        # Compute net power (kWh)
        net_power = p_charge_kwh - p_discharge_kwh + load_da - pv_da

        # Add variables for net import and net export
        net_import = m.add_variables(
            name="net_import", dims=["time"], coords=coords, lower=0
        )
        net_export = m.add_variables(
            name="net_export", dims=["time"], coords=coords, lower=0
        )

        # Price for importing electricity
        price_import = price_da + self.tax

        # Constraints
        # Initial SoC constraint
        # This constraint sets the initial SoC of the battery at the beginning of the prediction horizon to match the actual current SoC.
        # Ensures that the optimization model starts from the correct SoC, reflecting the real state of the battery.
        m.add_constraints(soc.sel(time=0) - soc_initial == 0, name="initial_soc")
        # SoC dynamics constraint
        # This constraint models the battery's SoC dynamics over the prediction horizon.
        # It enforces the relationship between the SoC at consecutive time steps based on the charging and discharging actions.
        soc_diff = (
            soc.sel(time=time + 1)
            - soc.sel(time=time)
            - (
                (
                    p_charge * self.dt * self.bat_efficiency
                    - p_discharge * self.dt / self.bat_efficiency
                )
                / self.bat_capacity
            )
        )
        m.add_constraints(soc_diff == 0, name="soc_dynamics")
        # Net power balance constraint
        # The constraint is essential to correctly model the net import and export of electricity to and
        # from the grid and to apply the tax only when importing electricity.
        m.add_constraints(
            net_power == net_import - net_export, name="net_power_balance"
        )

        # Objective: Minimize total cost
        cost = (price_import * net_import - price_da * net_export).sum(dim="time")
        m.objective = cost

        # Solve the optimization problem
        result = m.solve(
            solver_name="gurobi", OutputFlag=0, LogToConsole=0, Threads=self.n_threads
        )

        # Check if the optimization was successful
        if result != ("ok", "optimal"):
            raise RuntimeError(f"Optimization failed with status: {result}")

        if ret_full:
            p_charge_sol = p_charge.solution.values  # (T,)
            p_discharge_sol = p_discharge.solution.values
            net_import_sol = net_import.solution.values
            net_export_sol = net_export.solution.values
            return p_charge_sol, p_discharge_sol, net_import_sol, net_export_sol
        else:
            # Extract the optimal p_charge and p_discharge at time=0
            p_charge_opt = p_charge.solution.sel(time=0).item()
            p_discharge_opt = p_discharge.solution.sel(time=0).item()
            print(f"p_charge: {p_charge_opt}, p_discharge: {p_discharge_opt}")
            return p_charge_opt, p_discharge_opt

    def get_action(self, p_charge_opt, p_discharge_opt):
        # Compute net power (positive for charging, negative for discharging)
        p_net = p_charge_opt - p_discharge_opt  # in kW

        # Map net power to action space [-1, 1] by dividing by max power (kW)
        action = np.clip(p_net / self.bat_max_power, -1.0, 1.0)

        return action



class PerfectMPController(ControllerInterface):
    def __init__(self, optimizer: MPCOptimizer) -> None:
        self.optimizer = optimizer

    def step(self, obs: np.ndarray, info: Dict) -> int | float | np.ndarray:
        opti_params = self._get_params_from_info(info)
        # mypy: optimize may return 2-tuple or 4-tuple; here we request the short form
        result = self.optimizer.optimize(*opti_params)
        p_charge_opt, p_discharge_opt = cast(Tuple[float, float], result)
        return self.optimizer.get_action(p_charge_opt, p_discharge_opt)

    def _get_params_from_info(self, info: Dict) -> Any:
        obs_dict = info["clean_obs"]
        soc = obs_dict["soc"][0]
        opti_params = [soc] + [obs_dict[name] for name in ["loads", "prices", "gens"]]
        opti_params.append(info.get("episode", {}))
        return opti_params