# ReLLFloW Battery Bench

Benchmarking app for battery control algorithms (rule-based, MPC, RL) against the ReLLFloW building energy environment.

Features
- Simple controller interface (`battery_bench.controller.Interface`)
- Runner to execute experiments across datasets
- Built-in baselines: no battery, simple rule-based
- Metrics and plotting utilities
- Reproducible HTML and CSV reports under `reports/`
- Python 3.12+

Install
```bash
pip install -e .
```

Quickstart
```bash
battery-bench run \
  --config res/configs/base_controller.yml \
  --controller battery_bench.baselines:NoBatteryController \
  --out reports/example
```

Controller Interface
Implement `battery_bench.controller.Interface`:
```python
from battery_bench.controller import ControllerInterface, ControllerInfo

class MyController(ControllerInterface):
    def reset(self, obs, info: ControllerInfo):
        ...
    def act(self, obs, info: ControllerInfo):
        return 0  # discrete or float depending on env action space
```
Load dynamically via `--controller mypkg.mymod:MyController`.

Reports
The runner writes `metrics.csv`, `trajectory.csv`, and `report.html` under the given output directory.

Tests
```bash
pytest -q
```
