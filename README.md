# ReLLFloW Battery Bench

Benchmarking app for battery control algorithms (rule-based, MPC, RL) against the a building energy environment.

Features
- Simple controller interface (`battery_bench.controller.Interface`)
- Runner to execute experiments across datasets
- Built-in baselines: no battery, simple rule-based
- Metrics and plotting utilities
- Reproducible HTML and CSV reports under `reports/`

## Install
```bash
pip install -e .
```

## Quickstart
```bash
battery-bench run \
  --config res/configs/base_controller.yml \
  --controller battery_bench.baselines:NoBatteryController \
  --out reports/example
```

## Controller Interface
Implement `battery_bench.controller.Interface`:
```python
from battery_bench.controller import ControllerInterface, 
ControllerInfo

class MyController(ControllerInterface):
    def reset(self, obs, info: ControllerInfo):
        ...
    def act(self, obs, info: ControllerInfo):
        return 0  # discrete or float depending on env action space
```
Load dynamically via `--controller mypkg.mymod:MyController`.

Reports
The runner writes `metrics.csv`, `trajectory.csv`, and `report.html` 
under the given output directory.

## Test Data

Following months were selected for battery benchmarking:

### 1. Summer Period - July 2024
**Dataset ID:** `0080E1FA00236634`  
**Period:** July 2024 (with 24h buffer periods)  
**Characteristics:**
- Mean load: 0.0143 kWh/15min
- Max load: 0.2058 kWh/15min  
- Min load: 0.0032 kWh/15min
- Total consumption: 42.49 kWh
- Records: 2,976 (15-minute intervals)

![Summer Load Profile](plots/0080E1FA00236634_load_profile.png)

### 2. Autumn Period - November 2024
**Dataset ID:** `0080E1FA00236638`  
**Period:** November 2024 (with 24h buffer periods)  
**Characteristics:**
- Mean load: 0.0176 kWh/15min
- Max load: 0.3051 kWh/15min
- Min load: 0.0000 kWh/15min
- Total consumption: 50.65 kWh
- Records: 2,880 (15-minute intervals)

![Autumn Load Profile](plots/0080E1FA00236638_load_profile.png)

### 3. Spring Period - April 2025
**Dataset ID:** `0080E1FA00237198`  
**Period:** April 2025 (with 24h buffer periods)  
**Characteristics:**
- Mean load: 0.0149 kWh/15min
- Max load: 0.3596 kWh/15min
- Min load: 0.0026 kWh/15min
- Total consumption: 42.92 kWh
- Records: 2,880 (15-minute intervals)

![Spring Load Profile](plots/0080E1FA00237198_load_profile.png)

### Data Format

Each benchmark dataset is provided in CSV format with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `unixtime` | Unix timestamp | seconds |
| `load` | Energy consumption | kWh/15min |
| `pv` | Solar PV generation | kWh/15min (all zeros) |

### File Structure

```
selected_months/
├── 0080E1FA00236634.csv  # July 2024 (Summer)
├── 0080E1FA00237198.csv  # April 2025 (Spring)
└── 0080E1FA00236638.csv  # November 2024 (Autumn)
```


### Data Quality

All datasets show excellent quality:
- No missing values
- No negative values
- Minimal outliers (< 2%)
- Perfect time intervals
- High data completeness (100%)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
