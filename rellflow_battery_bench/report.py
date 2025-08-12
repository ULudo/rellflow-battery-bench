from pathlib import Path
from typing import Any, Dict, List
import csv

from .metrics import summarize_rewards


def write_csvs(out_dir: str, trajectory: List[Dict], summary: Dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # trajectory.csv
    if trajectory:
        fieldnames = sorted({k for row in trajectory for k in row.keys()})
    else:
        fieldnames = []
    with (out / "trajectory.csv").open("w", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in trajectory:
                writer.writerow(row)
        else:
            f.write("")

    # metrics.csv
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def render_html_report(out_dir: str, cfg: Dict[str, Any], trajectory: List[Dict], summary: Dict[str, Any]) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Battery Bench Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    h1 {{ margin-bottom: 0.2rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 520px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    pre {{ background: #f9f9f9; padding: 8px; border: 1px solid #eee; }}
  </style>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap\" rel=\"stylesheet\">
</head>
<body style=\"font-family: Inter, Arial, sans-serif;\">
  <h1>Battery Bench Report</h1>
  <p><strong>Config name:</strong> {cfg.get('name', 'run')}</p>
  <h2>Metrics</h2>
  <table>
    <tr><th>Total Reward</th><td>{summary.get('total_reward', 0.0):.4f}</td></tr>
    <tr><th>Mean Reward</th><td>{summary.get('mean_reward', 0.0):.4f}</td></tr>
    <tr><th>Std Reward</th><td>{summary.get('std_reward', 0.0):.4f}</td></tr>
    <tr><th>Steps</th><td>{summary.get('num_steps', 0)}</td></tr>
  </table>
  <h2>Config</h2>
  <pre>{cfg}</pre>
</body>
</html>
"""
    out_file = out / "report.html"
    out_file.write_text(html)
    return str(out_file)


def generate_all(out_dir: str, cfg: Dict[str, Any], trajectory: List[Dict]) -> Dict[str, Any]:
    summary = summarize_rewards(trajectory)
    write_csvs(out_dir, trajectory, summary)
    html_path = render_html_report(out_dir, cfg, trajectory, summary)
    return {"summary": summary, "html": html_path}


