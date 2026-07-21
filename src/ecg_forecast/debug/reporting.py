import os
import json
import csv
from typing import Dict, Any, List, Optional
import yaml


def save_debug_artifacts(
    output_dir: str,
    summary_data: Dict[str, Any],
    per_sample_rows: Optional[List[Dict[str, Any]]] = None,
    per_lead_rows: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Any] = None,
) -> str:
    """Saves standardized debug output files in artifacts/debug/<script_name>/<checkpoint_name>/."""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Save summary.json
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # 2. Save per_sample_metrics.csv
    if per_sample_rows and len(per_sample_rows) > 0:
        csv_path = os.path.join(output_dir, "per_sample_metrics.csv")
        headers = list(per_sample_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(per_sample_rows)

    # 3. Save per_lead_metrics.csv
    if per_lead_rows and len(per_lead_rows) > 0:
        csv_path = os.path.join(output_dir, "per_lead_metrics.csv")
        headers = list(per_lead_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(per_lead_rows)

    # 4. Save config.yaml if provided
    if config is not None:
        cfg_path = os.path.join(output_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            if hasattr(config, "__dict__"):
                yaml.dump(config.__dict__, f, default_flow_style=False)
            elif isinstance(config, dict):
                yaml.dump(config, f, default_flow_style=False)
            else:
                f.write(str(config))

    return output_dir
