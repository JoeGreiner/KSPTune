from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_trial_records(tuning_run_directory: str | Path) -> list[dict[str, Any]]:
    path = Path(tuning_run_directory) / "solver_configuration_trials.jsonl"
    if not path.exists():
        raise ValueError(f"No solver_configuration_trials.jsonl found in {tuning_run_directory}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def analyze_parameter_importance(
    tuning_run_directory: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    records = load_trial_records(tuning_run_directory)
    if len(records) < 2:
        result = {
            "method": "insufficient-data",
            "trial_count": len(records),
            "parameter_importance": {},
        }
    else:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor

        solver_configurations = [record["solver_configuration"] for record in records]
        objectives = [record["objective_value"] for record in records]
        table = pd.get_dummies(pd.DataFrame(solver_configurations), dummy_na=True)
        model = RandomForestRegressor(n_estimators=128, random_state=1)
        model.fit(table, objectives)
        importances = {
            name: float(value)
            for name, value in sorted(
                zip(table.columns, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        result = {
            "method": "random-forest-feature-importance",
            "trial_count": len(records),
            "parameter_importance": importances,
            "note": "fANOVA can be added on top of the same trial file once runhistory export is stable.",
        }

    if output_path is None:
        output_path = Path(tuning_run_directory) / "parameter_importance_summary.yaml"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return result

