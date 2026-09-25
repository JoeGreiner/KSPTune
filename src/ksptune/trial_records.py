from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import yaml

from .file_io import atomic_text_file, write_text_atomic
from .replay_results import REPLAY_METRIC_TYPES, is_number

FAILED_REPLAY_OBJECTIVE_VALUE = 1.0e6
BAD_COST = FAILED_REPLAY_OBJECTIVE_VALUE


# Trial bookkeeping surrounds the common replay metrics in CSV exports.
TRIAL_CSV_COLUMNS = (
    [
        "trial_number",
        "smac_configuration_tag",
        "objective_name",
        "objective_value",
        "failure_reason",
    ]
    + [
        name
        for name in REPLAY_METRIC_TYPES
        if name
        not in {
            "matrix_cache_hits",
            "use_initial_guess",
            "repeat",
            "snapshots",
            "warmup",
            "objective_time_sec_median",
            "matrix_cache_misses",
        }
    ]
    + [
        "subprocess_wall_time_sec",
        "hard_timeout_sec",
        "replay_startup_timeout_sec",
        "effective_hard_timeout_sec",
        "used_startup_timeout",
        "max_true_residual_norm",
        "max_true_relative_residual",
        "fast_fail",
        "fast_fail_ksp_max_it",
        "petsc_options_text",
        "solver_configuration_json",
        "replay_mode",
        "stdout_path",
        "stderr_path",
        "replay_result_path",
        "returncode",
        "replay_command_text",
        "replay_server_command_text",
    ]
)


def write_yaml_file(path: str | Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, yaml.safe_dump(data, sort_keys=False))


def append_json_line(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def load_trial_records(path: str | Path, *, repair: bool = False) -> list[dict[str, Any]]:
    trial_path = Path(path)
    if not trial_path.exists():
        return []
    records: list[dict[str, Any]] = []
    lines = trial_path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip():
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError(f"Invalid trial record in {trial_path}, line {index + 1}") from None
                warnings.warn(f"Ignoring incomplete final trial record in {trial_path}", RuntimeWarning)
                lines = lines[:index]
                break
            records.append(add_trial_metrics(record))
    if repair and lines and not lines[-1].endswith(b"\n"):
        lines[-1] += b"\n"
    if repair:
        write_text_atomic(trial_path, b"".join(lines).decode("utf-8"))
    return records


def max_trial_number(records: list[dict[str, Any]]) -> int:
    trial_numbers = [
        int(record["trial_number"])
        for record in records
        if isinstance(record.get("trial_number"), int)
        or str(record.get("trial_number", "")).isdigit()
    ]
    return max(trial_numbers) if trial_numbers else 0


def is_successful_trial(record: dict[str, Any]) -> bool:
    objective_value = record.get("objective_value")
    return (
        record.get("failure_reason") in (None, "")
        and is_number(objective_value)
        and float(objective_value) < BAD_COST
    )


def ranking_key(record: dict[str, Any]) -> tuple[int, float, int]:
    objective_value = record.get("objective_value")
    numeric_objective = float(objective_value) if is_number(objective_value) else BAD_COST
    failed = 0 if is_successful_trial(record) else 1
    return (failed, numeric_objective, int(record.get("trial_number") or 0))


def best_record_from_trial_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful_records = [record for record in records if is_successful_trial(record)]
    return min(successful_records, key=ranking_key) if successful_records else None


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, str):
        return value.replace("\x00", "\\0")
    return "" if value is None else value


def trial_record_to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {column: csv_value(record.get(column)) for column in TRIAL_CSV_COLUMNS}
    solver_configuration = record.get("solver_configuration") or {}
    row["ksp_type"] = solver_configuration.get("ksp_type", "")
    row["pc_type"] = solver_configuration.get("pc_type", "")
    petsc_options = record.get("petsc_options", [])
    row["petsc_options_text"] = (
        " ".join(str(value) for value in petsc_options)
        if petsc_options
        else record.get("petsc_options_text", "")
    )
    row["solver_configuration_json"] = json.dumps(
        solver_configuration,
        sort_keys=True,
    )
    return row


def write_trial_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_file(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(trial_record_to_csv_row(record))


def compact_trial_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    names = (
        *REPLAY_METRIC_TYPES,
        "trial_number",
        "smac_configuration_tag",
        "objective_value",
        "max_true_relative_residual",
        "max_true_residual_norm",
        "fast_fail",
        "fast_fail_ksp_max_it",
        "solver_configuration",
        "petsc_options",
    )
    return {
        **{name: record.get(name) for name in names},
        "petsc_options_text": " ".join(str(value) for value in record.get("petsc_options", [])),
    }


def build_tuning_summary(
    *,
    records: list[dict[str, Any]],
    status: str,
    output_directory: str | Path,
    objective_name: str,
) -> dict[str, Any]:
    successful_records = [record for record in records if is_successful_trial(record)]
    failed_records = [record for record in records if not is_successful_trial(record)]
    converged_records = [record for record in successful_records if record.get("converged") is True]
    best_record = min(successful_records, key=ranking_key) if successful_records else None
    fastest_converged_record = None
    converged_with_wall_time = [
        record for record in converged_records if is_number(record.get("total_wall_time_sec"))
    ]
    if converged_with_wall_time:
        fastest_converged_record = min(
            converged_with_wall_time,
            key=lambda record: float(record["total_wall_time_sec"]),
        )

    rss_sum_peak_values = [
        float(record["rss_request_peak_sample_mb_sum"])
        for record in records
        if is_number(record.get("rss_request_peak_sample_mb_sum"))
    ]
    rss_rank_peak_values = [
        float(record["rss_peak_sample_mb_max_rank"])
        for record in records
        if is_number(record.get("rss_peak_sample_mb_max_rank"))
    ]
    rss_node_peak_values = [
        float(record["rss_peak_sample_mb_max_node"])
        for record in records
        if is_number(record.get("rss_peak_sample_mb_max_node"))
    ]

    return {
        "status": status,
        "output_directory": str(Path(output_directory).resolve()),
        "objective_name": objective_name,
        "trial_count": len(records),
        "successful_trial_count": len(successful_records),
        "failed_trial_count": len(failed_records),
        "converged_trial_count": len(converged_records),
        "best_trial": compact_trial_summary(best_record),
        "fastest_converged_trial": compact_trial_summary(fastest_converged_record),
        "rss_request_peak_sample_mb_sum_max": (
            max(rss_sum_peak_values) if rss_sum_peak_values else None
        ),
        "rss_peak_sample_mb_max_rank": (
            max(rss_rank_peak_values) if rss_rank_peak_values else None
        ),
        "rss_peak_sample_mb_max_node": (
            max(rss_node_peak_values) if rss_node_peak_values else None
        ),
    }


def write_tuning_summary_outputs(
    *,
    output_directory: str | Path,
    objective_name: str,
    status: str,
) -> dict[str, Any]:
    output_path = Path(output_directory)
    trial_records_path = output_path / "solver_configuration_trials.jsonl"
    trial_records_path.parent.mkdir(parents=True, exist_ok=True)
    trial_records_path.touch(exist_ok=True)
    records = load_trial_records(trial_records_path)
    ranked_records = sorted(records, key=ranking_key)
    write_trial_csv(output_path / "solver_configuration_trials.csv", records)
    write_trial_csv(output_path / "solver_configuration_rankings.csv", ranked_records)
    summary = build_tuning_summary(
        records=records,
        status=status,
        output_directory=output_path,
        objective_name=objective_name,
    )
    summary.update(tuning_output_paths(output_path))
    write_yaml_file(output_path / "tuning_summary.yaml", summary)
    return summary


def tuning_output_paths(output_directory: str | Path) -> dict[str, str]:
    output_path = Path(output_directory)
    return {
        "tuning_summary_path": str((output_path / "tuning_summary.yaml").resolve()),
        "trial_records_path": str((output_path / "solver_configuration_trials.jsonl").resolve()),
        "trial_csv_path": str((output_path / "solver_configuration_trials.csv").resolve()),
        "ranking_csv_path": str((output_path / "solver_configuration_rankings.csv").resolve()),
        "best_solver_configuration_path": str(
            (output_path / "best_solver_configuration.yaml").resolve()
        ),
        "best_petsc_options_path": str((output_path / "best_petsc_options.txt").resolve()),
    }


def add_trial_metrics(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    phases = [
        enriched.get(name)
        for name in (
            "matrix_load_time_sec", "nullspace_time_sec", "true_residual_time_sec",
            "solver_setup_time_sec_actual", "solve_time_sec_total",
        )
    ]
    replay_wall = enriched.get("total_wall_time_sec")
    request_wall = enriched.get("subprocess_wall_time_sec")
    if all(is_number(value) for value in phases) and is_number(replay_wall):
        enriched["replay_other_time_sec"] = replay_wall - sum(phases)
    if is_number(request_wall) and is_number(replay_wall):
        enriched["outer_overhead_time_sec"] = request_wall - replay_wall
    iterations, count = enriched.get("iterations_total"), enriched.get("solve_count")
    if is_number(iterations) and is_number(count) and count > 0:
        enriched["iterations_mean"] = iterations / count
    return enriched
