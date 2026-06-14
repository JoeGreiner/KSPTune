from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

import yaml

from .parameter_search_spaces import (
    get_parameter_search_space_name,
    load_parameter_search_space,
    parameter_search_space_to_yaml,
    write_configspace_json,
    write_configspace_yaml,
)
from .nullspaces import parse_nullspace, parse_nullspace_actions
from .petsc_options import (
    petsc_options_to_text,
    render_petsc_options_from_solver_configuration,
)
from .replay import (
    ReplayWorkerPool,
    replay_failure_reason,
    replay_metric_summary,
    replay_objective_value,
    run_replay_server_for_solver_configuration,
    run_replay_for_solver_configuration,
)
from .snapshot_collections import (
    create_snapshot_collection_from_directory,
    summarize_snapshot_directory,
    write_resolved_snapshot_collection,
)
from .solver_configurations import (
    default_solver_configuration_from_parameter_search_space,
    solver_configuration_from_configspace_configuration,
)

FAILED_REPLAY_OBJECTIVE_VALUE = 1.0e6
BAD_COST = FAILED_REPLAY_OBJECTIVE_VALUE
RUN_UNTIL_STOPPED_TRIAL_LIMIT = 2_147_483_647
REPLAY_BINARY_NAME = "ksptune-petsc-replay"

TRIAL_CSV_COLUMNS = [
    "trial_number",
    "smac_configuration_tag",
    "objective_name",
    "objective_value",
    "converged",
    "failure_reason",
    "reason",
    "reason_code",
    "total_wall_time_sec",
    "subprocess_wall_time_sec",
    "matrix_load_time_sec",
    "solver_setup_time_sec",
    "solver_setup_time_sec_actual",
    "solver_setup_time_sec_logical",
    "ksp_setup_cache_hits",
    "ksp_setup_cache_misses",
    "solve_time_sec_total",
    "solve_time_sec_mean",
    "solve_time_sec_median",
    "solve_time_sec_min",
    "solve_time_sec_max",
    "solve_time_sec_stddev",
    "solve_time_sec_range",
    "solve_mpi_message_count",
    "solve_mpi_message_bytes",
    "solve_mpi_message_bytes_mean",
    "solve_mpi_reduction_count",
    "solve_count",
    "iterations_total",
    "iterations_median",
    "peak_memory_mb_sum",
    "peak_memory_mb_max_per_rank",
    "peak_memory_mb_mean_per_rank",
    "peak_memory_rank_count",
    "final_true_relative_residual_mean",
    "final_true_residual_norm_mean",
    "nullspace",
    "nullspace_source",
    "nullspace_kind",
    "nullspace_actions",
    "field_nullspace_index",
    "field_nullspace_block_size",
    "matrix_nullspace_attached_count",
    "transpose_nullspace_attached_count",
    "near_nullspace_attached_count",
    "rhs_nullspace_removed_count",
    "rhs_nullspace_removed_component_norm_max",
    "rhs_nullspace_removed_component_relative_norm_max",
    "ksp_type",
    "pc_type",
    "petsc_options_text",
    "solver_configuration_json",
    "replay_mode",
    "replay_result_path",
    "returncode",
    "replay_command_text",
    "replay_server_command_text",
]


def write_yaml_file(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def append_json_line(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def smac_configuration_tag(configuration: Any, chars: int = 6) -> str:
    return hashlib.sha1(str(configuration).encode("utf-8")).hexdigest()[:chars]


def emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def close_smac_runner(smac: Any) -> None:
    runner = getattr(smac, "_runner", None)
    close = getattr(runner, "close", None)
    if callable(close):
        close()


def count_tunable_hyperparameters(parameter_search_space: Any) -> int:
    from ConfigSpace import Constant

    return sum(
        1
        for hyperparameter in parameter_search_space.values()
        if not isinstance(hyperparameter, Constant)
    )


def resolve_sobol_initial_design_configuration_count(
    *,
    tunable_parameter_count: int,
    workers: int,
    effective_trials: int,
    default_solver_configuration_count: int,
    additional_solver_configuration_count: int,
) -> int:
    automatic_count = min(40, max(8, 2 * tunable_parameter_count, 4 * workers))
    reserved_trials = default_solver_configuration_count + additional_solver_configuration_count
    available_trials = max(0, effective_trials - reserved_trials)
    return min(automatic_count, available_trials)


def initial_solver_configuration_records_from_parameter_search_space(
    parameter_search_space: Any,
) -> list[dict[str, Any]]:
    from ConfigSpace import Configuration

    raw_records = list(getattr(parameter_search_space, "_ksptuneinitial_solver_configurations", []))
    records: list[dict[str, Any]] = []
    default_solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, dict):
            raise ValueError("Initial solver configurations must be dictionaries.")
        if "solver_configuration" in raw_record:
            label = str(raw_record.get("label") or f"initial_solver_configuration_{index}")
            solver_configuration_overrides = raw_record["solver_configuration"]
        else:
            label = f"initial_solver_configuration_{index}"
            solver_configuration_overrides = raw_record
        if not isinstance(solver_configuration_overrides, dict):
            raise ValueError(f"Initial solver configuration {label} must be a dictionary.")

        solver_configuration = dict(default_solver_configuration)
        solver_configuration.update(solver_configuration_overrides)
        try:
            configuration = Configuration(parameter_search_space, values=solver_configuration)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Initial solver configuration {label} is not legal: {exc}") from exc
        configuration.origin = f"KSPTune initial solver configuration: {label}"
        records.append(
            {
                "label": label,
                "solver_configuration": solver_configuration_from_configspace_configuration(
                    configuration
                ),
                "configuration": configuration,
            }
        )
    return records


def optimize_smac_with_known_warning_filter(smac: Any) -> Any:
    # SMAC local search can warn on empty conditional-neighbor sets. Filter only that known pair.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice.",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
        )
        return smac.optimize()


def load_trial_records(path: str | Path) -> list[dict[str, Any]]:
    trial_path = Path(path)
    if not trial_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in trial_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def output_directory_has_run_data(output_directory: str | Path) -> bool:
    output_path = Path(output_directory)
    trials_path = output_path / "solver_configuration_trials.jsonl"
    return (
        (output_path / "tuning_run.yaml").exists()
        or ((output_path / "smac").exists() and any((output_path / "smac").iterdir()))
        or (trials_path.exists() and trials_path.read_text(encoding="utf-8").strip() != "")
    )


def clear_restart_outputs(output_directory: str | Path) -> None:
    output_path = Path(output_directory)
    smac_path = output_path / "smac"
    if smac_path.exists():
        shutil.rmtree(smac_path)


def max_trial_number(records: list[dict[str, Any]]) -> int:
    trial_numbers = [
        int(record["trial_number"])
        for record in records
        if isinstance(record.get("trial_number"), int)
        or str(record.get("trial_number", "")).isdigit()
    ]
    return max(trial_numbers) if trial_numbers else 0


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(trial_record_to_csv_row(record))


def compact_trial_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "trial_number": record.get("trial_number"),
        "smac_configuration_tag": record.get("smac_configuration_tag"),
        "objective_value": record.get("objective_value"),
        "total_wall_time_sec": record.get("total_wall_time_sec"),
        "solver_setup_time_sec": record.get("solver_setup_time_sec"),
        "solver_setup_time_sec_actual": record.get("solver_setup_time_sec_actual"),
        "solver_setup_time_sec_logical": record.get("solver_setup_time_sec_logical"),
        "ksp_setup_cache_hits": record.get("ksp_setup_cache_hits"),
        "ksp_setup_cache_misses": record.get("ksp_setup_cache_misses"),
        "solve_time_sec_total": record.get("solve_time_sec_total"),
        "solve_time_sec_mean": record.get("solve_time_sec_mean"),
        "solve_time_sec_median": record.get("solve_time_sec_median"),
        "solve_time_sec_min": record.get("solve_time_sec_min"),
        "solve_time_sec_max": record.get("solve_time_sec_max"),
        "solve_time_sec_stddev": record.get("solve_time_sec_stddev"),
        "solve_time_sec_range": record.get("solve_time_sec_range"),
        "solve_mpi_message_count": record.get("solve_mpi_message_count"),
        "solve_mpi_message_bytes": record.get("solve_mpi_message_bytes"),
        "solve_mpi_message_bytes_mean": record.get("solve_mpi_message_bytes_mean"),
        "solve_mpi_reduction_count": record.get("solve_mpi_reduction_count"),
        "solve_count": record.get("solve_count"),
        "peak_memory_mb_sum": record.get("peak_memory_mb_sum"),
        "peak_memory_mb_max_per_rank": record.get("peak_memory_mb_max_per_rank"),
        "peak_memory_mb_mean_per_rank": record.get("peak_memory_mb_mean_per_rank"),
        "peak_memory_rank_count": record.get("peak_memory_rank_count"),
        "final_true_relative_residual_mean": record.get("final_true_relative_residual_mean"),
        "solver_configuration": record.get("solver_configuration"),
        "petsc_options": record.get("petsc_options"),
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
    converged_records = [record for record in records if record.get("converged") is True]
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

    memory_values = [
        float(record["peak_memory_mb_max_per_rank"])
        for record in records
        if is_number(record.get("peak_memory_mb_max_per_rank"))
    ]
    memory_sum_values = [
        float(record["peak_memory_mb_sum"])
        for record in records
        if is_number(record.get("peak_memory_mb_sum"))
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
        "peak_memory_mb_max_per_rank": max(memory_values) if memory_values else None,
        "peak_memory_mb_sum_max": max(memory_sum_values) if memory_sum_values else None,
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
    summary["trial_records_path"] = str(trial_records_path.resolve())
    summary["trial_csv_path"] = str((output_path / "solver_configuration_trials.csv").resolve())
    summary["ranking_csv_path"] = str((output_path / "solver_configuration_rankings.csv").resolve())
    summary["best_solver_configuration_path"] = str(
        (output_path / "best_solver_configuration.yaml").resolve()
    )
    summary["best_petsc_options_path"] = str((output_path / "best_petsc_options.txt").resolve())
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


def remove_best_solver_configuration_outputs(output_directory: str | Path) -> None:
    output_path = Path(output_directory)
    for filename in ("best_solver_configuration.yaml", "best_petsc_options.txt"):
        (output_path / filename).unlink(missing_ok=True)


def find_default_replay_binary() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    local_replay_binary = project_root / "build/cpp/petsc_replay" / REPLAY_BINARY_NAME
    if local_replay_binary.exists():
        return local_replay_binary.resolve()

    path_replay_binary = shutil.which(REPLAY_BINARY_NAME)
    if path_replay_binary:
        return Path(path_replay_binary).resolve()
    return None


def resolve_replay_binary(
    replay_binary: str | Path | None,
    *,
    dry_run: bool,
) -> tuple[Path | None, bool]:
    if replay_binary is not None:
        return Path(replay_binary), False
    if dry_run:
        return None, False

    default_replay_binary = find_default_replay_binary()
    if default_replay_binary is None:
        raise ValueError(
            "--replay-binary was not provided and KSPTune could not find "
            f"{REPLAY_BINARY_NAME}. Build it with `cmake --build build -j` or pass "
            "--replay-binary explicitly."
        )
    return default_replay_binary, True


def find_resume_smac_state_directory(output_directory: str | Path, seed: int) -> Path:
    smac_path = Path(output_directory) / "smac"
    if not smac_path.exists():
        raise ValueError(f"No SMAC state found in {smac_path}.")
    candidates: list[Path] = []
    for scenario_path in smac_path.glob(f"*/{seed}/scenario.json"):
        state_directory = scenario_path.parent
        if (
            (state_directory / "runhistory.json").exists()
            and (state_directory / "intensifier.json").exists()
        ):
            candidates.append(state_directory)
    if not candidates:
        raise ValueError(f"No complete SMAC state found in {smac_path} for seed {seed}.")
    if len(candidates) > 1:
        candidate_text = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple SMAC states found for seed {seed}: {candidate_text}")
    return candidates[0]


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def write_json_mapping(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.write_text(json.dumps(data, indent=4, allow_nan=True) + "\n", encoding="utf-8")


def normalize_resume_smac_scenario(
    smac_state_directory: str | Path,
    *,
    effective_trials: int,
    workers: int,
) -> dict[str, Any]:
    scenario_path = Path(smac_state_directory) / "scenario.json"
    scenario_data = load_json_mapping(scenario_path)
    scenario_data["n_trials"] = effective_trials
    scenario_data["n_workers"] = workers
    write_json_mapping(scenario_path, scenario_data)
    return scenario_data


def parameter_search_space_reference(tuning_run: dict[str, Any]) -> str:
    source = tuning_run.get("parameter_search_space_source")
    if isinstance(source, str) and source.startswith("builtin:"):
        return source.removeprefix("builtin:")
    if source:
        return str(source)
    parameter_search_space = tuning_run.get("parameter_search_space")
    if parameter_search_space:
        return str(parameter_search_space)
    raise ValueError("Cannot resume: tuning_run.yaml does not name a parameter search space.")


def resolve_resume_replay_binary(tuning_run: dict[str, Any]) -> str | Path | None:
    replay_binary = tuning_run.get("replay_binary")
    if tuning_run.get("replay_binary_auto_detected"):
        return None
    if replay_binary is None:
        return None
    replay_binary_path = Path(str(replay_binary))
    if not replay_binary_path.exists():
        raise ValueError(f"Cannot resume: replay binary no longer exists: {replay_binary_path}")
    return replay_binary_path


def nullspace_argument_from_configuration(configuration: dict[str, Any] | None) -> str | None:
    if not configuration:
        return "from-metadata"
    mode = configuration.get("mode")
    if mode in {"from-metadata", "none", "constant"}:
        return str(mode)
    if mode == "field":
        field_index = configuration.get("field_index")
        block_size = configuration.get("block_size", 2)
        return f"field:{field_index},block_size={block_size}"
    return str(configuration.get("label") or "from-metadata")


def nullspace_actions_argument_from_configuration(configuration: dict[str, Any] | None) -> str | None:
    if not configuration or configuration.get("mode") == "default":
        return None
    return str(configuration.get("label") or configuration.get("mode"))


def resume_timeout_value(tuning_run: dict[str, Any], override: float | None) -> float | None:
    if override is not None:
        return override
    if "timeout_sec" in tuning_run:
        return tuning_run.get("timeout_sec")
    return tuning_run.get("timeout_seconds")


def write_solver_configuration_outputs(
    *,
    output_directory: str | Path,
    parameter_search_space: Any,
    solver_configuration: dict[str, Any],
) -> None:
    directory = Path(output_directory)
    petsc_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    write_yaml_file(directory / "best_solver_configuration.yaml", solver_configuration)
    (directory / "best_petsc_options.txt").write_text(
        petsc_options_to_text(petsc_options) + "\n",
        encoding="utf-8",
    )


def run_tuning(
    *,
    snapshot_directory: str | Path,
    parameter_search_space: Any,
    output_directory: str | Path,
    replay_binary: str | Path | None = None,
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    threads_per_rank: int = 1,
    workers: int = 1,
    trials: int | None = 20,
    repeat: int = 1,
    warmup: int = 0,
    timeout_sec: float | None = None,
    objective_name: str = "solve_time_sec_mean",
    seed: int = 1,
    dry_run: bool = False,
    run_until_stopped: bool = False,
    nullspace: str | None = "from-metadata",
    nullspace_actions: str | None = None,
    deduplicate_matrices: bool = True,
    reuse_ksp_setup: bool = True,
    replay_cache_memory_mb: float | None = None,
    force_restart: bool = False,
    resume: bool = False,
    resume_smac_name: str | None = None,
    snapshot_collection_path_override: str | Path | None = None,
    replay_snapshot_collection_path_override: str | Path | None = None,
    existing_trial_records: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if resume and force_restart:
        raise ValueError("resume and force_restart cannot be used together.")
    resolved_mpiexec_args = list(mpiexec_args or [])

    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if not resume and output_directory_has_run_data(output_path):
        if not force_restart:
            raise ValueError(
                f"{output_path} already contains a KSPTune run. "
                "Use `ksptune resume RUN_DIR` to continue it or pass "
                "`--force-restart` to overwrite it."
            )
        clear_restart_outputs(output_path)

    if run_until_stopped:
        effective_trials = RUN_UNTIL_STOPPED_TRIAL_LIMIT
    else:
        if trials is None:
            raise ValueError("trials must be provided unless run_until_stopped is true.")
        effective_trials = trials
    replay_binary_path, replay_binary_auto_detected = resolve_replay_binary(
        replay_binary,
        dry_run=dry_run,
    )
    nullspace_configuration = parse_nullspace(nullspace)
    nullspace_action_configuration = parse_nullspace_actions(
        nullspace_actions,
        nullspace=nullspace,
    )
    nullspace_replay_options = [
        *list(nullspace_configuration["replay_options"]),
        *list(nullspace_action_configuration["replay_options"]),
    ]
    if not reuse_ksp_setup:
        nullspace_replay_options.extend(["-replay_reuse_ksp_setup", "false"])

    snapshot_directory_path = Path(snapshot_directory).resolve()
    snapshot_summary = summarize_snapshot_directory(snapshot_directory_path)
    if resume:
        snapshot_collection_path = Path(
            snapshot_collection_path_override or output_path / "snapshot_collection.csv"
        ).resolve()
        replay_snapshot_collection_path = Path(
            replay_snapshot_collection_path_override
            or output_path / "snapshot_collection.resolved.csv"
        ).resolve()
        if not snapshot_collection_path.exists():
            raise ValueError(f"Cannot resume: snapshot collection not found: {snapshot_collection_path}")
        if not replay_snapshot_collection_path.exists():
            raise ValueError(
                "Cannot resume: resolved snapshot collection not found: "
                f"{replay_snapshot_collection_path}"
            )
    else:
        snapshot_collection_path = create_snapshot_collection_from_directory(
            snapshot_directory_path,
            output_path / "snapshot_collection.csv",
            overwrite=True,
        )
        replay_snapshot_collection_path = write_resolved_snapshot_collection(
            snapshot_collection_path,
            output_path / "snapshot_collection.resolved.csv",
            deduplicate_matrices=deduplicate_matrices,
        )
    configuration_space = parameter_search_space

    use_default_solver_configuration = True
    default_solver_configuration_count = 1 if use_default_solver_configuration else 0
    all_additionalinitial_solver_configuration_records = (
        initial_solver_configuration_records_from_parameter_search_space(parameter_search_space)
    )
    maximum_additional_solver_configuration_count = max(
        0,
        effective_trials - default_solver_configuration_count,
    )
    additionalinitial_solver_configuration_records = (
        all_additionalinitial_solver_configuration_records[
            :maximum_additional_solver_configuration_count
        ]
    )
    tunable_parameter_count = count_tunable_hyperparameters(parameter_search_space)
    sobol_initial_design_configurations = resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=tunable_parameter_count,
        workers=workers,
        effective_trials=effective_trials,
        default_solver_configuration_count=default_solver_configuration_count,
        additional_solver_configuration_count=len(additionalinitial_solver_configuration_records),
    )
    additionalinitial_solver_configuration_labels = [
        record["label"] for record in additionalinitial_solver_configuration_records
    ]

    write_yaml_file(
        output_path / "tuning_run.yaml",
        {
            "snapshot_directory_path": str(snapshot_directory_path),
            "snapshot_collection_path": str(Path(snapshot_collection_path).resolve()),
            "replay_snapshot_collection_path": str(replay_snapshot_collection_path),
            "parameter_search_space": get_parameter_search_space_name(parameter_search_space),
            "parameter_search_space_source": getattr(
                parameter_search_space,
                "_ksptune_source",
                None,
            ),
            "replay_binary": str(replay_binary_path) if replay_binary_path else None,
            "replay_binary_auto_detected": replay_binary_auto_detected,
            "mpiexec": mpiexec,
            "mpiexec_args": resolved_mpiexec_args,
            "mpi_processes": mpi_processes,
            "threads_per_rank": threads_per_rank,
            "workers": workers,
            "trials": trials,
            "effective_trials": effective_trials,
            "run_until_stopped": run_until_stopped,
            "repeat": repeat,
            "warmup": warmup,
            "timeout_sec": timeout_sec,
            "objective_name": objective_name,
            "nullspace": nullspace_configuration,
            "nullspace_actions": nullspace_action_configuration,
            "deduplicate_matrices": deduplicate_matrices,
            "reuse_ksp_setup": reuse_ksp_setup,
            "replay_cache_memory_mb": replay_cache_memory_mb,
            "resume": resume,
            "force_restart": force_restart,
            "resumed_completed_trials": max_trial_number(existing_trial_records or []),
            "failed_replay_objective_value": BAD_COST,
            "smac_deterministic": True,
            "use_default_solver_configuration": use_default_solver_configuration,
            "tunable_parameter_count": tunable_parameter_count,
            "sobol_initial_design_configurations": sobol_initial_design_configurations,
            "additionalinitial_solver_configuration_count": len(
                additionalinitial_solver_configuration_records
            ),
            "additionalinitial_solver_configurations": additionalinitial_solver_configuration_labels,
            "seed": seed,
            "dry_run": dry_run,
            "snapshot_summary": snapshot_summary,
        },
    )
    (output_path / "parameter_search_space.yaml").write_text(
        parameter_search_space_to_yaml(parameter_search_space),
        encoding="utf-8",
    )
    write_configspace_yaml(configuration_space, output_path / "configspace.yaml")
    write_configspace_json(configuration_space, output_path / "configspace.json")

    parameter_search_space_name = get_parameter_search_space_name(parameter_search_space)
    output_paths = tuning_output_paths(output_path)
    trials_path = output_path / "solver_configuration_trials.jsonl"
    if resume:
        trials_path.touch(exist_ok=True)
    else:
        trials_path.write_text("", encoding="utf-8")
        remove_best_solver_configuration_outputs(output_path)
    completed_trial_count = max_trial_number(existing_trial_records or [])
    emit_progress(
        progress_callback,
        {
            "event": "run_started",
            "snapshot_directory_path": str(snapshot_directory_path),
            "snapshot_collection_path": str(Path(snapshot_collection_path).resolve()),
            "replay_snapshot_collection_path": str(replay_snapshot_collection_path),
            "snapshot_summary": snapshot_summary,
            "parameter_search_space_name": parameter_search_space_name,
            "output_directory": str(output_path),
            "replay_binary": str(replay_binary_path.resolve()) if replay_binary_path else None,
            "replay_binary_auto_detected": replay_binary_auto_detected,
            "mpiexec": mpiexec,
            "mpiexec_args": resolved_mpiexec_args,
            "mpi_processes": mpi_processes,
            "threads_per_rank": threads_per_rank,
            "workers": workers,
            "trials": trials,
            "effective_trials": effective_trials,
            "run_until_stopped": run_until_stopped,
            "repeat": repeat,
            "warmup": warmup,
            "timeout_sec": timeout_sec,
            "objective_name": objective_name,
            "use_default_solver_configuration": use_default_solver_configuration,
            "tunable_parameter_count": tunable_parameter_count,
            "sobol_initial_design_configurations": sobol_initial_design_configurations,
            "additionalinitial_solver_configuration_count": len(
                additionalinitial_solver_configuration_records
            ),
            "additionalinitial_solver_configurations": additionalinitial_solver_configuration_labels,
            "nullspace_configuration": nullspace_configuration,
            "nullspace_action_configuration": nullspace_action_configuration,
            "deduplicate_matrices": deduplicate_matrices,
            "reuse_ksp_setup": reuse_ksp_setup,
            "replay_cache_memory_mb": replay_cache_memory_mb,
            "resume": resume,
            "force_restart": force_restart,
            "completed_trials": completed_trial_count,
            "remaining_trials": (
                None
                if run_until_stopped
                else max(0, int(effective_trials) - int(completed_trial_count))
            ),
            "dry_run": dry_run,
        },
    )

    default_solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    if dry_run:
        write_solver_configuration_outputs(
            output_directory=output_path,
            parameter_search_space=parameter_search_space,
            solver_configuration=default_solver_configuration,
        )
        summary = write_tuning_summary_outputs(
            output_directory=output_path,
            objective_name=objective_name,
            status="dry-run",
        )
        result = {
            "status": "dry-run",
            "output_directory": str(output_path),
            "default_solver_configuration": default_solver_configuration,
            "summary": summary,
            **output_paths,
        }
        emit_progress(progress_callback, {"event": "run_finished", **result})
        return result

    from smac import HyperparameterOptimizationFacade, Scenario

    completed_trial_counter = {"value": completed_trial_count}
    best = {
        "cost": math.inf,
        "solver_configuration": None,
        "trial_number": None,
        "smac_configuration_tag": None,
    }
    existing_best_record = best_record_from_trial_records(existing_trial_records or [])
    if existing_best_record is not None:
        best["cost"] = float(existing_best_record["objective_value"])
        best["solver_configuration"] = existing_best_record.get("solver_configuration")
        best["trial_number"] = existing_best_record.get("trial_number")
        best["smac_configuration_tag"] = existing_best_record.get("smac_configuration_tag")
        if best["solver_configuration"] is not None:
            write_solver_configuration_outputs(
                output_directory=output_path,
                parameter_search_space=parameter_search_space,
                solver_configuration=best["solver_configuration"],
            )

    if resume and not run_until_stopped and completed_trial_count >= effective_trials:
        summary = write_tuning_summary_outputs(
            output_directory=output_path,
            objective_name=objective_name,
            status="completed",
        )
        best_trial = summary.get("best_trial") or {}
        best_solver_configuration = best_trial.get("solver_configuration")
        if best_solver_configuration is not None:
            write_solver_configuration_outputs(
                output_directory=output_path,
                parameter_search_space=parameter_search_space,
                solver_configuration=best_solver_configuration,
            )
        result = {
            "status": "completed",
            "output_directory": str(output_path),
            "stopped_by_user": False,
            "best_cost": best_trial.get("objective_value"),
            "best_solver_configuration": best_solver_configuration,
            "best_petsc_options": best_trial.get("petsc_options"),
            "summary": summary,
            **output_paths,
        }
        emit_progress(progress_callback, {"event": "run_finished", **result})
        return result

    replay_worker_pool = ReplayWorkerPool(
        replay_binary=replay_binary_path,
        snapshot_collection_path=replay_snapshot_collection_path,
        workers=workers,
        mpiexec=mpiexec,
        mpiexec_args=resolved_mpiexec_args,
        mpi_processes=mpi_processes,
        threads_per_rank=threads_per_rank,
        extra_replay_options=nullspace_replay_options,
        cache_memory_mb=replay_cache_memory_mb,
    )

    def best_so_far_progress_fields() -> dict[str, Any]:
        if math.isinf(float(best["cost"])):
            return {
                "best_objective_value_so_far": None,
                "best_trial_number_so_far": None,
                "best_smac_configuration_tag_so_far": None,
            }
        return {
            "best_objective_value_so_far": best["cost"],
            "best_trial_number_so_far": best["trial_number"],
            "best_smac_configuration_tag_so_far": best["smac_configuration_tag"],
        }

    class KSPTuneProgressCallback:
        def on_start(self, smbo: Any) -> None:  # noqa: ARG002
            return None

        def on_end(self, smbo: Any) -> None:  # noqa: ARG002
            return None

        def on_iteration_start(self, smbo: Any) -> None:  # noqa: ARG002
            return None

        def on_iteration_end(self, smbo: Any) -> None:  # noqa: ARG002
            return None

        def on_next_configurations_start(self, config_selector: Any) -> None:  # noqa: ARG002
            return None

        def on_next_configurations_end(self, config_selector: Any, config: Any) -> None:  # noqa: ARG002
            return None

        def on_ask_start(self, smbo: Any) -> None:  # noqa: ARG002
            return None

        def on_ask_end(self, smbo: Any, info: Any) -> None:  # noqa: ARG002
            return None

        def on_tell_start(self, smbo: Any, info: Any, value: Any) -> None:  # noqa: ARG002
            return None

        def on_tell_end(self, smbo: Any, info: Any, value: Any) -> None:  # noqa: ARG002
            completed_trial_counter["value"] += 1
            trial_number = completed_trial_counter["value"]
            record = dict(value.additional_info.get("ksptune_trial_record") or {})
            if not record:
                configuration_tag = smac_configuration_tag(info.config)
                solver_configuration = solver_configuration_from_configspace_configuration(info.config)
                record = {
                    "smac_configuration_tag": configuration_tag,
                    "solver_configuration": solver_configuration,
                    "petsc_options": render_petsc_options_from_solver_configuration(
                        parameter_search_space,
                        solver_configuration,
                    ),
                    "objective_name": objective_name,
                    "objective_value": value.cost,
                    "failure_reason": value.additional_info.get("error", "target function failed"),
                }

            record["trial_number"] = trial_number
            record["objective_value"] = value.cost
            append_json_line(trials_path, record)

            cost = float(value.cost)
            is_new_best_so_far = is_successful_trial(record) and cost < best["cost"]
            if is_new_best_so_far:
                best["cost"] = cost
                best["solver_configuration"] = record.get("solver_configuration")
                best["trial_number"] = trial_number
                best["smac_configuration_tag"] = record.get("smac_configuration_tag")
                write_solver_configuration_outputs(
                    output_directory=output_path,
                    parameter_search_space=parameter_search_space,
                    solver_configuration=record["solver_configuration"],
                )
            emit_progress(
                progress_callback,
                {
                    "event": "trial_finished",
                    "trial_count": None if run_until_stopped else trials,
                    "run_until_stopped": run_until_stopped,
                    **best_so_far_progress_fields(),
                    "is_new_best_so_far": is_new_best_so_far,
                    **record,
                },
            )

    def target_function(configuration, seed: int = 0) -> tuple[float, dict[str, Any]]:  # noqa: ARG001
        configuration_tag = smac_configuration_tag(configuration)
        solver_configuration = solver_configuration_from_configspace_configuration(configuration)
        petsc_options = render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
        replay_record = run_replay_server_for_solver_configuration(
            replay_worker_pool=replay_worker_pool,
            replay_binary=replay_binary_path,
            snapshot_collection_path=replay_snapshot_collection_path,
            replay_result_path=(
                output_path
                / "replay_results"
                / f"candidate_{configuration_tag}_{uuid.uuid4().hex}.json"
            ),
            petsc_options=petsc_options,
            mpiexec=mpiexec,
            mpiexec_args=resolved_mpiexec_args,
            mpi_processes=mpi_processes,
            repeat=repeat,
            warmup=warmup,
            timeout_sec=timeout_sec,
            threads_per_rank=threads_per_rank,
            extra_replay_options=nullspace_replay_options,
        )
        replay_result = replay_record.get("replay_result", {})
        replay_command = replay_record.get("command") or []
        failure_reason = replay_failure_reason(replay_record, objective_name=objective_name)
        cost = replay_objective_value(
            replay_record,
            objective_name=objective_name,
            bad_cost=BAD_COST,
        )
        record = {
            "smac_configuration_tag": configuration_tag,
            "solver_configuration": solver_configuration,
            "petsc_options": petsc_options,
            "petsc_options_text": petsc_options_to_text(petsc_options),
            "replay_command": replay_command,
            "replay_command_text": shlex.join(str(part) for part in replay_command),
            "replay_server_command": replay_record.get("replay_server_command"),
            "replay_server_command_text": shlex.join(
                str(part) for part in replay_record.get("replay_server_command", [])
            ),
            "replay_mode": replay_record.get("replay_mode", "server"),
            "objective_name": objective_name,
            "objective_value": cost,
            "replay_result_path": replay_record["replay_result_path"],
            "returncode": replay_record["returncode"],
            "failure_reason": failure_reason,
            "schema_errors": replay_record.get("schema_errors", []),
            "subprocess_wall_time_sec": replay_record.get("subprocess_wall_time_sec"),
            **replay_metric_summary(replay_result),
        }
        return cost, {"ksptune_trial_record": record}

    scenario_kwargs = {
        "configspace": configuration_space,
        "output_directory": output_path / "smac",
        "n_trials": effective_trials,
        "seed": seed,
        "deterministic": True,
        "crash_cost": BAD_COST,
        "n_workers": workers,
        "use_default_config": use_default_solver_configuration,
    }
    if resume_smac_name is not None:
        scenario_kwargs["name"] = resume_smac_name
    scenario = Scenario(**scenario_kwargs)
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=sobol_initial_design_configurations,
        max_ratio=1.0,
        additional_configs=[
            record["configuration"] for record in additionalinitial_solver_configuration_records
        ],
    )
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=target_function,
        callbacks=[KSPTuneProgressCallback()],
        initial_design=initial_design,
        overwrite=not resume,
    )
    status = "completed"
    try:
        optimize_smac_with_known_warning_filter(smac)
    except KeyboardInterrupt:
        status = "stopped"
        emit_progress(
            progress_callback,
            {
                "event": "run_stopping",
                "output_directory": str(output_path),
                "trial_count": completed_trial_counter["value"],
            },
        )
    finally:
        replay_worker_pool.close()
        close_smac_runner(smac)
    summary = write_tuning_summary_outputs(
        output_directory=output_path,
        objective_name=objective_name,
        status=status,
    )
    best_trial = summary.get("best_trial") or {}
    best_solver_configuration = best_trial.get("solver_configuration")
    if best_solver_configuration is not None:
        write_solver_configuration_outputs(
            output_directory=output_path,
            parameter_search_space=parameter_search_space,
            solver_configuration=best_solver_configuration,
        )
    result = {
        "status": status,
        "output_directory": str(output_path),
        "stopped_by_user": status == "stopped",
        "best_cost": best_trial.get("objective_value"),
        "best_solver_configuration": best_solver_configuration,
        "best_petsc_options": best_trial.get("petsc_options"),
        "summary": summary,
        **output_paths,
    }
    emit_progress(progress_callback, {"event": "run_finished", **result})
    return result


def resume_tuning(
    tuning_run_directory: str | Path,
    *,
    trials: int | None = None,
    run_until_stopped: bool | None = None,
    workers: int | None = None,
    timeout_sec: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output_path = Path(tuning_run_directory).resolve()
    tuning_run_path = output_path / "tuning_run.yaml"
    trials_path = output_path / "solver_configuration_trials.jsonl"
    if not tuning_run_path.exists():
        raise ValueError(f"Cannot resume: tuning_run.yaml not found in {output_path}.")
    if not trials_path.exists():
        raise ValueError(
            f"Cannot resume: solver_configuration_trials.jsonl not found in {output_path}."
        )

    tuning_run = load_yaml_mapping(tuning_run_path)
    if tuning_run.get("dry_run"):
        raise ValueError("Cannot resume a dry-run tuning directory.")

    seed = int(tuning_run.get("seed", 1))
    smac_state_directory = find_resume_smac_state_directory(output_path, seed)

    old_run_until_stopped = bool(tuning_run.get("run_until_stopped", False))
    if run_until_stopped is None:
        resolved_run_until_stopped = old_run_until_stopped if trials is None else False
    else:
        resolved_run_until_stopped = run_until_stopped
    resolved_trials = tuning_run.get("trials") if trials is None else trials
    if not resolved_run_until_stopped and resolved_trials is None:
        raise ValueError("Cannot resume with a fixed trial budget because the trial count is unknown.")
    effective_trials = (
        RUN_UNTIL_STOPPED_TRIAL_LIMIT
        if resolved_run_until_stopped
        else int(resolved_trials)
    )
    resolved_workers = int(workers if workers is not None else tuning_run.get("workers", 1))

    scenario_data = normalize_resume_smac_scenario(
        smac_state_directory,
        effective_trials=effective_trials,
        workers=resolved_workers,
    )
    smac_name = str(scenario_data.get("name") or smac_state_directory.parent.name)

    parameter_search_space = load_parameter_search_space(
        parameter_search_space_reference(tuning_run),
        seed=seed,
    )
    existing_records = load_trial_records(trials_path)

    return run_tuning(
        snapshot_directory=tuning_run["snapshot_directory_path"],
        parameter_search_space=parameter_search_space,
        output_directory=output_path,
        replay_binary=resolve_resume_replay_binary(tuning_run),
        mpiexec=str(tuning_run.get("mpiexec") or "mpiexec"),
        mpiexec_args=list(tuning_run.get("mpiexec_args") or []),
        mpi_processes=int(tuning_run.get("mpi_processes", 1)),
        threads_per_rank=int(tuning_run.get("threads_per_rank", 1)),
        workers=resolved_workers,
        trials=resolved_trials,
        repeat=int(tuning_run.get("repeat", 1)),
        warmup=int(tuning_run.get("warmup", 0)),
        timeout_sec=resume_timeout_value(tuning_run, timeout_sec),
        objective_name=str(tuning_run.get("objective_name") or "solve_time_sec_mean"),
        seed=seed,
        dry_run=False,
        run_until_stopped=resolved_run_until_stopped,
        nullspace=nullspace_argument_from_configuration(tuning_run.get("nullspace")),
        nullspace_actions=nullspace_actions_argument_from_configuration(
            tuning_run.get("nullspace_actions")
        ),
        deduplicate_matrices=bool(tuning_run.get("deduplicate_matrices", True)),
        reuse_ksp_setup=bool(tuning_run.get("reuse_ksp_setup", True)),
        replay_cache_memory_mb=tuning_run.get("replay_cache_memory_mb"),
        resume=True,
        resume_smac_name=smac_name,
        snapshot_collection_path_override=tuning_run.get("snapshot_collection_path"),
        replay_snapshot_collection_path_override=tuning_run.get(
            "replay_snapshot_collection_path"
        ),
        existing_trial_records=existing_records,
        progress_callback=progress_callback,
    )
