from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REQUIRED_REPLAY_RESULT_FIELDS = {
    "objective_sec",
    "total_wall_time_seconds",
    "converged",
    "reason_code",
    "solve_time_seconds_median",
    "solver_setup_time_seconds",
    "iterations_total",
    "snapshots",
    "steps",
}

NUMERIC_REPLAY_RESULT_FIELDS = {
    "objective_sec",
    "total_wall_time_seconds",
    "matrix_load_time_seconds",
    "solver_setup_time_seconds",
    "solve_time_seconds_total",
    "solve_time_seconds_mean",
    "solve_time_seconds_median",
    "iterations_total",
    "initial_true_residual_norm_mean",
    "initial_true_relative_residual_mean",
    "final_true_residual_norm_mean",
    "final_true_relative_residual_mean",
    "peak_memory_megabytes_max",
    "peak_memory_megabytes_sum",
    "solve_count",
    "field_nullspace_index",
    "field_nullspace_block_size",
    "matrix_nullspace_attached_count",
    "transpose_nullspace_attached_count",
    "near_nullspace_attached_count",
    "rhs_nullspace_removed_count",
    "rhs_nullspace_removed_component_norm_max",
    "rhs_nullspace_removed_component_relative_norm_max",
}

PETSC_RETURN_CODE_HINTS = {
    59: "PETSc caught a signal, commonly a SIGSEGV",
}


def build_replay_command(
    *,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    replay_result_path: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpi_processes: int = 1,
    repeat: int = 1,
    warmup: int = 0,
    extra_replay_options: list[str] | None = None,
) -> list[str]:
    replay_command = [
        str(replay_binary),
        "-snapshot_collection",
        str(snapshot_collection_path),
        "-replay_json_out",
        str(replay_result_path),
        "-replay_repeat",
        str(repeat),
        "-replay_warmup",
        str(warmup),
    ]
    if extra_replay_options:
        replay_command.extend(extra_replay_options)
    replay_command.extend(petsc_options)

    if mpi_processes > 1:
        return [mpiexec, "-n", str(mpi_processes), *replay_command]
    return replay_command


def replay_environment(threads_per_rank: int = 1) -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads_per_rank)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    return environment


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_replay_result(
    replay_result: dict[str, Any],
    *,
    objective_name: str = "objective_sec",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(replay_result, dict):
        return ["replay result must be a JSON object"]

    for field in sorted(REQUIRED_REPLAY_RESULT_FIELDS):
        if field not in replay_result:
            errors.append(f"missing replay metric: {field}")

    if objective_name not in replay_result:
        errors.append(f"missing objective metric: {objective_name}")

    for field in sorted(NUMERIC_REPLAY_RESULT_FIELDS | {objective_name}):
        if field in replay_result and not is_number(replay_result[field]):
            errors.append(f"replay metric must be numeric: {field}")

    if "converged" in replay_result and not isinstance(replay_result["converged"], bool):
        errors.append("replay metric must be boolean: converged")
    if "reason_code" in replay_result and not isinstance(replay_result["reason_code"], int):
        errors.append("replay metric must be integer: reason_code")
    if "steps" in replay_result and not isinstance(replay_result["steps"], list):
        errors.append("replay metric must be a list: steps")
    if "peak_memory_megabytes_per_rank" in replay_result and not isinstance(
        replay_result["peak_memory_megabytes_per_rank"],
        list,
    ):
        errors.append("replay metric must be a list: peak_memory_megabytes_per_rank")

    return errors


def replay_failure_reason(
    replay_record: dict[str, Any],
    *,
    objective_name: str = "objective_sec",
) -> str | None:
    if replay_record.get("failure_reason"):
        return str(replay_record["failure_reason"])

    returncode = replay_record.get("returncode")
    if returncode != 0:
        hint = PETSC_RETURN_CODE_HINTS.get(returncode)
        if hint:
            return f"replay command returned {returncode} ({hint})"
        return f"replay command returned {returncode}"

    schema_errors = replay_record.get("schema_errors") or validate_replay_result(
        replay_record.get("replay_result", {}),
        objective_name=objective_name,
    )
    if schema_errors:
        return "invalid replay result: " + "; ".join(str(error) for error in schema_errors)

    replay_result = replay_record.get("replay_result", {})
    if replay_result.get("converged") is False:
        reason = replay_result.get("reason", replay_result.get("reason_code", "unknown"))
        return f"not converged: {reason}"

    return None


def replay_objective_value(
    replay_record: dict[str, Any],
    *,
    objective_name: str = "objective_sec",
    bad_cost: float,
) -> float:
    if replay_failure_reason(replay_record, objective_name=objective_name):
        return bad_cost
    try:
        return float(replay_record["replay_result"][objective_name])
    except (KeyError, TypeError, ValueError):
        return bad_cost


def solve_step_times(replay_result: dict[str, Any]) -> list[float]:
    steps = replay_result.get("steps")
    if not isinstance(steps, list):
        return []

    solve_times: list[float] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        value = step.get("solve_time_seconds", step.get("solve_sec"))
        if is_number(value):
            solve_times.append(float(value))
    return solve_times


def expected_solve_count(replay_result: dict[str, Any]) -> int | None:
    if is_number(replay_result.get("snapshots")) and is_number(replay_result.get("repeat")):
        return int(replay_result["snapshots"]) * int(replay_result["repeat"])
    return None


def replay_metric_summary(replay_result: dict[str, Any]) -> dict[str, Any]:
    metric_names = [
        "total_wall_time_seconds",
        "matrix_load_time_seconds",
        "solver_setup_time_seconds",
        "solve_time_seconds_total",
        "solve_time_seconds_mean",
        "solve_time_seconds_median",
        "iterations_total",
        "iterations_median",
        "peak_memory_megabytes_max",
        "peak_memory_megabytes_sum",
        "initial_true_residual_norm_mean",
        "initial_true_relative_residual_mean",
        "final_true_residual_norm_mean",
        "final_true_relative_residual_mean",
        "converged",
        "reason",
        "reason_code",
        "snapshots",
        "repeat",
        "warmup",
        "solve_count",
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
    ]
    summary = {name: replay_result.get(name) for name in metric_names if name in replay_result}

    solve_times = solve_step_times(replay_result)
    if "solve_count" not in summary:
        if solve_times:
            summary["solve_count"] = len(solve_times)
        elif (expected_count := expected_solve_count(replay_result)) is not None:
            summary["solve_count"] = expected_count

    if "solve_time_seconds_total" not in summary:
        if solve_times:
            summary["solve_time_seconds_total"] = sum(solve_times)
        elif is_number(replay_result.get("solve_time_seconds_mean")) and is_number(
            summary.get("solve_count")
        ):
            summary["solve_time_seconds_total"] = (
                float(replay_result["solve_time_seconds_mean"]) * int(summary["solve_count"])
            )

    return summary


def run_replay_for_solver_configuration(
    *,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    replay_result_path: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpi_processes: int = 1,
    repeat: int = 1,
    warmup: int = 0,
    timeout_seconds: float | None = None,
    threads_per_rank: int = 1,
    extra_replay_options: list[str] | None = None,
) -> dict[str, Any]:
    result_path = Path(replay_result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_replay_command(
        replay_binary=replay_binary,
        snapshot_collection_path=snapshot_collection_path,
        replay_result_path=result_path,
        petsc_options=petsc_options,
        mpiexec=mpiexec,
        mpi_processes=mpi_processes,
        repeat=repeat,
        warmup=warmup,
        extra_replay_options=extra_replay_options,
    )
    start_time = time.perf_counter()
    failure_reason: str | None = None
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout_seconds,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=replay_environment(threads_per_rank),
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        failure_reason = f"replay command timed out after {timeout_seconds} seconds"
    subprocess_walltime_seconds = time.perf_counter() - start_time

    replay_result: dict[str, Any] = {}
    schema_errors: list[str] = []
    if result_path.exists():
        try:
            replay_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            schema_errors.append(f"invalid replay JSON: {exc}")
    else:
        schema_errors.append(f"missing replay JSON: {result_path}")

    if replay_result:
        schema_errors.extend(validate_replay_result(replay_result))

    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "subprocess_walltime_seconds": subprocess_walltime_seconds,
        "replay_result_path": str(result_path),
        "replay_result": replay_result,
        "schema_errors": schema_errors,
        "failure_reason": failure_reason,
    }
