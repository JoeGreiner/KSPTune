from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple

REPLAY_METRIC_TYPES = {
    "objective_time_sec_median": float,
    "total_wall_time_sec": float,
    "matrix_load_time_sec": float,
    "matrix_prepare_time_sec": float,
    "vector_prepare_time_sec": float,
    "nullspace_time_sec": float,
    "true_residual_time_sec": float,
    "use_initial_guess": bool,
    "soft_timeout_sec": float,
    "soft_timeout_elapsed_sec": float,
    "soft_timeout_triggered": bool,
    "matrix_cache_hits": int,
    "matrix_cache_misses": int,
    "replay_cache_memory_mb": float,
    "matrix_cache_memory_mb": float,
    "vector_cache_memory_mb": float,
    "replay_cache_memory_limit_mb": float,
    "replay_cache_evictions": int,
    "solver_setup_time_sec": float,
    "solver_setup_time_sec_actual": float,
    "solver_setup_time_sec_logical": float,
    "ksp_setup_cache_hits": int,
    "ksp_setup_cache_misses": int,
    "solve_time_sec_total": float,
    "solve_time_sec_mean": float,
    "solve_time_sec_median": float,
    "solve_time_sec_min": float,
    "solve_time_sec_max": float,
    "solve_time_sec_stddev": float,
    "solve_time_sec_sem": float,
    "solve_time_sec_relative_sem": float,
    "solve_time_sec_range": float,
    "solve_mpi_message_count": float,
    "solve_mpi_message_bytes": float,
    "solve_mpi_message_bytes_mean": float,
    "solve_mpi_reduction_count": float,
    "iterations_total": float,
    "iterations_median": float,
    "rss_request_start_mb_sum": float,
    "rss_request_end_mb_sum": float,
    "rss_request_peak_sample_mb_sum": float,
    "rss_request_delta_mb_sum": float,
    "rss_setup_delta_mb_sum": float,
    "rss_solve_delta_mb_sum": float,
    "rss_peak_sample_mb_max_rank": float,
    "rss_peak_sample_mb_max_node": float,
    "initial_true_residual_norm_mean": float,
    "initial_true_relative_residual_mean": float,
    "final_true_residual_norm_mean": float,
    "final_true_relative_residual_mean": float,
    "initial_ksp_residual_norm_mean": float,
    "final_ksp_residual_norm_mean": float,
    "ksp_rtol_reference_norm_mean": float,
    "ksp_convergence_threshold_norm_mean": float,
    "final_ksp_relative_residual_norm_mean": float,
    "ksp_residual_norm_type": str,
    "ksp_rtol_reference_source": str,
    "ksp_rtol": float,
    "ksp_atol": float,
    "ksp_dtol": float,
    "ksp_max_it": int,
    "converged": bool,
    "reason": str,
    "reason_code": int,
    "snapshots": int,
    "repeat": int,
    "warmup": int,
    "solve_count": int,
    "nullspace": str,
    "nullspace_source": str,
    "nullspace_kind": str,
    "nullspace_actions": str,
    "field_nullspace_index": int,
    "field_nullspace_block_size": int,
    "matrix_nullspace_attached_count": int,
    "transpose_nullspace_attached_count": int,
    "near_nullspace_attached_count": int,
    "rhs_nullspace_removed_count": int,
    "rhs_nullspace_removed_component_norm_max": float,
    "rhs_nullspace_removed_component_relative_norm_max": float,
    "ksp_type": str,
    "pc_type": str,
}

NULLABLE_REPLAY_RESULT_FIELDS = {
    "solve_time_sec_sem",
    "solve_time_sec_relative_sem",
}

PETSC_RETURN_CODE_HINTS = {
    59: "PETSc caught a signal, commonly a SIGSEGV",
}
DEFAULT_MAX_TRUE_RELATIVE_RESIDUAL = 1.0e-4
DEFAULT_MAX_TRUE_RESIDUAL_NORM = 1.0e-4
SOFT_TIMEOUT_FAILURE_COST_FACTOR = 1.5
HARD_TIMEOUT_FAILURE_COST_FACTOR = 2.0
OOM_FAILURE_COST_FACTOR = 10.0
OOM_RETURN_CODES = {-9, 137}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_string_metric_map(value: Any, name: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"replay metric must be an object: {name}"]
    return [
        f"replay metric value must be a string: {name}.{key}"
        for key, metric_value in value.items()
        if not isinstance(metric_value, str)
    ]


def validate_numeric_metric_map(value: Any, name: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"replay metric must be an object: {name}"]
    return [
        f"replay metric value must be numeric: {name}.{key}"
        for key, metric_value in value.items()
        if not is_number(metric_value)
    ]


def validate_pc_diagnostics(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["replay metric must be a list: pc_diagnostics"]

    errors: list[str] = []
    for record_index, record in enumerate(value):
        record_name = f"pc_diagnostics[{record_index}]"
        if not isinstance(record, dict):
            errors.append(f"replay metric must be an object: {record_name}")
            continue
        if "pc_type" in record and not isinstance(record["pc_type"], str):
            errors.append(f"replay metric must be a string: {record_name}.pc_type")
        if "setup_key" in record and not isinstance(record["setup_key"], str):
            errors.append(f"replay metric must be a string: {record_name}.setup_key")
        if "setup_index" in record and not isinstance(record["setup_index"], int):
            errors.append(f"replay metric must be an integer: {record_name}.setup_index")
        if "metrics" in record:
            errors.extend(validate_numeric_metric_map(record["metrics"], f"{record_name}.metrics"))
        if "string_metrics" in record:
            errors.extend(
                validate_string_metric_map(
                    record["string_metrics"],
                    f"{record_name}.string_metrics",
                )
            )
        levels = record.get("levels")
        if levels is None:
            continue
        if not isinstance(levels, list):
            errors.append(f"replay metric must be a list: {record_name}.levels")
            continue
        for level_index, level in enumerate(levels):
            level_name = f"{record_name}.levels[{level_index}]"
            if not isinstance(level, dict):
                errors.append(f"replay metric must be an object: {level_name}")
                continue
            if "level" in level and not isinstance(level["level"], int):
                errors.append(f"replay metric must be an integer: {level_name}.level")
            if "level_from_finest" in level and not isinstance(
                level["level_from_finest"],
                int,
            ):
                errors.append(f"replay metric must be an integer: {level_name}.level_from_finest")
            if "metrics" in level:
                errors.extend(
                    validate_numeric_metric_map(level["metrics"], f"{level_name}.metrics")
                )
            if "string_metrics" in level:
                errors.extend(
                    validate_string_metric_map(
                        level["string_metrics"],
                        f"{level_name}.string_metrics",
                    )
                )
    return errors


def validate_replay_result(
    replay_result: dict[str, Any],
    *,
    objective_name: str = "solve_time_sec_mean",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(replay_result, dict):
        return ["replay result must be a JSON object"]

    version = replay_result.get("schema_version")
    if type(version) is not int or version != 1:
        return [f"unsupported replay schema version: {version!r}; expected 1"]
    for field, kind in REPLAY_METRIC_TYPES.items():
        if field not in replay_result:
            errors.append(f"missing replay metric: {field}")
            continue
        value = replay_result[field]
        if value is None and field in NULLABLE_REPLAY_RESULT_FIELDS:
            continue
        valid = is_number(value) if kind is float else type(value) is kind
        if not valid:
            errors.append(f"replay metric must be {kind.__name__}: {field}")
    if objective_name not in replay_result:
        errors.append(f"missing objective metric: {objective_name}")
    elif not is_number(replay_result[objective_name]):
        errors.append(f"objective metric must be finite: {objective_name}")

    steps = replay_result.get("steps")
    if not isinstance(steps, list):
        errors.append("replay metric must be a list: steps")
    else:
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"replay step must be an object: steps[{index}]")
                continue
            for field in (
                "solve_time_sec", "final_true_residual_norm", "final_true_relative_residual",
            ):
                if not is_number(step.get(field)):
                    errors.append(f"replay step metric must be finite: steps[{index}].{field}")
    if "pc_diagnostics" in replay_result:
        errors.extend(validate_pc_diagnostics(replay_result["pc_diagnostics"]))
    return errors


class ReplayEvaluation(NamedTuple):
    status: str
    objective_value: float
    failure_reason: str | None = None


def evaluate_replay(
    replay_record: dict[str, Any],
    *,
    bad_cost: float,
    objective_name: str = "solve_time_sec_mean",
    max_true_relative_residual: float | None = DEFAULT_MAX_TRUE_RELATIVE_RESIDUAL,
    max_true_residual_norm: float | None = DEFAULT_MAX_TRUE_RESIDUAL_NORM,
) -> ReplayEvaluation:
    returncode = replay_record.get("returncode")
    failure_reason = replay_record.get("failure_reason")
    if failure_reason or returncode != 0:
        status = replay_record.get("failure_kind") or (
            "oom" if returncode in OOM_RETURN_CODES else "process_error"
        )
        if not failure_reason:
            failure_reason = f"replay command returned {returncode}"
            if hint := PETSC_RETURN_CODE_HINTS.get(returncode):
                failure_reason += f" ({hint})"
        timeout = replay_record.get("effective_hard_timeout_sec")
        factor = {
            "hard_timeout": HARD_TIMEOUT_FAILURE_COST_FACTOR,
            "oom": OOM_FAILURE_COST_FACTOR,
        }.get(status)
        cost = factor * timeout if factor is not None and is_number(timeout) else bad_cost
        return ReplayEvaluation(status, cost, failure_reason)

    result = replay_record.get("replay_result", {})
    errors = replay_record.get("schema_errors") or validate_replay_result(
        result, objective_name=objective_name
    )
    replay_record["schema_errors"] = errors
    if errors:
        return ReplayEvaluation(
            "invalid_result", bad_cost, "invalid replay result: " + "; ".join(errors)
        )
    if result["soft_timeout_triggered"]:
        timeout = result["soft_timeout_sec"]
        elapsed = result["soft_timeout_elapsed_sec"]
        return ReplayEvaluation(
            "soft_timeout",
            SOFT_TIMEOUT_FAILURE_COST_FACTOR * timeout,
            f"solve soft timeout after {timeout:.6g}s (elapsed={elapsed:.6g}s)",
        )
    if not result["converged"]:
        return ReplayEvaluation("nonconverged", bad_cost, f"not converged: {result['reason']}")

    for field, mean_field, limit, label in (
        (
            "final_true_relative_residual", "final_true_relative_residual_mean",
            max_true_relative_residual, "true relative residual",
        ),
        (
            "final_true_residual_norm", "final_true_residual_norm_mean",
            max_true_residual_norm, "true residual norm",
        ),
    ):
        residual = max((step[field] for step in result["steps"]), default=result[mean_field])
        if limit is not None and residual > limit:
            return ReplayEvaluation(
                "residual_rejected", bad_cost, f"{label} too large: {residual:.6g} > {limit:.6g}"
            )
    return ReplayEvaluation("success", result[objective_name])


def replay_metric_summary(replay_result: dict[str, Any]) -> dict[str, Any]:
    return {
        name: replay_result[name]
        for name in (*REPLAY_METRIC_TYPES, "pc_diagnostics", "schema_version")
        if name in replay_result
    }


def read_replay_result_file(result_path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [f"cannot read replay JSON {result_path}: {exc}"]
    if not isinstance(value, dict):
        return {}, ["replay result must be a JSON object"]
    return value, []
