from __future__ import annotations

from ksptune.replay import (
    replay_failure_reason,
    replay_metric_summary,
    replay_objective_value,
    validate_replay_result,
)
from ksptune.tuning_runs import BAD_COST


def valid_replay_result() -> dict:
    return {
        "objective_time_sec_median": 0.1,
        "total_wall_time_sec": 0.2,
        "matrix_load_time_sec": 0.03,
        "solver_setup_time_sec": 0.04,
        "solve_time_sec_total": 0.18,
        "solve_time_sec_mean": 0.09,
        "solve_time_sec_median": 0.1,
        "solve_count": 2,
        "iterations_total": 5,
        "snapshots": 2,
        "steps": [],
        "nullspace": "field",
        "field_nullspace_index": 1,
        "field_nullspace_block_size": 2,
        "converged": True,
        "reason_code": 2,
        "peak_memory_megabytes_max": 100.0,
        "peak_memory_megabytes_sum": 100.0,
        "initial_true_residual_norm_mean": 1.0,
        "initial_true_relative_residual_mean": 1.0,
        "final_true_residual_norm_mean": 1.0e-8,
        "final_true_relative_residual_mean": 1.0e-9,
    }


def test_validate_replay_result_accepts_clear_metric_names() -> None:
    assert validate_replay_result(valid_replay_result()) == []


def test_validate_replay_result_reports_missing_metric() -> None:
    replay_result = valid_replay_result()
    replay_result.pop("total_wall_time_sec")

    errors = validate_replay_result(replay_result)

    assert "missing replay metric: total_wall_time_sec" in errors


def test_replay_objective_uses_bad_cost_for_non_converged_result() -> None:
    replay_result = valid_replay_result()
    replay_result["converged"] = False
    replay_result["reason"] = "KSP_DIVERGED_ITS"
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert replay_failure_reason(replay_record) == "not converged: KSP_DIVERGED_ITS"
    assert replay_objective_value(
        replay_record,
        objective_name="objective_time_sec_median",
        bad_cost=BAD_COST,
    ) == BAD_COST


def test_replay_failure_reason_explains_petsc_signal_returncode() -> None:
    replay_record = {
        "returncode": 59,
        "replay_result": {},
        "schema_errors": [],
    }

    assert replay_failure_reason(replay_record) == (
        "replay command returned 59 (PETSc caught a signal, commonly a SIGSEGV)"
    )


def test_replay_metric_summary_keeps_diagnostics() -> None:
    summary = replay_metric_summary(valid_replay_result())

    assert summary["total_wall_time_sec"] == 0.2
    assert summary["solve_time_sec_total"] == 0.18
    assert summary["solve_count"] == 2
    assert summary["nullspace"] == "field"
    assert summary["field_nullspace_index"] == 1
    assert summary["peak_memory_megabytes_max"] == 100.0
    assert summary["final_true_relative_residual_mean"] == 1.0e-9


def test_replay_metric_summary_derives_solve_total_and_count_for_old_results() -> None:
    replay_result = valid_replay_result()
    replay_result.pop("solve_time_sec_total")
    replay_result.pop("solve_count")
    replay_result["steps"] = [
        {"solve_time_sec": 0.2},
        {"solve_time_sec": 0.3},
    ]

    summary = replay_metric_summary(replay_result)

    assert summary["solve_time_sec_total"] == 0.5
    assert summary["solve_count"] == 2
