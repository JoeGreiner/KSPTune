from __future__ import annotations

from ksptune.replay import (
    build_replay_command,
    build_replay_server_command,
    replay_failure_reason,
    replay_metric_summary,
    replay_objective_value,
    validate_replay_result,
)
from ksptune.tuning_runs import BAD_COST


def test_build_replay_command_accepts_launcher_arguments() -> None:
    command = build_replay_command(
        replay_binary="ksptune-petsc-replay",
        snapshot_collection_path="snapshots.csv",
        replay_result_path="result.json",
        petsc_options=["-ksp_type", "gmres"],
        mpiexec="srun",
        mpiexec_args=["--exclusive", "--cpu-bind=cores"],
        mpi_processes=16,
    )

    assert command[:5] == ["srun", "--exclusive", "--cpu-bind=cores", "-n", "16"]
    assert "-snapshot_collection" in command
    assert "-ksp_type" in command


def test_build_replay_server_command_uses_replay_server_mode() -> None:
    command = build_replay_server_command(
        replay_binary="ksptune-petsc-replay",
        snapshot_collection_path="snapshots.csv",
        mpiexec="mpiexec",
        mpi_processes=2,
        extra_replay_options=["-replay_nullspace", "from-metadata"],
        cache_memory_mb=2048,
    )

    assert command[:3] == ["mpiexec", "-n", "2"]
    assert "-replay_server" in command
    assert "-replay_cache_memory_mb" in command
    assert "2048" in command


def valid_replay_result() -> dict:
    return {
        "objective_time_sec_median": 0.1,
        "total_wall_time_sec": 0.2,
        "matrix_load_time_sec": 0.03,
        "matrix_cache_hits": 1,
        "matrix_cache_misses": 1,
        "solver_setup_time_sec": 0.04,
        "solver_setup_time_sec_actual": 0.02,
        "solver_setup_time_sec_logical": 0.04,
        "ksp_setup_cache_hits": 1,
        "ksp_setup_cache_misses": 1,
        "solve_time_sec_total": 0.18,
        "solve_time_sec_mean": 0.09,
        "solve_time_sec_median": 0.1,
        "solve_time_sec_min": 0.08,
        "solve_time_sec_max": 0.1,
        "solve_time_sec_stddev": 0.01,
        "solve_time_sec_range": 0.02,
        "solve_mpi_message_count": 28.0,
        "solve_mpi_message_bytes": 224.0,
        "solve_mpi_message_bytes_mean": 8.0,
        "solve_mpi_reduction_count": 44.0,
        "solve_count": 2,
        "iterations_total": 5,
        "snapshots": 2,
        "steps": [],
        "nullspace": "field",
        "field_nullspace_index": 1,
        "field_nullspace_block_size": 2,
        "converged": True,
        "reason_code": 2,
        "peak_memory_mb_per_rank": [100.0],
        "peak_memory_mb_max_per_rank": 100.0,
        "peak_memory_mb_mean_per_rank": 100.0,
        "peak_memory_mb_sum": 100.0,
        "peak_memory_rank_count": 1,
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
    assert summary["matrix_cache_hits"] == 1
    assert summary["matrix_cache_misses"] == 1
    assert summary["solver_setup_time_sec"] == 0.04
    assert summary["solver_setup_time_sec_actual"] == 0.02
    assert summary["solver_setup_time_sec_logical"] == 0.04
    assert summary["ksp_setup_cache_hits"] == 1
    assert summary["ksp_setup_cache_misses"] == 1
    assert summary["solve_time_sec_total"] == 0.18
    assert summary["solve_time_sec_min"] == 0.08
    assert summary["solve_time_sec_max"] == 0.1
    assert summary["solve_time_sec_stddev"] == 0.01
    assert summary["solve_time_sec_range"] == 0.02
    assert summary["solve_mpi_message_count"] == 28.0
    assert summary["solve_mpi_message_bytes"] == 224.0
    assert summary["solve_mpi_message_bytes_mean"] == 8.0
    assert summary["solve_mpi_reduction_count"] == 44.0
    assert summary["solve_count"] == 2
    assert summary["nullspace"] == "field"
    assert summary["field_nullspace_index"] == 1
    assert summary["peak_memory_mb_max_per_rank"] == 100.0
    assert summary["peak_memory_mb_mean_per_rank"] == 100.0
    assert summary["peak_memory_rank_count"] == 1
    assert summary["final_true_relative_residual_mean"] == 1.0e-9


def test_replay_metric_summary_derives_solve_total_and_count_for_old_results() -> None:
    replay_result = valid_replay_result()
    replay_result.pop("solve_time_sec_total")
    replay_result.pop("solve_time_sec_min")
    replay_result.pop("solve_time_sec_max")
    replay_result.pop("solve_time_sec_stddev")
    replay_result.pop("solve_time_sec_range")
    replay_result.pop("solve_count")
    replay_result["steps"] = [
        {"solve_time_sec": 0.2},
        {"solve_time_sec": 0.3},
    ]

    summary = replay_metric_summary(replay_result)

    assert summary["solve_time_sec_total"] == 0.5
    assert summary["solve_time_sec_min"] == 0.2
    assert summary["solve_time_sec_max"] == 0.3
    assert summary["solve_time_sec_stddev"] == 0.04999999999999999
    assert summary["solve_time_sec_range"] == 0.09999999999999998
    assert summary["solve_count"] == 2


def test_replay_metric_summary_derives_memory_rank_aggregates() -> None:
    replay_result = valid_replay_result()
    for key in (
        "peak_memory_mb_sum",
        "peak_memory_mb_max_per_rank",
        "peak_memory_mb_mean_per_rank",
        "peak_memory_rank_count",
    ):
        replay_result.pop(key)
    replay_result["peak_memory_mb_per_rank"] = [64.0, 128.0, 96.0]

    summary = replay_metric_summary(replay_result)

    assert summary["peak_memory_mb_sum"] == 288.0
    assert summary["peak_memory_mb_max_per_rank"] == 128.0
    assert summary["peak_memory_mb_mean_per_rank"] == 96.0
    assert summary["peak_memory_rank_count"] == 3
