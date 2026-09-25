from __future__ import annotations

from pathlib import Path

from ksptune.replay import (
    build_replay_command,
    build_replay_server_command,
    ReplayServerProcess,
    run_replay_server_for_solver_configuration,
)
from ksptune.replay_results import (
    evaluate_replay,
    replay_metric_summary,
    validate_replay_result,
)
from ksptune.trial_records import (
    BAD_COST,
)


def test_build_replay_command_accepts_launcher_arguments() -> None:
    command = build_replay_command(
        replay_binary="ksptune-petsc-replay",
        snapshot_collection_path="snapshots.csv",
        replay_result_path="result.json",
        petsc_options=["-ksp_type", "gmres"],
        mpiexec="srun",
        mpiexec_args=["--exclusive", "--cpu-bind=cores"],
        mpi_processes=16,
        soft_timeout_sec=12.5,
    )

    assert command[:5] == ["srun", "--exclusive", "--cpu-bind=cores", "-n", "16"]
    assert "-snapshot_collection" in command
    assert "-replay_soft_timeout_sec" in command
    assert "12.5" in command
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


def test_run_replay_server_for_solver_configuration_accepts_threads_per_rank() -> None:
    class DummyReplayWorkerPool:
        def run_solver_configuration(self, **kwargs):
            return kwargs

    record = run_replay_server_for_solver_configuration(
        replay_server=DummyReplayWorkerPool(),
        replay_binary="ksptune-petsc-replay",
        snapshot_collection_path="snapshots.csv",
        replay_result_path="result.json",
        petsc_options=["-ksp_type", "cg"],
        mpiexec="mpiexec",
        mpi_processes=2,
        threads_per_rank=4,
        soft_timeout_sec=9.0,
    )

    assert record["reproduction_command"][:3] == ["mpiexec", "-n", "2"]
    assert "-ksp_type" in record["reproduction_command"]
    assert "-replay_soft_timeout_sec" in record["reproduction_command"]
    assert record["soft_timeout_sec"] == 9.0


def test_replay_server_reports_early_exit_with_stderr(tmp_path: Path) -> None:
    replay_binary = tmp_path / "failing-replay-server"
    replay_binary.write_text("#!/bin/sh\necho missing binary >&2\nexit 42\n", encoding="utf-8")
    replay_binary.chmod(0o755)
    server = ReplayServerProcess(
        replay_binary=replay_binary,
        snapshot_collection_path=tmp_path / "snapshots.csv",
        log_directory=tmp_path / "logs",
    )

    try:
        record = server.run_solver_configuration(
            replay_result_path=tmp_path / "result.json",
            petsc_options=["-ksp_type", "gmres"],
            hard_timeout_sec=1,
            reproduction_command=["reproduce"],
        )
    finally:
        server.close(kill=True)

    assert record["returncode"] == 42
    assert "returncode=42" in record["failure_reason"]
    assert "missing binary" in record["failure_reason"]
    assert Path(record["stdout_path"]).exists()
    assert "missing binary" in Path(record["stderr_path"]).read_text(encoding="utf-8")


def test_replay_server_keeps_process_after_error_response(tmp_path: Path) -> None:
    replay_binary = tmp_path / "recovering-replay-server.py"
    replay_binary.write_text(
        """#!/usr/bin/env python3
import json
import sys

for index, line in enumerate(sys.stdin, start=1):
    request = json.loads(line)
    if request.get("command") == "shutdown":
        print(json.dumps({"id": request["id"], "returncode": 0}), flush=True)
        break
    if index == 1:
        print(
            json.dumps(
                {
                    "id": request["id"],
                    "returncode": 1,
                    "failure_reason": "PETSc error after resettable request",
                }
            ),
            flush=True,
        )
        continue
    result = REPLAY_RESULT
    with open(request["replay_json_out"], "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    print(json.dumps({"id": request["id"], "returncode": 0}), flush=True)
""".replace(
            "REPLAY_RESULT",
            repr(
                {
                    **valid_replay_result(),
                    "objective_time_sec_median": 0.1,
                    "total_wall_time_sec": 0.1,
                    "converged": True,
                    "reason_code": 2,
                    "solve_time_sec_mean": 0.1,
                    "solve_time_sec_median": 0.1,
                    "solver_setup_time_sec": 0.0,
                    "iterations_total": 1,
                    "snapshots": 1,
                    "steps": [],
                }
            ),
        ),
        encoding="utf-8",
    )
    replay_binary.chmod(0o755)
    server = ReplayServerProcess(
        replay_binary=replay_binary,
        snapshot_collection_path=tmp_path / "snapshots.csv",
        log_directory=tmp_path / "logs",
    )

    try:
        first_record = server.run_solver_configuration(
            replay_result_path=tmp_path / "first.json",
            petsc_options=["-ksp_type", "gmres"],
            hard_timeout_sec=1,
            reproduction_command=["reproduce"],
        )
        second_record = server.run_solver_configuration(
            replay_result_path=tmp_path / "second.json",
            petsc_options=["-ksp_type", "cg"],
            hard_timeout_sec=1,
            reproduction_command=["reproduce"],
        )
    finally:
        server.close()

    assert first_record["returncode"] == 1
    assert first_record["failure_reason"] == "PETSc error after resettable request"
    assert first_record["schema_errors"] == []
    assert second_record["returncode"] == 0
    assert second_record["failure_reason"] is None
    assert second_record["schema_errors"] == []


def test_replay_server_ignores_launcher_noise_on_stdout(tmp_path: Path) -> None:
    replay_binary = tmp_path / "noisy-replay-server.py"
    replay_binary.write_text(
        """#!/usr/bin/env python3
import json
import sys

request = json.loads(sys.stdin.readline())
assert request["soft_timeout_sec"] == 7.5
result = REPLAY_RESULT
with open(request["replay_json_out"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
print("launcher noise before JSON")
print(json.dumps({"id": request["id"], "returncode": 0}), flush=True)
""".replace(
            "REPLAY_RESULT",
            repr(
                {
                    **valid_replay_result(),
                    "objective_time_sec_median": 0.1,
                    "total_wall_time_sec": 0.1,
                    "converged": True,
                    "reason_code": 2,
                    "solve_time_sec_mean": 0.1,
                    "solve_time_sec_median": 0.1,
                    "solver_setup_time_sec": 0.0,
                    "iterations_total": 1,
                    "snapshots": 1,
                    "steps": [],
                }
            ),
        ),
        encoding="utf-8",
    )
    replay_binary.chmod(0o755)
    server = ReplayServerProcess(
        replay_binary=replay_binary,
        snapshot_collection_path=tmp_path / "snapshots.csv",
        log_directory=tmp_path / "logs",
    )

    try:
        record = server.run_solver_configuration(
            replay_result_path=tmp_path / "result.json",
            petsc_options=["-ksp_type", "gmres"],
            hard_timeout_sec=1,
            soft_timeout_sec=7.5,
            reproduction_command=["reproduce"],
        )
    finally:
        server.close(kill=True)

    assert record["returncode"] == 0
    assert record["failure_reason"] is None
    assert record["schema_errors"] == []
    assert "launcher noise before JSON" in record["stdout"]
    assert "launcher noise before JSON" in Path(record["stdout_path"]).read_text(encoding="utf-8")


def test_replay_server_uses_startup_timeout_only_for_first_request(tmp_path: Path) -> None:
    replay_binary = tmp_path / "startup-aware-replay-server.py"
    replay_binary.write_text(
        """#!/usr/bin/env python3
import json
import sys
import time

for index, line in enumerate(sys.stdin, start=1):
    request = json.loads(line)
    if index == 1:
        result = REPLAY_RESULT
        with open(request["replay_json_out"], "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        print(json.dumps({"id": request["id"], "returncode": 0}), flush=True)
    else:
        time.sleep(5)
""".replace(
            "REPLAY_RESULT",
            repr(
                {
                    **valid_replay_result(),
                    "objective_time_sec_median": 0.1,
                    "total_wall_time_sec": 0.1,
                    "converged": True,
                    "reason_code": 2,
                    "solve_time_sec_mean": 0.1,
                    "solve_time_sec_median": 0.1,
                    "solver_setup_time_sec": 0.0,
                    "iterations_total": 1,
                    "snapshots": 1,
                    "steps": [],
                }
            ),
        ),
        encoding="utf-8",
    )
    replay_binary.chmod(0o755)
    server = ReplayServerProcess(
        replay_binary=replay_binary,
        snapshot_collection_path=tmp_path / "snapshots.csv",
        log_directory=tmp_path / "logs",
    )

    try:
        first_record = server.run_solver_configuration(
            replay_result_path=tmp_path / "first.json",
            petsc_options=["-ksp_type", "gmres"],
            hard_timeout_sec=0.1,
            replay_startup_timeout_sec=2.0,
            reproduction_command=["reproduce"],
        )
        second_record = server.run_solver_configuration(
            replay_result_path=tmp_path / "second.json",
            petsc_options=["-ksp_type", "gmres"],
            hard_timeout_sec=0.1,
            replay_startup_timeout_sec=2.0,
            reproduction_command=["reproduce"],
        )
    finally:
        server.close(kill=True)

    assert first_record["failure_reason"] is None
    assert first_record["used_startup_timeout"] is True
    assert first_record["effective_hard_timeout_sec"] == 2.0
    assert second_record["used_startup_timeout"] is False
    assert second_record["effective_hard_timeout_sec"] == 0.1
    assert second_record["failure_reason"] == "replay server timed out after 0.1 seconds"


def test_replay_server_reports_exit_after_stdout_noise_as_exit(tmp_path: Path) -> None:
    replay_binary = tmp_path / "noisy-failing-replay-server.py"
    replay_binary.write_text(
        """#!/usr/bin/env python3
import sys

print("launcher noise before failure", flush=True)
print("PETSc failed after launcher noise", file=sys.stderr, flush=True)
sys.exit(59)
""",
        encoding="utf-8",
    )
    replay_binary.chmod(0o755)
    server = ReplayServerProcess(
        replay_binary=replay_binary,
        snapshot_collection_path=tmp_path / "snapshots.csv",
        log_directory=tmp_path / "logs",
    )

    try:
        record = server.run_solver_configuration(
            replay_result_path=tmp_path / "result.json",
            petsc_options=["-ksp_type", "gmres"],
            hard_timeout_sec=15,
            reproduction_command=["reproduce"],
        )
    finally:
        server.close(kill=True)

    assert record["returncode"] == 59
    assert "timed out" not in record["failure_reason"]
    assert "returncode=59" in record["failure_reason"]
    assert "PETSc failed after launcher noise" in record["failure_reason"]
    assert "PETSc failed after launcher noise" in record["stderr"]
    assert "launcher noise before failure" in record["stdout"]
    assert "PETSc failed after launcher noise" in Path(record["stderr_path"]).read_text(
        encoding="utf-8"
    )
    assert "launcher noise before failure" in Path(record["stdout_path"]).read_text(
        encoding="utf-8"
    )


def valid_replay_result() -> dict:
    return {
        "matrix_prepare_time_sec": 0.001229000000000001,
        "vector_prepare_time_sec": 0.00014199999999999977,
        "nullspace_time_sec": 0.0,
        "true_residual_time_sec": 2.3999999999999716e-05,
        "use_initial_guess": True,
        "soft_timeout_sec": -1.0,
        "soft_timeout_elapsed_sec": 0.0,
        "soft_timeout_triggered": False,
        "replay_cache_memory_mb": 0.0,
        "matrix_cache_memory_mb": 0.0,
        "vector_cache_memory_mb": 0.0,
        "replay_cache_memory_limit_mb": 0.0,
        "replay_cache_evictions": 0,
        "iterations_median": 1.0,
        "reason": "KSP_CONVERGED_ATOL",
        "repeat": 1,
        "warmup": 0,
        "nullspace_source": "from-metadata",
        "nullspace_kind": "none",
        "nullspace_actions": "none",
        "matrix_nullspace_attached_count": 0,
        "transpose_nullspace_attached_count": 0,
        "near_nullspace_attached_count": 0,
        "rhs_nullspace_removed_count": 0,
        "rhs_nullspace_removed_component_norm_max": 0.0,
        "rhs_nullspace_removed_component_relative_norm_max": 0.0,
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "schema_version": 1,
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
        "solve_time_sec_sem": 0.01,
        "solve_time_sec_relative_sem": 0.11111111111111112,
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
        "rss_request_start_mb_sum": 100.0,
        "rss_request_end_mb_sum": 120.0,
        "rss_request_peak_sample_mb_sum": 160.0,
        "rss_request_delta_mb_sum": 20.0,
        "rss_setup_delta_mb_sum": 30.0,
        "rss_solve_delta_mb_sum": 4.0,
        "rss_peak_sample_mb_max_rank": 80.0,
        "rss_peak_sample_mb_max_node": 160.0,
        "initial_true_residual_norm_mean": 1.0,
        "initial_true_relative_residual_mean": 1.0,
        "final_true_residual_norm_mean": 1.0e-8,
        "final_true_relative_residual_mean": 1.0e-9,
        "initial_ksp_residual_norm_mean": 0.75,
        "final_ksp_residual_norm_mean": 1.0e-8,
        "ksp_rtol_reference_norm_mean": 1.25,
        "ksp_convergence_threshold_norm_mean": 1.25e-8,
        "final_ksp_relative_residual_norm_mean": 8.0e-9,
        "ksp_residual_norm_type": "preconditioned",
        "ksp_rtol_reference_source": "petsc-rnorm0",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 1.0e-50,
        "ksp_dtol": 1.0e4,
        "ksp_max_it": 10000,
        "pc_diagnostics": [
            {
                "pc_type": "gamg",
                "setup_index": 1,
                "setup_key": "gamg-key",
                "metrics": {
                    "levels": 4,
                    "grid_complexity": 1.25,
                    "operator_complexity": 1.75,
                    "finest_rows": 1000.0,
                    "finest_nonzeros": 7000.0,
                    "coarsest_rows": 25.0,
                    "coarsest_nonzeros": 175.0,
                    "total_rows": 1250.0,
                    "total_nonzeros": 12250.0,
                    "interpolation_nonzeros": 4000.0,
                    "max_operator_nonzeros_per_row": 12.25,
                    "max_interpolation_nonzeros_per_row": 4.0,
                    "min_active_ranks": 2.0,
                },
                "string_metrics": {},
                "levels": [
                    {
                        "level": 3,
                        "level_from_finest": 0,
                        "metrics": {
                            "rows": 1000.0,
                            "nonzeros": 7000.0,
                            "avg_nonzeros_per_row": 7.0,
                            "active_ranks": 2.0,
                            "interpolation_nonzeros": 4000.0,
                            "interpolation_avg_nonzeros_per_row": 4.0,
                        },
                        "string_metrics": {},
                    }
                ],
            },
            {
                "pc_type": "hypre",
                "setup_index": 2,
                "setup_key": "hypre-key",
                "metrics": {
                    "levels": 5,
                    "grid_complexity": 1.1,
                    "operator_complexity": 1.4,
                    "interpolation_complexity": 0.3,
                    "finest_rows": 1000.0,
                    "finest_nonzeros": 7000.0,
                    "coarsest_rows": 10.0,
                    "coarsest_nonzeros": 30.0,
                    "total_rows": 1100.0,
                    "total_nonzeros": 9800.0,
                    "interpolation_nonzeros": 2100.0,
                    "max_row_nonzeros": 42.0,
                    "offdiag_nonzeros": 500.0,
                    "max_operator_nonzeros_per_row": 9.8,
                    "max_interpolation_nonzeros_per_row": 3.0,
                },
                "string_metrics": {},
                "levels": [],
            },
        ],
    }


def test_validate_replay_result_accepts_clear_metric_names() -> None:
    assert validate_replay_result(valid_replay_result()) == []


def test_validate_replay_result_accepts_unavailable_solve_time_sem() -> None:
    replay_result = valid_replay_result()
    replay_result["solve_count"] = 1
    replay_result["solve_time_sec_sem"] = None
    replay_result["solve_time_sec_relative_sem"] = None

    assert validate_replay_result(replay_result) == []


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

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == "not converged: KSP_DIVERGED_ITS"
    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == BAD_COST
    )


def test_replay_objective_uses_scaled_cost_for_soft_timeout() -> None:
    replay_result = valid_replay_result()
    replay_result["converged"] = False
    replay_result["reason"] = "KSP_DIVERGED_ITS"
    replay_result["soft_timeout_triggered"] = True
    replay_result["soft_timeout_sec"] = 12.5
    replay_result["soft_timeout_elapsed_sec"] = 12.8
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == (
        "solve soft timeout after 12.5s (elapsed=12.8s)"
    )
    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == 18.75
    )


def test_replay_objective_uses_scaled_cost_for_hard_timeout() -> None:
    replay_record = {
        "returncode": None,
        "failure_reason": "replay server timed out after 600.0 seconds",
        "failure_kind": "hard_timeout",
        "effective_hard_timeout_sec": 600.0,
        "replay_result": {},
        "schema_errors": [],
    }

    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == 1200.0
    )


def test_replay_objective_uses_scaled_cost_for_oom_kill() -> None:
    replay_record = {
        "returncode": 137,
        "failure_reason": "replay server exited before completing request (returncode=137)",
        "effective_hard_timeout_sec": 600.0,
        "replay_result": {},
        "schema_errors": [],
    }

    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == 6000.0
    )


def test_replay_objective_uses_bad_cost_for_large_true_relative_residual() -> None:
    replay_result = valid_replay_result()
    replay_result["final_true_relative_residual_mean"] = 2.235
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == (
        "true relative residual too large: 2.235 > 0.0001"
    )
    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == BAD_COST
    )


def test_replay_objective_accepts_true_relative_residual_below_default_gate() -> None:
    replay_result = valid_replay_result()
    replay_result["final_true_relative_residual_mean"] = 4.5e-5
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason is None
    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == replay_result["objective_time_sec_median"]
    )


def test_replay_objective_uses_bad_cost_for_large_true_residual_norm() -> None:
    replay_result = valid_replay_result()
    replay_result["final_true_relative_residual_mean"] = 4.5e-5
    replay_result["final_true_residual_norm_mean"] = 2.0e-4
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == ("true residual norm too large: 0.0002 > 0.0001")
    assert (
        evaluate_replay(
            replay_record,
            objective_name="objective_time_sec_median",
            bad_cost=BAD_COST,
        ).objective_value
        == BAD_COST
    )


def test_replay_failure_reason_checks_step_max_before_mean_residual() -> None:
    replay_result = valid_replay_result()
    replay_result["final_true_relative_residual_mean"] = 1.0e-9
    replay_result["steps"] = [
        {
            "solve_time_sec": 0.1,
            "final_true_residual_norm": 1e-10,
            "final_true_relative_residual": 2.0e-6,
        },
        {
            "solve_time_sec": 0.1,
            "final_true_residual_norm": 1e-10,
            "final_true_relative_residual": 2.235,
        },
    ]
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == (
        "true relative residual too large: 2.235 > 0.0001"
    )


def test_replay_failure_reason_checks_step_max_before_mean_residual_norm() -> None:
    replay_result = valid_replay_result()
    replay_result["final_true_residual_norm_mean"] = 1.0e-9
    replay_result["steps"] = [
        {
            "solve_time_sec": 0.1,
            "final_true_relative_residual": 1e-10,
            "final_true_residual_norm": 2.0e-6,
        },
        {
            "solve_time_sec": 0.1,
            "final_true_relative_residual": 1e-10,
            "final_true_residual_norm": 2.0e-4,
        },
    ]
    replay_record = {
        "returncode": 0,
        "replay_result": replay_result,
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == ("true residual norm too large: 0.0002 > 0.0001")


def test_replay_failure_reason_explains_petsc_signal_returncode() -> None:
    replay_record = {
        "returncode": 59,
        "replay_result": {},
        "schema_errors": [],
    }

    assert evaluate_replay(replay_record, bad_cost=BAD_COST).failure_reason == (
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
    assert summary["solve_time_sec_sem"] == 0.01
    assert summary["solve_time_sec_relative_sem"] == 0.11111111111111112
    assert summary["solve_time_sec_range"] == 0.02
    assert summary["solve_mpi_message_count"] == 28.0
    assert summary["solve_mpi_message_bytes"] == 224.0
    assert summary["solve_mpi_message_bytes_mean"] == 8.0
    assert summary["solve_mpi_reduction_count"] == 44.0
    assert summary["solve_count"] == 2
    assert summary["nullspace"] == "field"
    assert summary["field_nullspace_index"] == 1
    assert summary["rss_request_peak_sample_mb_sum"] == 160.0
    assert summary["rss_request_delta_mb_sum"] == 20.0
    assert summary["rss_setup_delta_mb_sum"] == 30.0
    assert summary["rss_solve_delta_mb_sum"] == 4.0
    assert summary["rss_peak_sample_mb_max_rank"] == 80.0
    assert summary["rss_peak_sample_mb_max_node"] == 160.0
    assert summary["initial_true_residual_norm_mean"] == 1.0
    assert summary["initial_true_relative_residual_mean"] == 1.0
    assert summary["final_true_residual_norm_mean"] == 1.0e-8
    assert summary["final_true_relative_residual_mean"] == 1.0e-9
    assert summary["initial_ksp_residual_norm_mean"] == 0.75
    assert summary["final_ksp_residual_norm_mean"] == 1.0e-8
    assert summary["ksp_rtol_reference_norm_mean"] == 1.25
    assert summary["ksp_convergence_threshold_norm_mean"] == 1.25e-8
    assert summary["final_ksp_relative_residual_norm_mean"] == 8.0e-9
    assert summary["ksp_residual_norm_type"] == "preconditioned"
    assert summary["ksp_rtol_reference_source"] == "petsc-rnorm0"
    assert summary["ksp_rtol"] == 1.0e-8
    assert summary["ksp_atol"] == 1.0e-50
    assert summary["ksp_dtol"] == 1.0e4
    assert summary["ksp_max_it"] == 10000
    assert summary["pc_diagnostics"][0]["pc_type"] == "gamg"
    assert summary["pc_diagnostics"][0]["metrics"]["operator_complexity"] == 1.75
    assert summary["pc_diagnostics"][0]["levels"][0]["metrics"]["nonzeros"] == 7000.0
    assert summary["pc_diagnostics"][1]["pc_type"] == "hypre"
    assert summary["pc_diagnostics"][1]["metrics"]["interpolation_complexity"] == 0.3
    assert summary["pc_diagnostics"][1]["metrics"]["max_row_nonzeros"] == 42.0
