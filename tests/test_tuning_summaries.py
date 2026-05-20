from __future__ import annotations

import csv
from pathlib import Path

from ksptune.tuning_runs import BAD_COST, append_json_line, write_tuning_summary_outputs


def test_write_tuning_summary_outputs_ranks_successful_trials_first(tmp_path: Path) -> None:
    append_json_line(
        tmp_path / "solver_configuration_trials.jsonl",
        {
            "trial_number": 1,
            "smac_configuration_tag": "failed1",
            "objective_name": "objective_sec",
            "objective_value": BAD_COST,
            "failure_reason": "not converged: KSP_DIVERGED_ITS",
            "converged": False,
            "total_wall_time_seconds": 0.20,
            "peak_memory_megabytes_max": 256.0,
            "final_true_relative_residual_mean": 0.5,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "none"},
            "petsc_options": ["-ksp_type", "cg", "-pc_type", "none"],
        },
    )
    append_json_line(
        tmp_path / "solver_configuration_trials.jsonl",
        {
            "trial_number": 2,
            "smac_configuration_tag": "best02",
            "objective_name": "objective_sec",
            "objective_value": 0.05,
            "failure_reason": None,
            "converged": True,
            "total_wall_time_seconds": 0.08,
            "peak_memory_megabytes_max": 128.0,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "petsc_options": ["-ksp_type", "cg", "-pc_type", "jacobi"],
        },
    )

    summary = write_tuning_summary_outputs(
        output_directory=tmp_path,
        objective_name="objective_sec",
        status="completed",
    )

    assert summary["trial_count"] == 2
    assert summary["successful_trial_count"] == 1
    assert summary["failed_trial_count"] == 1
    assert summary["best_trial"]["trial_number"] == 2
    assert summary["best_trial"]["smac_configuration_tag"] == "best02"
    assert summary["peak_memory_megabytes_max"] == 256.0
    assert (tmp_path / "tuning_summary.yaml").exists()

    ranking_rows = list(csv.DictReader((tmp_path / "solver_configuration_rankings.csv").open()))
    assert [row["trial_number"] for row in ranking_rows] == ["2", "1"]
    assert [row["smac_configuration_tag"] for row in ranking_rows] == ["best02", "failed1"]
    assert ranking_rows[0]["petsc_options_text"] == "-ksp_type cg -pc_type jacobi"


def test_write_tuning_summary_outputs_handles_all_failed_trials(tmp_path: Path) -> None:
    append_json_line(
        tmp_path / "solver_configuration_trials.jsonl",
        {
            "trial_number": 1,
            "objective_name": "objective_sec",
            "objective_value": BAD_COST,
            "failure_reason": "subprocess failed with returncode 1",
            "converged": False,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "none"},
            "petsc_options": ["-ksp_type", "cg", "-pc_type", "none"],
        },
    )

    summary = write_tuning_summary_outputs(
        output_directory=tmp_path,
        objective_name="objective_sec",
        status="completed",
    )

    assert summary["successful_trial_count"] == 0
    assert summary["failed_trial_count"] == 1
    assert summary["best_trial"] is None
    assert summary["fastest_converged_trial"] is None
