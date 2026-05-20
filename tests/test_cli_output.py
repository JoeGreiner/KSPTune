from __future__ import annotations

import re

from ksptune import cli as cli_module
from ksptune.cli import print_tune_progress, build_parser

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def test_tune_parser_accepts_quiet_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--dry-run",
            "--quiet",
        ]
    )

    assert args.quiet is True
    assert args.trials is None
    assert args.color == "auto"
    assert args.workers == 1
    assert args.nullspace == "from-metadata"


def test_tune_parser_accepts_run_until_stopped_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--replay-binary",
            "ksptune-petsc-replay",
            "--run-until-stopped",
        ]
    )

    assert args.run_until_stopped is True


def test_tune_parser_accepts_color_and_worker_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--color",
            "always",
            "--workers",
            "3",
        ]
    )

    assert args.color == "always"
    assert args.workers == 3


def test_tune_parser_accepts_nullspace_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--nullspace",
            "field:1,block_size=2",
        ]
    )

    assert args.nullspace == "field:1,block_size=2"


def test_analyze_snapshots_parser_accepts_petsc_options() -> None:
    args = build_parser().parse_args(
        [
            "analyze-snapshots",
            "snapshots",
            "--petsc-analysis-binary",
            "ksptune-petsc-snapshot-analysis",
            "--np",
            "2",
            "--nullspace",
            "field:1,block_size=2",
        ]
    )

    assert args.snapshot_directory_argument == "snapshots"
    assert args.snapshot_directory is None
    assert args.petsc_analysis_binary == "ksptune-petsc-snapshot-analysis"
    assert args.np == 2
    assert args.nullspace == "field:1,block_size=2"


def test_analyse_snapshots_alias_accepts_positional_directory() -> None:
    args = build_parser().parse_args(["analyse-snapshots", "."])

    assert args.command == "analyse-snapshots"
    assert args.snapshot_directory_argument == "."
    assert args.snapshot_directory is None


def test_analyze_snapshots_parser_keeps_snapshot_directory_option() -> None:
    args = build_parser().parse_args(
        [
            "analyze-snapshots",
            "--snapshot-directory",
            "snapshots",
        ]
    )

    assert args.snapshot_directory == "snapshots"
    assert args.snapshot_directory_argument is None


def testcolor_enabled_respects_mode_and_no_color(monkeypatch) -> None:
    class TtyStdout:
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            return None

    monkeypatch.setattr(cli_module.sys, "stdout", TtyStdout())
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert cli_module.color_enabled("auto") is True
    assert cli_module.color_enabled("always") is True
    assert cli_module.color_enabled("never") is False

    monkeypatch.setenv("NO_COLOR", "1")

    assert cli_module.color_enabled("auto") is False
    assert cli_module.color_enabled("always") is True


def test_tune_command_runs_until_stopped_by_default(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--dry-run",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(**kwargs):
        captured.update(kwargs)
        return {"status": "dry-run"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["trials"] is None
    assert captured["run_until_stopped"] is True


def test_tune_command_uses_fixed_trials_when_requested(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--trials",
            "5",
            "--workers",
            "4",
            "--nullspace",
            "field:1,block_size=2",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(**kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["trials"] == 5
    assert captured["workers"] == 4
    assert captured["nullspace"] == "field:1,block_size=2"
    assert captured["run_until_stopped"] is False


def test_tune_command_quiet_mode_does_not_emit_color(monkeypatch, capsys) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.hypre-basic",
            "--output-directory",
            "run",
            "--dry-run",
            "--quiet",
            "--color",
            "always",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(**kwargs):
        captured.update(kwargs)
        return {"status": "dry-run"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0

    output = capsys.readouterr().out
    assert captured["progress_callback"] is None
    assert "\x1b[" not in output
    assert "status: dry-run" in output


def test_tune_progress_prints_run_until_stopped_header(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_started",
            "snapshot_directory_path": "snapshots.csv",
            "parameter_search_space_name": "petsc.hypre-basic",
            "output_directory": "run",
            "trials": 20,
            "repeat": 1,
            "warmup": 0,
            "run_until_stopped": True,
            "replay_binary": "ksptune-petsc-replay",
            "replay_binary_auto_detected": True,
            "nullspace_configuration": {
                "mode": "field",
                "label": "field:1",
                "field_index": 1,
                "block_size": 2,
            },
            "dry_run": False,
        }
    )

    output = capsys.readouterr().out
    assert "mode: until stopped" in output
    assert "workers=1" in output
    assert "replay: ksptune-petsc-replay (auto)" in output
    assert "nullspace: field:1 (field=1, block_size=2)" in output
    assert "Ctrl+C" in output
    assert "SMAC config IDs match tags in trial headings" in output
    assert not output.startswith("\n")
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_colorizes_run_header(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_started",
            "snapshot_directory_path": "snapshots.csv",
            "parameter_search_space_name": "petsc.hypre-basic",
            "output_directory": "run",
            "trials": 20,
            "repeat": 1,
            "warmup": 0,
            "run_until_stopped": True,
            "replay_binary": "ksptune-petsc-replay",
            "replay_binary_auto_detected": True,
            "dry_run": False,
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    plain_output = strip_ansi(output)
    assert "\033[1mKSPTune tuning run\033[0m" in output
    assert "\033[36msnapshots.csv\033[0m" in output
    assert "\033[33mCtrl+C\033[0m" in output
    assert "mode: until stopped" in plain_output
    assert "replay: ksptune-petsc-replay (auto)" in plain_output


def test_tune_progress_prints_readable_trial_line(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 0.1234,
            "total_wall_time_seconds": 0.15,
            "solver_setup_time_seconds": 0.03,
            "solve_time_seconds_median": 0.10,
            "peak_memory_megabytes_max": 151.25,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "abc123",
            "best_objective_value_so_far": 0.101,
            "is_new_best_so_far": False,
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\ntrial 003/020 [abc123] ok  objective: 0.1234  (best: 0.101)")
    assert "  solver: ksp=cg pc=jacobi" in output
    assert "  runtime: solve=0.100s  setup=0.030s  wall=0.150s" in output
    assert "mem=151.2MB" in output
    assert "true_rel_res=1.000e-09" in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_prints_total_and_average_solve_time(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 4.12,
            "total_wall_time_seconds": 250.0,
            "solver_setup_time_seconds": 0.03,
            "solve_time_seconds_total": 232.1,
            "solve_time_seconds_mean": 4.12,
            "solve_time_seconds_median": 3.9,
            "solve_count": 56,
            "peak_memory_megabytes_max": 151.25,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "abc123",
            "best_objective_value_so_far": 4.0,
            "is_new_best_so_far": False,
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert "  runtime: solve=232.100s (avg=4.120s, n=56)  setup=0.030s  wall=250.000s" in output


def test_tune_progress_colorizes_successful_trial_line(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 0.1234,
            "total_wall_time_seconds": 0.15,
            "solver_setup_time_seconds": 0.03,
            "solve_time_seconds_median": 0.10,
            "peak_memory_megabytes_max": 151.25,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "abc123",
            "best_objective_value_so_far": 0.101,
            "is_new_best_so_far": False,
            "failure_reason": None,
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    plain_output = strip_ansi(output)
    assert "\033[36m[abc123]\033[0m" in output
    assert "\033[32mok\033[0m" in output
    assert "\033[1m\033[33m0.1234\033[0m" in output
    assert "\033[1m\033[36m0.101\033[0m" in output
    assert plain_output.startswith("\ntrial 003/020 [abc123] ok  objective: 0.1234  (best: 0.101)")
    assert plain_output.endswith("\n")
    assert not plain_output.endswith("\n\n")


def test_tune_progress_colorizes_new_best_objective_green(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 4,
            "trial_count": 20,
            "objective_value": 0.09,
            "objective_name": "objective_sec",
            "total_wall_time_seconds": 0.15,
            "solver_setup_time_seconds": 0.03,
            "solve_time_seconds_median": 0.10,
            "peak_memory_megabytes_max": 151.25,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "be5701",
            "best_objective_value_so_far": 0.09,
            "is_new_best_so_far": True,
            "failure_reason": None,
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    plain_output = strip_ansi(output)
    assert "\033[1m\033[32m0.09s\033[0m" in output
    assert "\033[1m\033[36m0.09s\033[0m" in output
    assert plain_output.startswith("\ntrial 004/020 [be5701] ok  objective: 0.09s  (best: 0.09s)")


def test_tune_progress_prints_open_ended_trial_count(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": None,
            "objective_value": 0.1234,
            "total_wall_time_seconds": 0.15,
            "solver_setup_time_seconds": 0.03,
            "solve_time_seconds_median": 0.10,
            "peak_memory_megabytes_max": 151.25,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "def456",
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\ntrial 003/... [def456] ok  objective: 0.1234  (best: n/a)")
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_colorizes_failed_trial_line(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 2,
            "objective_value": cli_module.BAD_COST,
            "total_wall_time_seconds": 0.2,
            "peak_memory_megabytes_max": 200.0,
            "final_true_relative_residual_mean": 0.4,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "none"},
            "smac_configuration_tag": "7bb1e4",
            "best_objective_value_so_far": 0.05,
            "is_new_best_so_far": False,
            "failure_reason": "not converged: KSP_DIVERGED_ITS",
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    plain_output = strip_ansi(output)
    assert "\033[36m[7bb1e4]\033[0m" in output
    assert "\033[31mfailed\033[0m" in output
    assert "\033[1m\033[31mbad-cost\033[0m" in output
    assert "\033[1m\033[36m0.05\033[0m" in output
    assert "\033[31mnot converged: KSP_DIVERGED_ITS\033[0m" in output
    assert plain_output.startswith("\ntrial 001/002 [7bb1e4] failed  objective: bad-cost  (best: 0.05)")
    assert "  reason: not converged: KSP_DIVERGED_ITS" in plain_output


def test_tune_progress_prints_stopping_block(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_stopping",
            "trial_count": 7,
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\nstopping: Ctrl+C received after 7 completed trials")
    assert "  writing summary..." in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_colorizes_stopping_block(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_stopping",
            "trial_count": 7,
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    assert "\033[33mCtrl+C\033[0m" in output
    assert strip_ansi(output).startswith("\nstopping: Ctrl+C received after 7 completed trials")


def test_tune_progress_prints_failure_reason(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 2,
            "objective_value": cli_module.BAD_COST,
            "total_wall_time_seconds": 0.2,
            "peak_memory_megabytes_max": 200.0,
            "final_true_relative_residual_mean": 0.4,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "none"},
            "smac_configuration_tag": "7bb1e4",
            "failure_reason": "not converged: KSP_DIVERGED_ITS",
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\ntrial 001/002 [7bb1e4] failed  objective: bad-cost  (best: n/a)")
    assert "  runtime: subprocess=n/a" in output
    assert "  reason: not converged: KSP_DIVERGED_ITS" in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_prints_replay_command_for_subprocess_failure(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 56,
            "trial_count": None,
            "objective_value": cli_module.BAD_COST,
            "subprocess_walltime_seconds": 0.285,
            "solver_configuration": {"ksp_type": "fgmres", "pc_type": "hypre"},
            "smac_configuration_tag": "ee9549",
            "failure_reason": "replay command returned 65",
            "returncode": 65,
            "replay_command_text": (
                "/mnt/work/thirdparty/KSPTune/build/cpp/petsc_replay/ksptune-petsc-replay "
                "-snapshot_collection snapshots.csv "
                "-replay_json_out candidate_ee9549.json "
                "-pc_hypre_boomeramg_strong_threshold 0.393357344158 "
                "-pc_hypre_boomeramg_truncfactor 0.18209642413999999 "
                "-ksp_type fgmres -pc_type hypre"
            ),
        }
    )

    output = capsys.readouterr().out
    assert "  reason: replay command returned 65" in output
    reproduce_lines = [line for line in output.splitlines() if line.startswith("  reproduce: ")]
    assert len(reproduce_lines) == 1
    assert (
        reproduce_lines[0]
        == "  reproduce: "
        "/mnt/work/thirdparty/KSPTune/build/cpp/petsc_replay/ksptune-petsc-replay "
        "-snapshot_collection snapshots.csv "
        "-replay_json_out candidate_ee9549.json "
        "-pc_hypre_boomeramg_strong_threshold 0.393357344158 "
        "-pc_hypre_boomeramg_truncfactor 0.18209642413999999 "
        "-ksp_type fgmres -pc_type hypre"
    )
    assert "-snapshot_collection snapshots.csv" in output
    assert "-pc_type hypre" in output


def test_tune_progress_does_not_print_replay_command_for_timeout(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 45,
            "trial_count": None,
            "objective_value": cli_module.BAD_COST,
            "subprocess_walltime_seconds": 10.011,
            "solver_configuration": {"ksp_type": "fgmres", "pc_type": "hypre"},
            "smac_configuration_tag": "fde4f0",
            "failure_reason": "replay command timed out after 10.0 seconds",
            "returncode": None,
            "replay_command_text": (
                "ksptune-petsc-replay -snapshot_collection snapshots.csv "
                "-ksp_type fgmres -pc_type hypre"
            ),
        }
    )

    output = capsys.readouterr().out
    assert "  reason: replay command timed out after 10.0 seconds" in output
    assert "reproduce:" not in output


def test_tune_progress_does_not_print_replay_command_for_nonconvergence(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 2,
            "objective_value": cli_module.BAD_COST,
            "subprocess_walltime_seconds": 0.285,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "7bb1e4",
            "failure_reason": "not converged: KSP_DIVERGED_ITS",
            "returncode": 0,
            "replay_command_text": "ksptune-petsc-replay -ksp_type cg -pc_type jacobi",
        }
    )

    output = capsys.readouterr().out
    assert "  reason: not converged: KSP_DIVERGED_ITS" in output
    assert "reproduce:" not in output


def test_tune_progress_colorizes_completed_finish_status(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_finished",
            "status": "completed",
            "summary": {},
            "tuning_summary_path": "run/tuning_summary.yaml",
            "trial_csv_path": "run/solver_configuration_trials.csv",
            "ranking_csv_path": "run/solver_configuration_rankings.csv",
            "best_petsc_options_path": "run/best_petsc_options.txt",
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    assert "\033[32mcompleted\033[0m" in output
    assert strip_ansi(output).startswith("\nfinished: completed")


def test_tune_progress_colorizes_best_configuration_on_finish(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_finished",
            "status": "stopped",
            "summary": {
                "best_trial": {
                    "trial_number": 7,
                    "smac_configuration_tag": "e29b19",
                    "objective_value": 2.5,
                    "petsc_options_text": "-ksp_type gmres -pc_type asm",
                }
            },
            "tuning_summary_path": "run/tuning_summary.yaml",
            "trial_csv_path": "run/solver_configuration_trials.csv",
            "ranking_csv_path": "run/solver_configuration_rankings.csv",
            "best_petsc_options_path": "run/best_petsc_options.txt",
        },
        color_enabled=True,
    )

    output = capsys.readouterr().out
    plain_output = strip_ansi(output)
    assert "\033[33mstopped\033[0m" in output
    assert "\033[36m[e29b19]\033[0m" in output
    assert "\033[1m\033[32m2.5\033[0m" in output
    assert "\033[36mrun/tuning_summary.yaml\033[0m" in output
    assert "best: trial 7 [e29b19]  objective=2.5" in plain_output


def test_tune_progress_prints_best_configuration_on_finish(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_finished",
            "status": "stopped",
            "summary": {
                "best_trial": {
                    "trial_number": 7,
                    "smac_configuration_tag": "e29b19",
                    "objective_value": 2.5,
                    "petsc_options_text": "-ksp_type gmres -pc_type asm",
                }
            },
            "tuning_summary_path": "run/tuning_summary.yaml",
            "trial_csv_path": "run/solver_configuration_trials.csv",
            "ranking_csv_path": "run/solver_configuration_rankings.csv",
            "best_petsc_options_path": "run/best_petsc_options.txt",
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\nfinished: stopped")
    assert "best: trial 7 [e29b19]  objective=2.5" in output
    assert "PETSc options:" in output
    assert "-ksp_type gmres -pc_type asm" in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_wraps_long_petsc_options(capsys, monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "60")
    print_tune_progress(
        {
            "event": "run_finished",
            "status": "completed",
            "summary": {
                "objective_name": "objective_sec",
                "best_trial": {
                    "trial_number": 3,
                    "smac_configuration_tag": "abc123",
                    "objective_value": 1.25,
                    "solver_configuration": {"ksp_type": "gmres", "pc_type": "asm"},
                    "petsc_options_text": (
                        "-ksp_pc_side right -ksp_type gmres -pc_type asm "
                        "-ksp_gmres_restart 42 -pc_asm_overlap 3 "
                        "-sub_pc_factor_levels 1 -sub_pc_type ilu"
                    ),
                }
            },
            "tuning_summary_path": "run/tuning_summary.yaml",
            "trial_csv_path": "run/solver_configuration_trials.csv",
            "ranking_csv_path": "run/solver_configuration_rankings.csv",
            "best_petsc_options_path": "run/best_petsc_options.txt",
        }
    )

    output = capsys.readouterr().out
    assert "best: trial 3 [abc123]  objective=1.25s" in output
    assert "    -ksp_pc_side right -ksp_type gmres -pc_type asm" in output
    assert "\n    -ksp_gmres_restart 42" in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")
