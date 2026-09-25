from __future__ import annotations

from dataclasses import asdict
from ksptune.trial_records import BAD_COST

import re

import pytest

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
            "petsc.boomeramg_basic",
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
    assert args.deduplicate_matrices is True
    assert args.reuse_ksp_setup is True
    assert args.use_initial_guess is True
    assert args.mpiexec_args == []
    assert args.replay_cache_memory_mb is None
    assert args.replay_startup_timeout_sec is None
    assert args.soft_timeout_sec is None
    assert args.hard_timeout_sec is None
    assert args.hypre_hierarchy_diagnostics == "auto"
    assert args.max_true_relative_residual == 1.0e-4
    assert args.max_true_residual_norm == 1.0e-4
    assert args.fast_fail is False


def test_tune_parser_accepts_fast_fail_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--fast-fail",
        ]
    )

    assert args.fast_fail is True


def test_tune_parser_accepts_launcher_args_and_replay_cache_budget() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--mpiexec",
            "srun",
            "--mpiexec-arg=--exclusive",
            "--mpiexec-arg=--cpu-bind=cores",
            "--replay-cache-memory-mb",
            "4096",
        ]
    )

    assert args.mpiexec == "srun"
    assert args.mpiexec_args == ["--exclusive", "--cpu-bind=cores"]
    assert args.replay_cache_memory_mb == 4096


def test_tune_parser_accepts_replay_startup_timeout() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--hard-timeout-sec",
            "240",
            "--replay-startup-timeout-sec",
            "3600",
        ]
    )

    assert args.hard_timeout_sec == 240
    assert args.replay_startup_timeout_sec == 3600


def test_tune_parser_accepts_soft_and_hard_timeouts() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--soft-timeout-sec",
            "100",
            "--hard-timeout-sec",
            "150",
        ]
    )

    assert args.soft_timeout_sec == 100
    assert args.hard_timeout_sec == 150


def test_tune_parser_accepts_true_relative_residual_gate() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--max-true-relative-residual",
            "1e-3",
        ]
    )

    assert args.max_true_relative_residual == 1.0e-3


def test_tune_parser_accepts_true_residual_norm_gate() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--max-true-residual-norm",
            "1e-3",
        ]
    )

    assert args.max_true_residual_norm == 1.0e-3


def test_tune_parser_accepts_matrix_deduplication_opt_out() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--no-deduplicate-matrices",
        ]
    )

    assert args.deduplicate_matrices is False


def test_tune_parser_accepts_ksp_setup_reuse_opt_out() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--no-reuse-ksp-setup",
        ]
    )

    assert args.reuse_ksp_setup is False


def test_tune_parser_accepts_initial_guess_opt_out() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--no-initial-guess",
        ]
    )

    assert args.use_initial_guess is False


def test_tune_parser_accepts_run_until_stopped_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--replay-binary",
            "ksptune-petsc-replay",
            "--run-until-stopped",
        ]
    )

    assert args.run_until_stopped is True


def test_tune_parser_accepts_force_restart() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--force-restart",
        ]
    )

    assert args.force_restart is True


def test_resume_parser_accepts_runtime_overrides() -> None:
    args = build_parser().parse_args(
        [
            "resume",
            "run",
            "--trials",
            "200",
            "--workers",
            "4",
            "--soft-timeout-sec",
            "120",
            "--hard-timeout-sec",
            "240",
            "--replay-startup-timeout-sec",
            "1200",
            "--quiet",
        ]
    )

    assert args.tuning_run == "run"
    assert args.trials == 200
    assert args.workers == 4
    assert args.soft_timeout_sec == 120
    assert args.hard_timeout_sec == 240
    assert args.replay_startup_timeout_sec == 1200
    assert args.run_until_stopped is None
    assert args.quiet is True


def test_tune_parser_accepts_color_and_worker_mode() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
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
            "petsc.boomeramg_basic",
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


def test_petsc_options_parser_accepts_petsc_style_options() -> None:
    args = build_parser().parse_args(
        [
            "petsc-options",
            "-pc_type",
            "hypre",
            "-pc_hypre_type",
            "boomeramg",
            "--",
            "-mat_type",
            "aij",
        ]
    )

    assert args.pc_type == "hypre"
    assert args.pc_hypre_type == "boomeramg"
    assert args.petsc_options == ["--", "-mat_type", "aij"]


def test_petsc_options_help_shows_hypre_boomeramg_example(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["petsc-options", "--help"])

    output = capsys.readouterr().out
    assert "ksptune petsc-options -pc_type hypre -pc_hypre_type boomeramg" in output


def test_petsc_options_json_returns_replay_returncode(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "resolve_replay_binary",
        lambda replay_binary, dry_run: ("ksptune-petsc-replay", True),
    )
    monkeypatch.setattr(
        cli_module,
        "query_petsc_options_help",
        lambda **kwargs: {
            "command": ["ksptune-petsc-replay", "-replay_options_help"],
            "returncode": 7,
            "petsc_options": [],
            "filters": [],
            "option_lines": [],
            "filtered_lines": [],
            "raw_output": "failed",
            "timed_out": False,
        },
    )
    args = build_parser().parse_args(["petsc-options", "--json"])

    assert cli_module.cmd_petsc_options(args) == 7
    assert '"returncode": 7' in capsys.readouterr().out


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
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--dry-run",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "dry-run"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["trials"] is None
    assert captured["run_until_stopped"] is True
    assert captured["deduplicate_matrices"] is True
    assert captured["reuse_ksp_setup"] is True
    assert captured["use_initial_guess"] is True
    assert captured["mpiexec_args"] == []
    assert captured["replay_cache_memory_mb"] is None
    assert captured["replay_startup_timeout_sec"] is None
    assert captured["soft_timeout_sec"] is None
    assert captured["hard_timeout_sec"] is None
    assert captured["hypre_hierarchy_diagnostics"] == "auto"
    assert captured["max_true_relative_residual"] == 1.0e-4
    assert captured["max_true_residual_norm"] == 1.0e-4
    assert captured["fast_fail"] is False
    assert captured["force_restart"] is False


def test_tune_command_uses_fixed_trials_when_requested(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--trials",
            "5",
            "--workers",
            "4",
            "--nullspace",
            "field:1,block_size=2",
            "--max-true-relative-residual",
            "1e-3",
            "--max-true-residual-norm",
            "2e-3",
            "--soft-timeout-sec",
            "100",
            "--hard-timeout-sec",
            "150",
            "--replay-startup-timeout-sec",
            "900",
            "--fast-fail",
            "--hypre-hierarchy-diagnostics",
            "off",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["trials"] == 5
    assert captured["workers"] == 4
    assert captured["nullspace"] == "field:1,block_size=2"
    assert captured["run_until_stopped"] is False
    assert captured["deduplicate_matrices"] is True
    assert captured["reuse_ksp_setup"] is True
    assert captured["max_true_relative_residual"] == 1.0e-3
    assert captured["max_true_residual_norm"] == 2.0e-3
    assert captured["soft_timeout_sec"] == 100
    assert captured["hard_timeout_sec"] == 150
    assert captured["replay_startup_timeout_sec"] == 900
    assert captured["hypre_hierarchy_diagnostics"] == "off"
    assert captured["fast_fail"] is True
    assert captured["force_restart"] is False


def test_tune_command_forwards_matrix_deduplication_opt_out(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--no-deduplicate-matrices",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["deduplicate_matrices"] is False


def test_tune_command_forwards_ksp_setup_reuse_opt_out(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--no-reuse-ksp-setup",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["reuse_ksp_setup"] is False


def test_tune_command_forwards_initial_guess_opt_out(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--no-initial-guess",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["use_initial_guess"] is False


def test_tune_command_forwards_force_restart(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--force-restart",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["force_restart"] is True


def test_tune_command_forwards_launcher_args_and_cache_budget(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--mpiexec",
            "srun",
            "--mpiexec-arg=--exclusive",
            "--replay-cache-memory-mb",
            "1024",
            "--quiet",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "run_tuning", fake_run_tuning)

    assert cli_module.cmd_tune(args) == 0
    assert captured["mpiexec"] == "srun"
    assert captured["mpiexec_args"] == ["--exclusive"]
    assert captured["replay_cache_memory_mb"] == 1024


def test_resume_command_forwards_runtime_overrides(monkeypatch) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "resume",
            "run",
            "--trials",
            "200",
            "--run-until-stopped",
            "--workers",
            "3",
            "--soft-timeout-sec",
            "90",
            "--hard-timeout-sec",
            "180",
            "--replay-startup-timeout-sec",
            "600",
            "--max-true-relative-residual",
            "1e-3",
            "--max-true-residual-norm",
            "2e-3",
            "--fast-fail",
            "--hypre-hierarchy-diagnostics",
            "on",
            "--quiet",
        ]
    )

    def fake_resume_tuning(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs.pop("overrides"))
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(cli_module, "resume_tuning", fake_resume_tuning)

    assert cli_module.cmd_resume(args) == 0
    assert captured["args"] == ("run",)
    assert captured["trials"] == 200
    assert captured["run_until_stopped"] is True
    assert captured["workers"] == 3
    assert captured["soft_timeout_sec"] == 90
    assert captured["hard_timeout_sec"] == 180
    assert captured["replay_startup_timeout_sec"] == 600
    assert captured["hypre_hierarchy_diagnostics"] == "on"
    assert captured["max_true_relative_residual"] == 1.0e-3
    assert captured["max_true_residual_norm"] == 2.0e-3
    assert captured["fast_fail"] is True
    assert captured["progress_callback"] is None


def test_tune_command_quiet_mode_does_not_emit_color(monkeypatch, capsys) -> None:
    captured = {}
    args = build_parser().parse_args(
        [
            "tune",
            "--snapshot-directory",
            "snapshots.csv",
            "--parameter-search-space",
            "petsc.boomeramg_basic",
            "--output-directory",
            "run",
            "--dry-run",
            "--quiet",
            "--color",
            "always",
        ]
    )

    monkeypatch.setattr(cli_module, "load_parameter_search_space", lambda name, seed: "space")

    def fake_run_tuning(settings, **kwargs):
        captured.update(asdict(settings))
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
            "snapshot_directory": "snapshots.csv",
            "parameter_search_space_name": "petsc.boomeramg_basic",
            "output_directory": "run",
            "trials": 20,
            "repeat": 1,
            "warmup": 0,
            "run_until_stopped": True,
            "replay_binary": "ksptune-petsc-replay",
            "replay_binary_auto_detected": True,
            "default_solver_configuration": {
                "ksp_type": "gmres",
                "ksp_pc_side": "left",
                "ksp_norm_type": "preconditioned",
                "ksp_rtol": 1.0e-8,
            },
            "max_true_residual_norm": 1.0e-4,
            "max_true_relative_residual": 1.0e-4,
            "fast_fail": True,
            "fast_fail_ksp_max_it": 1,
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
    assert (
        "numerics: ksp=gmres side=left norm=preconditioned rtol=1e-8 "
        "atol=PETSc-default max_it=PETSc-default true_abs_gate=1e-4 "
        "true_rel_gate=1e-4"
    ) in output
    assert "diagnostic: fast-fail  ksp_max_it=1" in output
    assert "replay: ksptune-petsc-replay (auto)" in output
    assert "nullspace: field:1 (field=1, block_size=2)" in output
    assert "Ctrl+C" in output
    assert "SMAC config IDs match tags in trial headings" in output
    assert not output.startswith("\n")
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_prints_timeout_header(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_started",
            "snapshot_directory": "snapshots.csv",
            "parameter_search_space_name": "petsc.boomeramg_basic",
            "output_directory": "run",
            "trials": 20,
            "repeat": 1,
            "warmup": 0,
            "run_until_stopped": False,
            "workers": 1,
            "hard_timeout_sec": 240.0,
            "replay_startup_timeout_sec": 3600.0,
            "sobol_initial_design_configurations": 6,
            "additionalinitial_solver_configuration_count": 8,
            "startup_initial_solver_configuration_count": 1,
            "use_default_solver_configuration": True,
            "tunable_parameter_count": 15,
            "replay_binary": "ksptune-petsc-replay",
            "dry_run": False,
        }
    )

    output = capsys.readouterr().out
    assert (
        "initial design: total=15 | startup=1 seeded=7 default=1 sobol=6 "
        "(in order) | tunable_parameters=15"
    ) in output
    assert (
        "timeouts: solve_soft=none request_wall_hard=240s startup_wall_hard=3600s"
        in output
    )


def test_tune_progress_colorizes_run_header(capsys) -> None:
    print_tune_progress(
        {
            "event": "run_started",
            "snapshot_directory": "snapshots.csv",
            "parameter_search_space_name": "petsc.boomeramg_basic",
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
            "total_wall_time_sec": 0.15,
            "solver_setup_time_sec": 0.03,
            "solve_time_sec_median": 0.10,
            "iterations_total": 5,
            "iterations_median": 5,
            "solve_count": 1,
            "rss_request_peak_sample_mb_sum": 2048.0,
            "rss_setup_delta_mb_sum": 128.0,
            "rss_solve_delta_mb_sum": 16.0,
            "rss_request_delta_mb_sum": 32.0,
            "rss_peak_sample_mb_max_node": 2048.0,
            "rss_peak_sample_mb_max_rank": 1024.0,
            "initial_true_residual_norm_mean": 2.0,
            "initial_true_relative_residual_mean": 1.0,
            "final_true_residual_norm_mean": 2.0e-8,
            "final_true_relative_residual_mean": 1.0e-9,
            "initial_ksp_residual_norm_mean": 1.5,
            "final_ksp_residual_norm_mean": 2.0e-8,
            "ksp_rtol_reference_norm_mean": 2.0,
            "final_ksp_relative_residual_norm_mean": 1.0e-8,
            "ksp_residual_norm_type": "preconditioned",
            "ksp_rtol": 1.0e-8,
            "ksp_atol": 1.0e-50,
            "ksp_dtol": 1.0e4,
            "ksp_max_it": 10000,
            "reason": "KSP_CONVERGED_RTOL",
            "max_true_residual_norm": 1.0e-4,
            "max_true_relative_residual": 1.0e-4,
            "solver_configuration": {
                "ksp_type": "cg",
                "pc_type": "jacobi",
                "ksp_norm_type": "preconditioned",
                "ksp_rtol": 1.0e-8,
                "ksp_atol": 1.0e-50,
            },
            "smac_configuration_tag": "abc123",
            "best_objective_value_so_far": 0.101,
            "is_new_best_so_far": False,
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\ntrial 003/020 [abc123] ok  objective: 0.1234  (best: 0.101)")
    assert "  solver: ksp=cg pc=jacobi norm=preconditioned rtol=1e-8 atol=1e-50" in output
    assert "  runtime: solve=0.100s  setup=0.030s  wall=0.150s" in output
    assert (
        "  memory: rss_sum sample_peak=2.0GB request_delta=32.0MB "
        "pcsetup_delta=128.0MB solve_delta=16.0MB max_node=2.0GB max_rank=1.0GB"
    ) in output
    assert (
        "  tolerances: rtol=1e-8 atol=1e-50 dtol=10000 max_it=10000 "
        "true_abs_gate<=1e-4 true_rel_gate<=1e-4"
    ) in output
    assert "  diagnostics: iters=5 (samples=1, mean=5) reason=KSP_CONVERGED_RTOL" in output
    assert (
        "  true residual: abs_initial=2 rel_initial=1 "
        "abs_final=2.000e-08 rel_final=1.000e-09"
    ) in output
    assert (
        "  ksp residual: norm=preconditioned rtol_ref=2 "
        "abs_initial=1.5 abs_final=2.000e-08 rel_final=1.000e-08"
    ) in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_prints_replay_timing_breakdown(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 15,
            "objective_value": 18.253,
            "total_wall_time_sec": 204.252,
            "subprocess_wall_time_sec": 407.813,
            "matrix_load_time_sec": 167.512,
            "matrix_prepare_time_sec": 0.212,
            "vector_prepare_time_sec": 167.300,
            "nullspace_time_sec": 0.100,
            "true_residual_time_sec": 1.660,
            "matrix_cache_hits": 0,
            "matrix_cache_misses": 1,
            "effective_hard_timeout_sec": 3600.0,
            "used_startup_timeout": True,
            "solver_setup_time_sec": 16.727,
            "solver_setup_time_sec_actual": 16.727,
            "solve_time_sec_median": 18.253,
            "solve_time_sec_total": 18.253,
            "iterations_total": 22,
            "iterations_median": 22,
            "solve_count": 1,
            "solver_configuration": {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_pc_side": "left",
            },
            "smac_configuration_tag": "764275",
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert "  runtime: solve=18.253s  setup=16.727s  wall=204.252s" in output
    assert (
        "  replay: request=407.813s data-load=167.512s matrix=0.212s "
        "vectors=167.300s nullspace=0.100s true-residuals=1.660s "
        "outer-overhead=203.561s startup_wall_hard=3600s matrix-cache=0/1"
    ) in output


def test_tune_progress_prints_gamg_diagnostics(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 3,
            "objective_value": 0.25,
            "total_wall_time_sec": 0.4,
            "solver_setup_time_sec": 0.2,
            "solve_time_sec_median": 0.1,
            "iterations_total": 5,
            "iterations_median": 5,
            "solve_count": 1,
            "pc_diagnostics": [
                {
                    "pc_type": "gamg",
                    "setup_index": 1,
                    "setup_key": "gamg-key",
                    "metrics": {
                        "levels": 4,
                        "grid_complexity": 1.25,
                        "operator_complexity": 1.75,
                        "finest_nonzeros": 7000.0,
                        "total_nonzeros": 12250.0,
                        "coarsest_rows": 25.0,
                        "interpolation_nonzeros": 4000.0,
                        "max_operator_nonzeros_per_row": 12.25,
                        "max_interpolation_nonzeros_per_row": 4.0,
                        "min_active_ranks": 2.0,
                    },
                    "string_metrics": {},
                    "levels": [],
                }
            ],
            "solver_configuration": {"ksp_type": "gmres", "pc_type": "gamg"},
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert (
        "  pc: gamg setup=1 levels=4 grid=1.25 op=1.75 max_op_nnz/row=12.25 "
        "max_interp_nnz/row=4 min_active_ranks=2 fine_nnz=7.000e+03 "
        "total_nnz=1.225e+04 coarse_rows=25 interp_nnz=4.000e+03"
    ) in output


def test_tune_progress_prints_hypre_diagnostics(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 3,
            "objective_value": 0.25,
            "total_wall_time_sec": 0.4,
            "solver_setup_time_sec": 0.2,
            "solve_time_sec_median": 0.1,
            "iterations_total": 5,
            "iterations_median": 5,
            "solve_count": 1,
            "pc_diagnostics": [
                {
                    "pc_type": "hypre",
                    "setup_index": 1,
                    "setup_key": "hypre-key",
                    "metrics": {
                        "levels": 5,
                        "grid_complexity": 1.1,
                        "operator_complexity": 1.4,
                        "interpolation_complexity": 0.3,
                        "finest_nonzeros": 7000.0,
                        "total_nonzeros": 9800.0,
                        "coarsest_rows": 10.0,
                        "interpolation_nonzeros": 2100.0,
                        "max_row_nonzeros": 42.0,
                        "max_operator_nonzeros_per_row": 9.8,
                        "max_interpolation_nonzeros_per_row": 3.0,
                    },
                    "string_metrics": {},
                    "levels": [],
                }
            ],
            "solver_configuration": {"ksp_type": "gmres", "pc_type": "hypre"},
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert (
        "  pc: hypre setup=1 levels=5 grid=1.1 op=1.4 interp=0.3 "
        "max_op_nnz/row=9.8 max_interp_nnz/row=3 fine_nnz=7.000e+03 "
        "total_nnz=9.800e+03 coarse_rows=10 interp_nnz=2.100e+03 max_row_nnz=42"
    ) in output


def test_tune_progress_prints_non_amg_pc_diagnostics(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 3,
            "objective_value": 0.25,
            "total_wall_time_sec": 0.4,
            "solver_setup_time_sec": 0.2,
            "solve_time_sec_median": 0.1,
            "iterations_total": 5,
            "iterations_median": 5,
            "solve_count": 1,
            "pc_diagnostics": [
                {
                    "pc_type": "bjacobi",
                    "setup_index": 1,
                    "setup_key": "bjacobi-key",
                    "metrics": {
                        "global_blocks": 10,
                        "local_blocks": 1,
                    },
                    "string_metrics": {
                        "sub_ksp_type": "preonly",
                        "sub_pc_type": "ilu",
                    },
                    "levels": [],
                }
            ],
            "solver_configuration": {"ksp_type": "gmres", "pc_type": "bjacobi"},
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert (
        "  pc: bjacobi setup=1 global_blocks=10 local_blocks=1 "
        "sub_ksp=preonly sub_pc=ilu"
    ) in output


def test_tune_progress_prints_queued_trial(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "current_progress_timestamp", lambda: "12:34:56")
    print_tune_progress(
        {
            "event": "trial_queued",
            "trial_number": 3,
            "trial_count": 15,
            "smac_configuration_tag": "c4eb44",
            "solver_configuration": {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_pc_side": "left",
                "ksp_norm_type": "preconditioned",
                "ksp_rtol": 1.0e-8,
                "ksp_atol": 1.0e-8,
            },
            "hard_timeout_sec": 240.0,
            "replay_startup_timeout_sec": 3600.0,
        }
    )

    output = capsys.readouterr().out
    assert output == (
        "\n[12:34:56] trial 003/015 [c4eb44] queued "
        "(timeouts: solve_soft=none request_wall_hard=240s startup_wall_hard=3600s)\n"
    )


def test_tune_progress_prints_total_and_average_solve_time(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 4.12,
            "total_wall_time_sec": 250.0,
            "solver_setup_time_sec": 0.03,
            "solve_time_sec_total": 232.1,
            "solve_time_sec_mean": 4.12,
            "solve_time_sec_median": 3.9,
            "solve_time_sec_min": 3.25,
            "solve_time_sec_max": 5.5,
            "solve_time_sec_stddev": 0.5,
            "solve_time_sec_sem": 0.067,
            "solve_time_sec_relative_sem": 0.0163,
            "solve_time_sec_range": 2.25,
            "solve_count": 56,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "abc123",
            "best_objective_value_so_far": 4.0,
            "is_new_best_so_far": False,
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert (
        "  runtime: solve=232.100s (samples=56, avg=4.120s, "
        "sem=0.067s rel_sem=1.6%, min-max=3.250s-5.500s)  "
        "setup=0.030s  wall=250.000s"
    ) in output


def test_tune_progress_prints_ksp_setup_cache_summary(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 4.12,
            "total_wall_time_sec": 250.0,
            "solver_setup_time_sec": 0.10,
            "solver_setup_time_sec_actual": 0.02,
            "ksp_setup_cache_hits": 4,
            "ksp_setup_cache_misses": 1,
            "solve_time_sec_total": 232.1,
            "solve_time_sec_mean": 4.12,
            "solve_count": 56,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "abc123",
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert "setup=0.020s (ksp-cache=4/1)" in output


def test_tune_progress_prints_solve_mpi_summary(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 4.12,
            "total_wall_time_sec": 250.0,
            "solver_setup_time_sec": 0.03,
            "solve_time_sec_total": 232.1,
            "solve_time_sec_mean": 4.12,
            "solve_count": 56,
            "solve_mpi_message_count": 824.0,
            "solve_mpi_message_bytes": 33.6 * 1024 * 1024,
            "solve_mpi_message_bytes_mean": 42_767.28,
            "solve_mpi_reduction_count": 216.0,
            "final_true_relative_residual_mean": 1.0e-9,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
            "smac_configuration_tag": "abc123",
            "failure_reason": None,
        }
    )

    output = capsys.readouterr().out
    assert "  mpi: solve msgs=824 bytes=33.6MB reductions=216 avg=41.8KB/msg" in output


def test_tune_progress_colorizes_successful_trial_line(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 3,
            "trial_count": 20,
            "objective_value": 0.1234,
            "total_wall_time_sec": 0.15,
            "solver_setup_time_sec": 0.03,
            "solve_time_sec_median": 0.10,
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
            "objective_name": "objective_time_sec_median",
            "total_wall_time_sec": 0.15,
            "solver_setup_time_sec": 0.03,
            "solve_time_sec_median": 0.10,
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
            "total_wall_time_sec": 0.15,
            "solver_setup_time_sec": 0.03,
            "solve_time_sec_median": 0.10,
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
            "objective_value": BAD_COST,
            "total_wall_time_sec": 0.2,
            "rss_request_peak_sample_mb_sum": 768.0,
            "rss_setup_delta_mb_sum": 64.0,
            "rss_solve_delta_mb_sum": 0.0,
            "rss_request_delta_mb_sum": 128.0,
            "rss_peak_sample_mb_max_node": 768.0,
            "rss_peak_sample_mb_max_rank": 256.0,
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
            "objective_value": BAD_COST,
            "total_wall_time_sec": 0.2,
            "rss_request_peak_sample_mb_sum": 768.0,
            "rss_setup_delta_mb_sum": 64.0,
            "rss_solve_delta_mb_sum": 0.0,
            "rss_request_delta_mb_sum": 128.0,
            "rss_peak_sample_mb_max_node": 768.0,
            "rss_peak_sample_mb_max_rank": 256.0,
            "final_true_relative_residual_mean": 0.4,
            "solver_configuration": {"ksp_type": "cg", "pc_type": "none"},
            "smac_configuration_tag": "7bb1e4",
            "failure_reason": "not converged: KSP_DIVERGED_ITS",
            "stdout_path": "/work/run/replay_logs/replay_server.stdout.log",
            "stderr_path": "/work/run/replay_logs/replay_server.stderr.log",
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\ntrial 001/002 [7bb1e4] failed  objective: bad-cost  (best: n/a)")
    assert "  runtime: subprocess=n/a" in output
    assert "  reason: not converged: KSP_DIVERGED_ITS" in output
    assert "  logs: stdout=/work/run/replay_logs/replay_server.stdout.log" in output
    assert "stderr=/work/run/replay_logs/replay_server.stderr.log" in output
    assert (
        "  memory: rss_sum sample_peak=768.0MB request_delta=128.0MB "
        "pcsetup_delta=64.0MB solve_delta=0B max_node=768.0MB max_rank=256.0MB"
    ) in output
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_tune_progress_prints_true_relative_residual_failure_reason(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "trial_number": 1,
            "trial_count": 2,
            "objective_value": BAD_COST,
            "initial_true_residual_norm_mean": 2.0,
            "initial_true_relative_residual_mean": 1.0,
            "final_true_residual_norm_mean": 1.18648,
            "final_true_relative_residual_mean": 2.235,
            "initial_ksp_residual_norm_mean": 0.9,
            "final_ksp_residual_norm_mean": 1.0e-8,
            "ksp_rtol_reference_norm_mean": 1.0,
            "final_ksp_relative_residual_norm_mean": 1.0e-8,
            "ksp_residual_norm_type": "preconditioned",
            "ksp_rtol": 1.0e-8,
            "ksp_atol": 1.0e-50,
            "ksp_dtol": 1.0e4,
            "ksp_max_it": 10000,
            "reason": "KSP_CONVERGED_RTOL",
            "max_true_residual_norm": 1.0e-4,
            "max_true_relative_residual": 1.0e-4,
            "solver_configuration": {"ksp_type": "gmres", "pc_type": "hypre"},
            "smac_configuration_tag": "bad123",
            "failure_reason": "true relative residual too large: 2.235 > 0.0001",
        }
    )

    output = capsys.readouterr().out
    assert output.startswith("\ntrial 001/002 [bad123] failed  objective: bad-cost")
    assert "  reason: true relative residual too large: 2.235 > 0.0001" in output
    assert (
        "  tolerances: rtol=1e-8 atol=1e-50 dtol=10000 max_it=10000 "
        "true_abs_gate<=1e-4 true_rel_gate<=1e-4"
    ) in output
    assert "  diagnostics: iters=n/a reason=KSP_CONVERGED_RTOL" in output
    assert (
        "  true residual: abs_initial=2 rel_initial=1 "
        "abs_final=1.18648 rel_final=2.235"
    ) in output
    assert (
        "  ksp residual: norm=preconditioned rtol_ref=1 "
        "abs_initial=0.9 abs_final=1.000e-08 rel_final=1.000e-08"
    ) in output


def test_tune_progress_prints_replay_command_for_subprocess_failure(capsys) -> None:
    print_tune_progress(
        {
            "event": "trial_finished",
            "status": "process_error",
            "trial_number": 56,
            "trial_count": None,
            "objective_value": BAD_COST,
            "subprocess_wall_time_sec": 0.285,
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
            "status": "hard_timeout",
            "trial_number": 45,
            "trial_count": None,
            "objective_value": BAD_COST,
            "subprocess_wall_time_sec": 10.011,
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
            "status": "nonconverged",
            "trial_number": 1,
            "trial_count": 2,
            "objective_value": BAD_COST,
            "subprocess_wall_time_sec": 0.285,
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
                "objective_name": "objective_time_sec_median",
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
