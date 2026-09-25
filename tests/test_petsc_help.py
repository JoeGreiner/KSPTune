from __future__ import annotations

import subprocess

from ksptune.petsc_help import (
    build_petsc_options_help_command,
    extract_petsc_option_lines,
    filter_petsc_option_lines,
    infer_option_filters,
    normalize_petsc_help_options,
    query_petsc_options_help,
)


def test_build_petsc_options_help_command_uses_replay_help_mode() -> None:
    command = build_petsc_options_help_command(
        replay_binary="ksptune-petsc-replay",
        petsc_options=["-pc_type", "gamg"],
        mpiexec="srun",
        mpiexec_args=["--exclusive"],
        mpi_processes=4,
    )

    assert command[:4] == ["srun", "--exclusive", "-n", "4"]
    assert command[4:6] == ["ksptune-petsc-replay", "-replay_options_help"]
    assert command[-2:] == ["-pc_type", "gamg"]


def test_build_petsc_options_help_command_uses_explicit_launcher_rank_for_args() -> None:
    command = build_petsc_options_help_command(
        replay_binary="ksptune-petsc-replay",
        petsc_options=[],
        mpiexec="srun",
        mpiexec_args=["--exclusive"],
        mpi_processes=1,
    )

    assert command == [
        "srun",
        "--exclusive",
        "-n",
        "1",
        "ksptune-petsc-replay",
        "-replay_options_help",
    ]


def test_build_petsc_options_help_command_uses_named_launcher_with_one_rank() -> None:
    command = build_petsc_options_help_command(
        replay_binary="ksptune-petsc-replay",
        petsc_options=[],
        mpiexec="srun",
        mpi_processes=1,
    )

    assert command == ["srun", "-n", "1", "ksptune-petsc-replay", "-replay_options_help"]


def test_normalize_petsc_help_options_accepts_boomeramg_shortcut() -> None:
    options, notes = normalize_petsc_help_options(pc_type="boomeramg")

    assert options == ["-pc_type", "hypre", "-pc_hypre_type", "boomeramg"]
    assert notes == [
        "interpreted pc_type=boomeramg as -pc_type hypre -pc_hypre_type boomeramg"
    ]


def test_infer_option_filters_prefers_specific_hypre_subtype() -> None:
    filters = infer_option_filters(
        ["-ksp_type", "gmres", "-pc_type", "hypre", "-pc_hypre_type", "boomeramg"]
    )

    assert filters == [
        "-ksp_type",
        "-ksp_gmres",
        "-pc_type",
        "-pc_hypre",
        "-pc_hypre_boomeramg",
    ]


def test_infer_option_filters_includes_gamg_and_mg_options() -> None:
    filters = infer_option_filters(["-pc_type", "gamg"])

    assert filters == ["-pc_type", "-pc_gamg", "-pc_mg", "-mg_"]


def test_filter_petsc_option_lines_uses_prefixes() -> None:
    lines = [
        "-ksp_type <method>: Krylov method",
        "-pc_type <type>: Preconditioner",
        "-pc_gamg_threshold <threshold>: Drop tolerance",
        "-mat_type <type>: Matrix type",
    ]

    assert filter_petsc_option_lines(lines, filters=["-pc_gamg"]) == [
        "-pc_gamg_threshold <threshold>: Drop tolerance"
    ]


def test_extract_petsc_option_lines_ignores_mpi_separators() -> None:
    output = """
--------------------------------------------------------------------------
-pc_type <type>: Preconditioner
--------------------------------------------------------------------------
"""

    assert extract_petsc_option_lines(output) == ["-pc_type <type>: Preconditioner"]


def test_query_petsc_options_help_reports_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        raise subprocess.TimeoutExpired(
            cmd=["ksptune-petsc-replay"],
            timeout=0.01,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr("ksptune.petsc_help.subprocess.run", fake_run)

    result = query_petsc_options_help(
        replay_binary="ksptune-petsc-replay",
        petsc_options=["-pc_type", "gamg"],
        timeout_sec=0.01,
    )

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "partial stdout" in result["raw_output"]
    assert "timed out" in result["error"]
