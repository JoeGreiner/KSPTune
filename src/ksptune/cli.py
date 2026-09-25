from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import yaml

from .parameter_importance import analyze_parameter_importance
from .parameter_search_spaces import (
    build_configspace_from_parameter_search_space,
    configspace_to_pretty_json,
    list_builtin_parameter_search_spaces,
    load_parameter_search_space,
    parameter_search_space_to_yaml,
)
from .petsc_options import petsc_options_to_text, render_petsc_options_from_solver_configuration
from .petsc_help import (
    normalize_petsc_help_options,
    petsc_options_help_to_json,
    query_petsc_options_help,
)
from .snapshot_analysis import analyze_snapshots
from .tuning_runs import (
    TuningSettings,
    RESUME_OVERRIDE_FIELDS,
    resolve_replay_binary,
    resume_tuning,
    run_tuning,
)

from .progress_output import (
    color_enabled,
    print_tune_progress as render_tune_progress,
)
from .trial_records import add_trial_metrics


def print_yaml(data: Any) -> None:
    print(yaml.safe_dump(data, sort_keys=False).rstrip())


def current_progress_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def print_tune_progress(event: dict[str, Any], *, color_enabled: bool = False) -> None:
    render_tune_progress(
        add_trial_metrics(event),
        color_enabled=color_enabled,
        timestamp=current_progress_timestamp(),
    )


def cmd_parameter_search_spaces_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    for name in list_builtin_parameter_search_spaces():
        print(name)
    return 0


def cmd_parameter_search_spaces_show(args: argparse.Namespace) -> int:
    parameter_search_space = load_parameter_search_space(args.name_or_path)
    if args.configspace_json:
        configuration_space = build_configspace_from_parameter_search_space(parameter_search_space)
        print(configspace_to_pretty_json(configuration_space))
    else:
        print(parameter_search_space_to_yaml(parameter_search_space).rstrip())
    return 0


def cmd_parameter_search_spaces_validate(args: argparse.Namespace) -> int:
    parameter_search_space = load_parameter_search_space(args.name_or_path)
    build_configspace_from_parameter_search_space(parameter_search_space)
    print("ok")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    parameter_search_space = load_parameter_search_space(
        args.parameter_search_space, seed=args.seed
    )
    use_color = False if args.quiet else color_enabled(args.color)
    progress_callback = (
        None if args.quiet else lambda event: print_tune_progress(event, color_enabled=use_color)
    )
    result = run_tuning(
        TuningSettings(**{setting.name: getattr(args, setting.name) for setting in fields(TuningSettings)}),
        parameter_search_space=parameter_search_space,
        progress_callback=progress_callback,
    )
    if args.quiet:
        print_yaml(result)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    use_color = False if args.quiet else color_enabled(args.color)
    progress_callback = (
        None if args.quiet else lambda event: print_tune_progress(event, color_enabled=use_color)
    )
    result = resume_tuning(
        args.tuning_run,
        overrides={name: getattr(args, name) for name in RESUME_OVERRIDE_FIELDS},
        progress_callback=progress_callback,
    )
    if args.quiet:
        print_yaml(result)
    return 0


def cmd_analyze_snapshots(args: argparse.Namespace) -> int:
    snapshot_directory = args.snapshot_directory or args.snapshot_directory_argument
    if snapshot_directory is None:
        raise ValueError("snapshot directory is required")
    if (
        args.snapshot_directory is not None
        and args.snapshot_directory_argument is not None
        and Path(args.snapshot_directory) != Path(args.snapshot_directory_argument)
    ):
        raise ValueError(
            "pass the snapshot directory either positionally or with --snapshot-directory"
        )

    output_path = (
        args.output
        or Path(snapshot_directory) / "ksptune_snapshot_analysis" / "snapshot_analysis_summary.yaml"
    )
    result = analyze_snapshots(
        snapshot_directory,
        output_path,
        petsc_snapshot_analysis_binary=args.petsc_analysis_binary,
        mpiexec=args.mpiexec,
        mpi_processes=args.np,
        threads_per_rank=args.threads_per_rank,
        nullspace=args.nullspace,
        solve_index=args.solve_index,
        snapshot_id=args.snapshot_id,
        rhs_compatibility_tolerance=args.rhs_compatibility_tolerance,
        nullspace_residual_tolerance=args.nullspace_residual_tolerance,
        metadata_only=args.metadata_only,
    )
    print_yaml(result)
    return 0


def cmd_analyze_parameter_importance(args: argparse.Namespace) -> int:
    output_path = args.output or Path(args.tuning_run) / "parameter_importance_summary.yaml"
    result = analyze_parameter_importance(args.tuning_run, output_path)
    print_yaml(result)
    return 0


def cmd_export_petsc_options(args: argparse.Namespace) -> int:
    tuning_run_directory = Path(args.tuning_run)
    configspace_path = tuning_run_directory / "configspace.json"
    parameter_search_space = load_parameter_search_space(configspace_path)
    solver_configuration = yaml.safe_load(
        (tuning_run_directory / "best_solver_configuration.yaml").read_text(encoding="utf-8")
    )
    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    print(petsc_options_to_text(options))
    return 0


def cmd_petsc_options(args: argparse.Namespace) -> int:
    replay_binary, replay_binary_auto_detected = resolve_replay_binary(
        args.replay_binary,
        dry_run=False,
    )
    extra_petsc_options = list(args.petsc_options or [])
    if extra_petsc_options and extra_petsc_options[0] == "--":
        extra_petsc_options = extra_petsc_options[1:]
    petsc_options, notes = normalize_petsc_help_options(
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        pc_hypre_type=args.pc_hypre_type,
        extra_petsc_options=extra_petsc_options,
    )
    result = query_petsc_options_help(
        replay_binary=replay_binary,
        petsc_options=petsc_options,
        mpiexec=args.mpiexec,
        mpiexec_args=args.mpiexec_arg,
        mpi_processes=args.np,
        filters=args.filter,
        raw=args.raw,
        timeout_sec=args.timeout_sec,
    )
    if args.json:
        print(petsc_options_help_to_json(result))
        return int(result["returncode"])

    print("PETSc options help")
    replay_label = str(replay_binary)
    if replay_binary_auto_detected:
        replay_label += " (auto)"
    print(f"  replay: {replay_label}")
    print(f"  command: {shlex.join(str(part) for part in result['command'])}")
    for note in notes:
        print(f"  note: {note}")
    if not args.raw and result["filters"]:
        print(f"  filters: {' '.join(result['filters'])}")
    print()

    lines = result["raw_output"].splitlines() if args.raw else result["filtered_lines"]
    for line in lines:
        print(line)
    if not lines:
        print("No matching PETSc option lines found. Use --raw to inspect full PETSc -help output.")
    if result["returncode"] != 0:
        if not args.raw and result["raw_output"].strip():
            print()
            print("Command failed; raw output:")
            print(result["raw_output"].rstrip())
        return int(result["returncode"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksptune")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parameter_search_spaces = subparsers.add_parser("parameter-search-spaces")
    pss_subparsers = parameter_search_spaces.add_subparsers(
        dest="parameter_search_space_command",
        required=True,
    )

    pss_list = pss_subparsers.add_parser("list")
    pss_list.set_defaults(func=cmd_parameter_search_spaces_list)

    pss_show = pss_subparsers.add_parser("show")
    pss_show.add_argument("name_or_path")
    pss_show.add_argument("--configspace-json", action="store_true")
    pss_show.set_defaults(func=cmd_parameter_search_spaces_show)

    pss_validate = pss_subparsers.add_parser("validate")
    pss_validate.add_argument("name_or_path")
    pss_validate.set_defaults(func=cmd_parameter_search_spaces_validate)

    tune = subparsers.add_parser("tune")
    tune.add_argument("--snapshot-directory", required=True)
    tune.add_argument("--parameter-search-space", required=True)
    tune.add_argument("--output-directory", required=True)
    tune.add_argument("--replay-binary")
    tune.add_argument("--mpiexec")
    tune.add_argument(
        "--mpiexec-arg", dest="mpiexec_args",
        action="append",
        help=(
            "Additional argument passed to the MPI launcher. Repeat for multiple arguments. "
            "Use --mpiexec-arg=--flag when the launcher argument starts with '-'."
        ),
    )
    tune.add_argument("--np", dest="mpi_processes", type=int)
    tune.add_argument("--threads-per-rank", type=int)
    tune.add_argument(
        "--workers",
        type=int,
        help="Number of parallel SMAC workers. Each worker may launch --np MPI ranks.",
    )
    tune.add_argument(
        "--trials",
        type=int,
        help="Run a fixed number of trials. Omit to run until stopped with Ctrl+C.",
    )
    tune.add_argument("--repeat", type=int)
    tune.add_argument("--warmup", type=int)
    tune.add_argument(
        "--soft-timeout-sec",
        type=float,
        help="Cumulative KSPSolve time budget per trial. A soft timeout keeps the replay server alive.",
    )
    tune.add_argument(
        "--hard-timeout-sec",
        type=float,
        help="Python-side safety timeout. A hard timeout kills and restarts the replay server.",
    )
    tune.add_argument(
        "--replay-startup-timeout-sec",
        type=float,
        help=(
            "Timeout for the first request handled by a newly started replay server. "
            "Use this for large first matrix loads; later trials still use --hard-timeout-sec."
        ),
    )
    tune.add_argument("--objective", dest="objective_name")
    tune.add_argument(
        "--nullspace",
        help=(
            "Replay nullspace: from-metadata, none, constant, or field:<index>[,block_size=<n>]."
        ),
    )
    tune.add_argument(
        "--nullspace-actions",
        help=(
            "Replay nullspace actions: metadata, matrix, transpose, near, remove-rhs, "
            "all, none, or a comma-separated list. Aliases: right=matrix, left=transpose."
        ),
    )
    tune.add_argument(
        "--no-deduplicate-matrices", dest="deduplicate_matrices",
        action="store_false",
        help="Do not hash duplicate matrix candidates when writing the resolved snapshot collection.",
    )
    tune.add_argument(
        "--no-reuse-ksp-setup", dest="reuse_ksp_setup",
        action="store_false",
        help="Do not reuse KSP/PC setup across snapshots with identical matrix cache keys.",
    )
    tune.add_argument(
        "--no-initial-guess", dest="use_initial_guess",
        action="store_false",
        help="Replay solves from the zero vector instead of using the exported x0 vectors.",
    )
    tune.add_argument(
        "--replay-cache-memory-mb",
        type=float,
        help="Approximate total replay-server cache memory budget per worker.",
    )
    tune.add_argument(
        "--hypre-hierarchy-diagnostics",
        choices=["auto", "on", "off"],
        help=(
            "Collect BoomerAMG hierarchy diagnostics when supported by the linked PETSc "
            "build. Default: auto."
        ),
    )
    tune.add_argument(
        "--max-true-relative-residual",
        type=float,
        help="Reject PETSc-converged trials whose measured true relative residual exceeds this value.",
    )
    tune.add_argument(
        "--max-true-residual-norm",
        type=float,
        help="Reject PETSc-converged trials whose measured true residual norm exceeds this value.",
    )
    tune.add_argument(
        "--fast-fail",
        action="store_true",
        help=(
            "Diagnostic mode: run real replay trials but force -ksp_max_it 1 "
            "to expose crashing or very slow solver setups quickly."
        ),
    )
    tune.add_argument(
        "--force-restart",
        action="store_true",
        help="Overwrite an existing tuning run directory instead of refusing to replace it.",
    )
    tune.add_argument("--seed", type=int)
    tune.add_argument("--dry-run", action="store_true")
    tune.add_argument(
        "--run-until-stopped",
        action="store_true",
        help="Run until stopped with Ctrl+C. This is the default when --trials is omitted.",
    )
    tune.add_argument(
        "--color",
        default="auto",
        choices=["auto", "always", "never"],
        help="Colorize tune output. Default: auto.",
    )
    tune.add_argument("--quiet", action="store_true")
    tune.set_defaults(func=cmd_tune, **{
        setting.name: setting.default if setting.default is not MISSING else setting.default_factory()
        for setting in fields(TuningSettings)
        if setting.default is not MISSING or setting.default_factory is not MISSING
    })

    resume = subparsers.add_parser("resume")
    resume.add_argument("tuning_run")
    resume.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Total trial target, including already completed trials.",
    )
    resume.add_argument(
        "--run-until-stopped",
        action="store_true",
        default=None,
        help="Continue until stopped with Ctrl+C.",
    )
    resume.add_argument(
        "--workers",
        type=int,
        help="Override the number of parallel SMAC workers for the resumed run.",
    )
    resume.add_argument(
        "--soft-timeout-sec",
        type=float,
        help="Override the cumulative KSPSolve time budget per trial.",
    )
    resume.add_argument(
        "--hard-timeout-sec",
        type=float,
        help="Override the Python-side safety timeout.",
    )
    resume.add_argument(
        "--replay-startup-timeout-sec",
        type=float,
        help=(
            "Override the timeout for the first request handled by a newly started replay server."
        ),
    )
    resume.add_argument(
        "--hypre-hierarchy-diagnostics",
        choices=["auto", "on", "off"],
        help=(
            "Override BoomerAMG hierarchy diagnostics for the resumed run. "
            "Default: use the saved run setting."
        ),
    )
    resume.add_argument("--max-true-relative-residual", type=float)
    resume.add_argument("--max-true-residual-norm", type=float)
    resume.add_argument(
        "--fast-fail",
        action="store_true",
        default=None,
        help=(
            "Diagnostic mode: run real replay trials but force -ksp_max_it 1 "
            "to expose crashing or very slow solver setups quickly."
        ),
    )
    resume.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize resume output. Default: auto.",
    )
    resume.add_argument("--quiet", action="store_true")
    resume.set_defaults(func=cmd_resume)

    analyze_snapshots = subparsers.add_parser(
        "analyze-snapshots",
        aliases=["analyse-snapshots"],
    )
    analyze_snapshots.add_argument("snapshot_directory_argument", nargs="?")
    analyze_snapshots.add_argument("--snapshot-directory")
    analyze_snapshots.add_argument("--output")
    analyze_snapshots.add_argument("--petsc-analysis-binary")
    analyze_snapshots.add_argument("--mpiexec", default="mpiexec")
    analyze_snapshots.add_argument("--np", type=int, default=1)
    analyze_snapshots.add_argument("--threads-per-rank", type=int, default=1)
    snapshot_selection = analyze_snapshots.add_mutually_exclusive_group()
    snapshot_selection.add_argument("--solve-index", type=int)
    snapshot_selection.add_argument("--snapshot-id")
    analyze_snapshots.add_argument(
        "--nullspace",
        default="none",
        help=("Candidate nullspace to test: none, constant, or field:<index>[,block_size=<n>]."),
    )
    analyze_snapshots.add_argument("--rhs-compatibility-tolerance", type=float, default=1.0e-10)
    analyze_snapshots.add_argument("--nullspace-residual-tolerance", type=float, default=1.0e-10)
    analyze_snapshots.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only summarize snapshot metadata; do not run the PETSc snapshot analysis binary.",
    )
    analyze_snapshots.set_defaults(func=cmd_analyze_snapshots)

    analyze_importance = subparsers.add_parser("analyze-parameter-importance")
    analyze_importance.add_argument("--tuning-run", required=True)
    analyze_importance.add_argument("--output")
    analyze_importance.set_defaults(func=cmd_analyze_parameter_importance)

    export_options = subparsers.add_parser("export-petsc-options")
    export_options.add_argument("--tuning-run", required=True)
    export_options.set_defaults(func=cmd_export_petsc_options)

    petsc_options = subparsers.add_parser(
        "petsc-options",
        aliases=["petsc-parameters"],
        help="Query PETSc options exposed by the linked PETSc version.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("example:\n  ksptune petsc-options -pc_type hypre -pc_hypre_type boomeramg"),
    )
    petsc_options.add_argument("--replay-binary")
    petsc_options.add_argument("--mpiexec", default="mpiexec")
    petsc_options.add_argument(
        "--mpiexec-arg",
        action="append",
        default=[],
        help=(
            "Additional argument passed to the MPI launcher. Repeat for multiple arguments. "
            "Use --mpiexec-arg=--flag when the launcher argument starts with '-'."
        ),
    )
    petsc_options.add_argument("--np", type=int, default=1)
    petsc_options.add_argument("--timeout-sec", type=float, default=30.0)
    petsc_options.add_argument("-ksp_type", "--ksp-type")
    petsc_options.add_argument("-pc_type", "--pc-type")
    petsc_options.add_argument("-pc_hypre_type", "--pc-hypre-type")
    petsc_options.add_argument(
        "--filter",
        action="append",
        help="Only print PETSc option lines starting with this prefix. Repeatable.",
    )
    petsc_options.add_argument(
        "--raw",
        action="store_true",
        help="Print full PETSc -help output instead of inferred option lines.",
    )
    petsc_options.add_argument("--json", action="store_true")
    petsc_options.add_argument(
        "petsc_options",
        nargs=argparse.REMAINDER,
        help="Additional PETSc options after --, for example -- -mat_type aij.",
    )
    petsc_options.set_defaults(func=cmd_petsc_options)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
