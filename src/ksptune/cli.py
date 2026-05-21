from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
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
from .snapshot_analysis import analyze_snapshots
from .tuning_runs import BAD_COST, run_tuning

ANSI_STYLES = {
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


def print_yaml(data: Any) -> None:
    print(yaml.safe_dump(data, sort_keys=False).rstrip())


def style(text: Any, *styles: str, color_enabled: bool = False) -> str:
    plain_text = str(text)
    if not color_enabled or not styles:
        return plain_text
    prefix = "".join(ANSI_STYLES[style] for style in styles)
    return f"{prefix}{plain_text}{ANSI_STYLES['reset']}"


def color_enabled(color: str) -> bool:
    if color == "always":
        return True
    if color == "never":
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def format_sec(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def format_megabytes(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}MB"


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    numeric_value = float(value)
    if abs(numeric_value) >= 1000 or (numeric_value != 0.0 and abs(numeric_value) < 0.001):
        return f"{numeric_value:.3e}"
    return f"{numeric_value:.6g}"


def format_objective_value_text(event: dict[str, Any], value_key: str) -> str:
    value = event.get(value_key)
    if value is None:
        return "n/a"
    numeric_value = float(value)
    if numeric_value >= BAD_COST * 0.1:
        return "bad-cost"
    text = format_float(numeric_value)
    objective_name = str(event.get("objective_name") or "")
    if "_sec" in objective_name:
        text = f"{text}s"
    return text


def format_objective(event: dict[str, Any], *, color_enabled: bool = False) -> str:
    value = event.get("objective_value")
    if value is None:
        return "n/a"
    numeric_value = float(value)
    if numeric_value >= BAD_COST * 0.1:
        return style("bad-cost", "bold", "red", color_enabled=color_enabled)
    text = format_objective_value_text(event, "objective_value")
    if event.get("is_new_best_so_far"):
        return style(text, "bold", "green", color_enabled=color_enabled)
    if event.get("best_objective_value_so_far") is not None:
        return style(text, "bold", "yellow", color_enabled=color_enabled)
    return style(text, "bold", "green", color_enabled=color_enabled)


def format_best_objective_so_far(event: dict[str, Any], *, color_enabled: bool = False) -> str:
    value = event.get("best_objective_value_so_far")
    if value is None:
        return "n/a"
    text = format_objective_value_text(event, "best_objective_value_so_far")
    return style(text, "bold", "cyan", color_enabled=color_enabled)


def format_trial_index(trial_number: Any, trial_count: Any) -> str:
    if trial_number is None:
        return "?/?"
    if trial_count is None:
        return f"{int(trial_number):03d}/..."
    width = max(3, len(str(int(trial_count))))
    return f"{int(trial_number):0{width}d}/{int(trial_count):0{width}d}"


def short_text(value: Any, max_length: int = 120) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def display_path(value: Any) -> str:
    if value is None:
        return "n/a"
    path = Path(str(value))
    text = str(path)
    if not path.is_absolute():
        return text
    try:
        relative_path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return text
    relative_text = str(relative_path)
    return relative_text if len(relative_text) < len(text) else text


def terminal_width() -> int:
    return shutil.get_terminal_size(fallback=(100, 24)).columns


def wrapped_lines(
    text: str,
    *,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> list[str]:
    return textwrap.wrap(
        text,
        width=terminal_width(),
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [initial_indent.rstrip()]


def wrapped_labeled_lines(
    label: str,
    value: Any,
    *,
    indent: str = "  ",
    valuestyles: tuple[str, ...] = (),
    color_enabled: bool = False,
) -> list[str]:
    prefix = f"{indent}{label}: "
    lines = wrapped_lines(str(value), initial_indent=prefix, subsequent_indent=" " * len(prefix))
    if not valuestyles or not color_enabled:
        return lines
    styled_lines: list[str] = []
    value_start = len(prefix)
    for line in lines:
        if len(line) <= value_start:
            styled_lines.append(line)
        else:
            styled_lines.append(
                line[:value_start]
                + style(line[value_start:], *valuestyles, color_enabled=color_enabled)
            )
    return styled_lines


def print_block(lines: list[str], *, leading_blank: bool = True) -> None:
    if leading_blank:
        print()
    for line in lines:
        print(line)


def petsc_option_groups(options_text: str) -> list[str]:
    tokens = options_text.split()
    groups: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if token.startswith("-") and next_token is not None and not next_token.startswith("-"):
            groups.append(f"{token} {next_token}")
            index += 2
        else:
            groups.append(token)
            index += 1
    return groups


def wrapped_petsc_options_lines(options_text: str, *, indent: str = "    ") -> list[str]:
    groups = petsc_option_groups(options_text)
    if not groups:
        return [f"{indent}n/a"]

    lines: list[str] = []
    current = ""
    width = terminal_width()
    for group in groups:
        candidate = group if current == "" else f"{current} {group}"
        if current and len(indent) + len(candidate) > width:
            lines.append(f"{indent}{current}")
            current = group
        else:
            current = candidate

    if current:
        lines.append(f"{indent}{current}")
    return lines


def format_solver_configuration(event: dict[str, Any]) -> str:
    solver_configuration = event.get("solver_configuration") or {}
    known_parameters = [
        ("ksp_type", "ksp"),
        ("pc_type", "pc"),
        ("ksp_gmres_restart", "restart"),
        ("pc_asm_overlap", "overlap"),
        ("sub_pc_type", "sub_pc"),
        ("sub_pc_factor_levels", "levels"),
        ("ksp_pc_side", "side"),
        ("ksp_max_it", "max_it"),
        ("pc_hypre_type", "hypre"),
        ("pc_hypre_boomeramg_cycle_type", "bamg_cycle"),
        ("pc_hypre_boomeramg_strong_threshold", "bamg_strong"),
    ]
    parts = [
        f"{label}={format_float(value) if isinstance(value, float) else value}"
        for key, label in known_parameters
        if (value := solver_configuration.get(key)) is not None
    ]
    known_keys = {key for key, _label in known_parameters}
    parts.extend(
        f"{key}={format_float(value) if isinstance(value, float) else value}"
        for key, value in sorted(solver_configuration.items())
        if key not in known_keys
    )
    return " ".join(str(part) for part in parts) if parts else "n/a"


def should_print_replay_command(event: dict[str, Any]) -> bool:
    if not event.get("failure_reason") or not event.get("replay_command_text"):
        return False
    failure_reason = str(event.get("failure_reason") or "")
    if failure_reason.startswith("replay command timed out"):
        return False
    returncode = event.get("returncode")
    if returncode not in (None, 0):
        return True
    return failure_reason.startswith("replay command returned ")


def format_replay_command_line(command_text: Any, *, color_enabled: bool = False) -> str:
    single_line_command = str(command_text).replace("\r", " ").replace("\n", " ")
    return "  reproduce: " + style(single_line_command, "cyan", color_enabled=color_enabled)


def format_solve_runtime(event: dict[str, Any]) -> str:
    solve_count = event.get("solve_count")
    try:
        solve_count_int = int(solve_count) if solve_count is not None else None
    except (TypeError, ValueError):
        solve_count_int = None

    solve_total = event.get("solve_time_sec_total")
    solve_mean = event.get("solve_time_sec_mean")
    if solve_count_int is not None and solve_count_int > 1:
        if solve_total is None and solve_mean is not None:
            solve_total = float(solve_mean) * solve_count_int
        if solve_total is not None:
            if solve_mean is None:
                solve_mean = float(solve_total) / solve_count_int
            return (
                f"{format_sec(solve_total)} "
                f"(avg={format_sec(solve_mean)}, n={solve_count_int})"
            )

    return format_sec(event.get("solve_time_sec_median"))


def format_nullspace_configuration(configuration: dict[str, Any] | None) -> str:
    if not configuration or configuration.get("mode") == "none":
        return "none"
    if configuration.get("mode") == "constant":
        return "constant"
    if configuration.get("mode") == "field":
        label = configuration.get("label", "field")
        return (
            f"{label} "
            f"(field={configuration.get('field_index')}, "
            f"block_size={configuration.get('block_size')})"
        )
    return str(configuration.get("label") or configuration)


def format_status(status: str, *, color_enabled: bool = False) -> str:
    if status in {"ok", "completed"}:
        return style(status, "green", color_enabled=color_enabled)
    if status in {"failed", "error"}:
        return style(status, "red", color_enabled=color_enabled)
    if status in {"stopped", "dry-run"}:
        return style(status, "yellow", color_enabled=color_enabled)
    return status


def format_configuration_tag(tag: Any, *, color_enabled: bool = False) -> str:
    return style(f"[{tag}]", "cyan", color_enabled=color_enabled)


def format_trial_heading(
    event: dict[str, Any],
    status: str,
    *,
    color_enabled: bool = False,
) -> str:
    trial_index = format_trial_index(event.get("trial_number"), event.get("trial_count"))
    smac_configuration_tag = event.get("smac_configuration_tag")
    heading = f"trial {trial_index}"
    if smac_configuration_tag:
        heading += f" {format_configuration_tag(smac_configuration_tag, color_enabled=color_enabled)}"
    return (
        f"{heading} {format_status(status, color_enabled=color_enabled)}  "
        f"objective: {format_objective(event, color_enabled=color_enabled)}  "
        f"(best: {format_best_objective_so_far(event, color_enabled=color_enabled)})"
    )


def finish_block_lines(event: dict[str, Any], *, color_enabled: bool = False) -> list[str]:
    status = str(event["status"])
    lines = [f"finished: {format_status(status, color_enabled=color_enabled)}"]
    summary = event.get("summary") or {}
    best_trial = summary.get("best_trial")
    if best_trial:
        best_event = {
            "objective_value": best_trial.get("objective_value"),
            "objective_name": summary.get("objective_name"),
            "solver_configuration": best_trial.get("solver_configuration"),
        }
        best_line = f"  best: trial {best_trial.get('trial_number')}"
        if best_trial.get("smac_configuration_tag"):
            best_line += (
                f" {format_configuration_tag(best_trial['smac_configuration_tag'], color_enabled=color_enabled)}"
            )
        best_line += f"  objective={format_objective(best_event, color_enabled=color_enabled)}"
        lines.append(best_line)
        lines.extend(
            wrapped_labeled_lines(
                "solver",
                format_solver_configuration(best_event),
                indent="  ",
            )
        )
        lines.append("  PETSc options:")
        lines.extend(wrapped_petsc_options_lines(str(best_trial.get("petsc_options_text") or "")))
    else:
        lines.append("  best: no successful trial yet")

    lines.append("  files:")
    lines.extend(
        wrapped_labeled_lines(
            "summary",
            display_path(event["tuning_summary_path"]),
            indent="    ",
            valuestyles=("cyan",),
            color_enabled=color_enabled,
        )
    )
    lines.extend(
        wrapped_labeled_lines(
            "trials",
            display_path(event["trial_csv_path"]),
            indent="    ",
            valuestyles=("cyan",),
            color_enabled=color_enabled,
        )
    )
    lines.extend(
        wrapped_labeled_lines(
            "rankings",
            display_path(event["ranking_csv_path"]),
            indent="    ",
            valuestyles=("cyan",),
            color_enabled=color_enabled,
        )
    )
    if best_trial:
        lines.extend(
            wrapped_labeled_lines(
                "best options",
                display_path(event["best_petsc_options_path"]),
                indent="    ",
                valuestyles=("cyan",),
                color_enabled=color_enabled,
            )
        )
    return lines


def print_tune_progress(event: dict[str, Any], *, color_enabled: bool = False) -> None:
    event_name = event.get("event")
    if event_name == "run_started":
        workers = event.get("workers", 1)
        lines = [style("KSPTune tuning run", "bold", color_enabled=color_enabled)]
        lines.extend(
            wrapped_labeled_lines(
                "snapshots",
                display_path(event["snapshot_directory_path"]),
                valuestyles=("cyan",),
                color_enabled=color_enabled,
            )
        )
        lines.extend(wrapped_labeled_lines("search space", event["parameter_search_space_name"]))
        lines.extend(
            wrapped_labeled_lines(
                "output",
                display_path(event["output_directory"]),
                valuestyles=("cyan",),
                color_enabled=color_enabled,
            )
        )
        if event.get("run_until_stopped"):
            lines.append(
                "  mode: "
                + style("until stopped", "yellow", color_enabled=color_enabled)
                + f"  workers={workers}  repeat={event['repeat']}  warmup={event['warmup']}"
            )
            lines.append(
                "  stop: "
                + style("Ctrl+C", "yellow", color_enabled=color_enabled)
                + " writes the current summary and prints the best configuration"
            )
        else:
            lines.append(
                f"  mode: {event['trials']} trials  workers={workers}  "
                f"repeat={event['repeat']}  warmup={event['warmup']}"
            )
        if "sobol_initial_design_configurations" in event:
            initial_design_parts = []
            if event.get("use_default_solver_configuration"):
                initial_design_parts.append("default")
            additional_count = event.get("additionalinitial_solver_configuration_count", 0)
            if additional_count:
                initial_design_parts.append(f"{additional_count} seeded")
            initial_design_parts.append(
                f"{event.get('sobol_initial_design_configurations', 0)} sobol"
            )
            lines.append(
                "  initial design: "
                + " + ".join(initial_design_parts)
                + f"  tunable_parameters={event.get('tunable_parameter_count', 'n/a')}"
            )
        if event.get("replay_binary"):
            replay_text = display_path(event["replay_binary"])
            if event.get("replay_binary_auto_detected"):
                replay_text += " (auto)"
            lines.extend(
                wrapped_labeled_lines(
                    "replay",
                    replay_text,
                    valuestyles=("cyan",),
                    color_enabled=color_enabled,
                )
            )
            nullspace_text = format_nullspace_configuration(event.get("nullspace_configuration"))
            if nullspace_text != "none":
                lines.extend(wrapped_labeled_lines("nullspace", nullspace_text))
            if not event.get("dry_run"):
                lines.append("  note: SMAC config IDs match tags in trial headings")
        else:
            lines.append("  replay: dry-run")
        print_block(lines, leading_blank=False)
        return

    if event_name == "trial_finished":
        wall_time = format_sec(event.get("total_wall_time_sec"))
        memory = format_megabytes(event.get("peak_memory_megabytes_max"))
        residual = format_float(event.get("final_true_relative_residual_mean"))
        solver_configuration = format_solver_configuration(event)
        failure_reason = event.get("failure_reason")
        lines = [
            format_trial_heading(
                event,
                "failed" if failure_reason else "ok",
                color_enabled=color_enabled,
            )
        ]
        lines.extend(wrapped_labeled_lines("solver", solver_configuration))
        if failure_reason:
            reason = short_text(failure_reason)
            lines.append(
                f"  runtime: subprocess={format_sec(event.get('subprocess_wall_time_sec'))}"
            )
            lines.extend(
                wrapped_labeled_lines(
                    "reason",
                    reason,
                    valuestyles=("red",),
                    color_enabled=color_enabled,
                )
            )
            if should_print_replay_command(event):
                lines.append(
                    format_replay_command_line(
                        event["replay_command_text"],
                        color_enabled=color_enabled,
                    )
                )
        else:
            setup_time = format_sec(event.get("solver_setup_time_sec"))
            solve_time = format_solve_runtime(event)
            lines.append(f"  runtime: solve={solve_time}  setup={setup_time}  wall={wall_time}")
            iterations = event.get("iterations_total")
            lines.append(f"  diagnostics: iters={iterations if iterations is not None else 'n/a'}  mem={memory}  true_rel_res={residual}")
        print_block(lines)
        return

    if event_name == "run_stopping":
        print_block(
            [
                "stopping: "
                + style("Ctrl+C", "yellow", color_enabled=color_enabled)
                + f" received after {event['trial_count']} completed trials",
                "  writing summary...",
            ]
        )
        return

    if event_name == "run_finished":
        print_block(finish_block_lines(event, color_enabled=color_enabled))


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
    parameter_search_space = load_parameter_search_space(args.parameter_search_space, seed=args.seed)
    use_color = False if args.quiet else color_enabled(args.color)
    progress_callback = None if args.quiet else lambda event: print_tune_progress(event, color_enabled=use_color)
    run_until_stopped = args.run_until_stopped or args.trials is None
    result = run_tuning(
        snapshot_directory=args.snapshot_directory,
        parameter_search_space=parameter_search_space,
        output_directory=args.output_directory,
        replay_binary=args.replay_binary,
        mpiexec=args.mpiexec,
        mpi_processes=args.np,
        threads_per_rank=args.threads_per_rank,
        workers=args.workers,
        trials=args.trials,
        repeat=args.repeat,
        warmup=args.warmup,
        timeout_sec=args.timeout_sec,
        objective_name=args.objective,
        seed=args.seed,
        dry_run=args.dry_run,
        run_until_stopped=run_until_stopped,
        nullspace=args.nullspace,
        nullspace_actions=args.nullspace_actions,
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
        raise ValueError("pass the snapshot directory either positionally or with --snapshot-directory")

    output_path = (
        args.output
        or Path(snapshot_directory)
        / "ksptune_snapshot_analysis"
        / "snapshot_analysis_summary.yaml"
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
    configspace_path = tuning_run_directory / "configspace.yaml"
    if not configspace_path.exists():
        configspace_path = tuning_run_directory / "parameter_search_space.yaml"
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
    tune.add_argument("--mpiexec", default="mpiexec")
    tune.add_argument("--np", type=int, default=1)
    tune.add_argument("--threads-per-rank", type=int, default=1)
    tune.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel SMAC workers. Each worker may launch --np MPI ranks.",
    )
    tune.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Run a fixed number of trials. Omit to run until stopped with Ctrl+C.",
    )
    tune.add_argument("--repeat", type=int, default=1)
    tune.add_argument("--warmup", type=int, default=0)
    tune.add_argument("--timeout-sec", type=float)
    tune.add_argument("--objective", default="solve_time_sec_mean")
    tune.add_argument(
        "--nullspace",
        default="from-metadata",
        help=(
            "Replay nullspace: from-metadata, none, constant, or "
            "field:<index>[,block_size=<n>]."
        ),
    )
    tune.add_argument(
        "--nullspace-actions",
        help=(
            "Replay nullspace actions: metadata, matrix, transpose, near, remove-rhs, "
            "all, none, or a comma-separated list. Aliases: right=matrix, left=transpose."
        ),
    )
    tune.add_argument("--seed", type=int, default=1)
    tune.add_argument("--dry-run", action="store_true")
    tune.add_argument(
        "--run-until-stopped",
        action="store_true",
        help="Run until stopped with Ctrl+C. This is the default when --trials is omitted.",
    )
    tune.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize tune output. Default: auto.",
    )
    tune.add_argument("--quiet", action="store_true")
    tune.set_defaults(func=cmd_tune)

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
    analyze_snapshots.add_argument("--solve-index", type=int)
    analyze_snapshots.add_argument(
        "--nullspace",
        default="none",
        help=(
            "Candidate nullspace to test: none, constant, or "
            "field:<index>[,block_size=<n>]."
        ),
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
