from __future__ import annotations

import os
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .trial_records import BAD_COST

ANSI_STYLES = {
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


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


def format_sec(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def format_bytes(value: float | None) -> str:
    if value is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while abs(value) >= 1024.0 and unit_index + 1 < len(units):
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{value:.0f}{units[unit_index]}"
    return f"{value:.1f}{units[unit_index]}"


def format_megabytes(value: float | None) -> str:
    if value is None:
        return "n/a"
    return format_bytes(value * 1024.0 * 1024.0)


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000 or (value != 0.0 and abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.6g}"


def format_compact_number(value: float) -> str:
    if value != 0.0 and abs(value) < 0.001:
        coefficient, exponent = f"{value:.6e}".split("e")
        coefficient = coefficient.rstrip("0").rstrip(".")
        return f"{coefficient}e{int(exponent)}"
    return f"{value:.6g}"


def format_policy_value(value: Any) -> str:
    if value is None:
        return "PETSc-default"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_compact_number(value)
    return str(value)


def format_metric_fields(
    values: dict[str, Any],
    fields: Iterable[tuple[str, str, Callable[[Any], str]]],
    *,
    include_missing: bool = False,
) -> list[str]:
    return [
        f"{label}={formatter(values.get(name))}"
        for name, label, formatter in fields
        if include_missing or values.get(name) is not None
    ]


def ordered_fields(values: dict[str, Any], labels: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (name, label) for name, label in labels.items() if name in values
    ] + [(name, name) for name in sorted(values.keys() - labels.keys())]


def format_objective(
    event: dict[str, Any], *, best: bool = False, color_enabled: bool = False
) -> str:
    value = event.get("best_objective_value_so_far" if best else "objective_value")
    if value is None:
        return "n/a"
    bad_cost = value >= BAD_COST * 0.1
    text = "bad-cost" if bad_cost else format_float(value)
    if not bad_cost and "_sec" in (event.get("objective_name") or ""):
        text += "s"
    if best:
        color = "cyan"
    elif bad_cost:
        color = "red"
    elif event.get("is_new_best_so_far") or event.get("best_objective_value_so_far") is None:
        color = "green"
    else:
        color = "yellow"
    return style(text, "bold", color, color_enabled=color_enabled)


def format_trial_index(trial_number: int | None, trial_count: int | None) -> str:
    if trial_number is None:
        return "?/?"
    if trial_count is None:
        return f"{trial_number:03d}/..."
    width = max(3, len(str(trial_count)))
    return f"{trial_number:0{width}d}/{trial_count:0{width}d}"


def short_text(value: Any, max_length: int = 120) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def display_path(value: str | Path | None) -> str:
    if value is None:
        return "n/a"
    path = Path(value)
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
    sys.stdout.flush()


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
    known_parameters = {
        "ksp_type": "ksp",
        "pc_type": "pc",
        "ksp_pc_side": "side",
        "ksp_norm_type": "norm",
        "ksp_rtol": "rtol",
        "ksp_atol": "atol",
        "ksp_dtol": "dtol",
        "ksp_gmres_restart": "restart",
        "pc_asm_overlap": "overlap",
        "sub_pc_type": "sub_pc",
        "sub_pc_factor_levels": "levels",
        "ksp_max_it": "max_it",
        "pc_hypre_type": "hypre",
        "pc_hypre_boomeramg_cycle_type": "bamg_cycle",
        "pc_hypre_boomeramg_strong_threshold": "bamg_strong",
    }
    parts = []
    for key, label in ordered_fields(solver_configuration, known_parameters):
        value = solver_configuration[key]
        if value is None and key in known_parameters:
            continue
        if key in {"ksp_atol", "ksp_rtol", "ksp_dtol"}:
            value = format_policy_value(value)
        elif isinstance(value, float):
            value = format_float(value)
        parts.append(f"{label}={value}")
    return " ".join(parts) if parts else "n/a"


def format_numerics_summary(event: dict[str, Any]) -> str:
    solver_configuration = event.get("default_solver_configuration") or {}
    return (
        f"ksp={format_policy_value(solver_configuration.get('ksp_type'))} "
        f"side={format_policy_value(solver_configuration.get('ksp_pc_side'))} "
        f"norm={format_policy_value(solver_configuration.get('ksp_norm_type'))} "
        f"rtol={format_policy_value(solver_configuration.get('ksp_rtol'))} "
        f"atol={format_policy_value(solver_configuration.get('ksp_atol'))} "
        f"max_it={format_policy_value(solver_configuration.get('ksp_max_it'))} "
        f"true_abs_gate={format_policy_value(event.get('max_true_residual_norm'))} "
        f"true_rel_gate={format_policy_value(event.get('max_true_relative_residual'))}"
    )


def format_timeout_summary(event: dict[str, Any]) -> str | None:
    soft_timeout_sec = event.get("soft_timeout_sec")
    hard_timeout_sec = event.get("hard_timeout_sec")
    if hard_timeout_sec is None:
        hard_timeout_sec = event.get("effective_hard_timeout_sec")
    startup_timeout_sec = event.get("replay_startup_timeout_sec")
    if soft_timeout_sec is None and hard_timeout_sec is None and startup_timeout_sec is None:
        return None
    soft_timeout = (
        "none"
        if soft_timeout_sec is None or soft_timeout_sec < 0.0
        else f"{format_policy_value(soft_timeout_sec)}s"
    )
    hard_timeout = (
        "none" if hard_timeout_sec is None else f"{format_policy_value(hard_timeout_sec)}s"
    )
    startup_timeout = (
        "same-as-request"
        if startup_timeout_sec is None
        else f"{format_policy_value(startup_timeout_sec)}s"
    )
    return (
        f"solve_soft={soft_timeout} "
        f"request_wall_hard={hard_timeout} "
        f"startup_wall_hard={startup_timeout}"
    )


def format_ksp_tolerances(event: dict[str, Any]) -> str | None:
    if not any(
        event.get(key) is not None
        for key in (
            "ksp_rtol",
            "ksp_atol",
            "ksp_dtol",
            "ksp_max_it",
            "max_true_residual_norm",
            "max_true_relative_residual",
        )
    ):
        return None
    return (
        f"rtol={format_policy_value(event.get('ksp_rtol'))} "
        f"atol={format_policy_value(event.get('ksp_atol'))} "
        f"dtol={format_policy_value(event.get('ksp_dtol'))} "
        f"max_it={format_policy_value(event.get('ksp_max_it'))} "
        f"true_abs_gate<={format_policy_value(event.get('max_true_residual_norm'))} "
        f"true_rel_gate<={format_policy_value(event.get('max_true_relative_residual'))}"
    )


def format_diagnostics_summary(event: dict[str, Any]) -> str:
    parts = [f"iters={format_iterations(event)}"]
    reason = event.get("reason")
    if reason is None:
        reason = event.get("reason_code")
    if reason is not None:
        parts.append(f"reason={reason}")
    return " ".join(parts)


def format_residual_summary_lines(event: dict[str, Any]) -> list[str]:
    true_fields = (
        ("initial_true_residual_norm_mean", "abs_initial", format_float),
        ("initial_true_relative_residual_mean", "rel_initial", format_float),
        ("final_true_residual_norm_mean", "abs_final", format_float),
        ("final_true_relative_residual_mean", "rel_final", format_float),
    )
    ksp_fields = (
        ("initial_ksp_residual_norm_mean", "abs_initial", format_float),
        ("final_ksp_residual_norm_mean", "abs_final", format_float),
        ("final_ksp_relative_residual_norm_mean", "rel_final", format_float),
    )
    reference_fields = (("ksp_rtol_reference_norm_mean", "rtol_ref", format_float),)
    if not any(event.get(name) is not None for name, _, _ in true_fields + ksp_fields + reference_fields):
        return []

    solver_configuration = event.get("solver_configuration") or {}
    ksp_norm_type = (
        event.get("ksp_residual_norm_type") or solver_configuration.get("ksp_norm_type") or "PETSc"
    )
    true_parts = format_metric_fields(event, true_fields, include_missing=True)
    ksp_parts = [f"norm={ksp_norm_type}"]
    ksp_parts.extend(format_metric_fields(event, reference_fields))
    ksp_parts.extend(format_metric_fields(event, ksp_fields, include_missing=True))
    return [
        "  true residual: " + " ".join(true_parts),
        "  ksp residual: " + " ".join(ksp_parts),
    ]


def should_print_replay_command(event: dict[str, Any]) -> bool:
    return bool(event.get("replay_command_text") and event.get("status") in {"process_error", "oom"})


def format_replay_command_line(command_text: str, *, color_enabled: bool = False) -> str:
    single_line_command = command_text.replace("\r", " ").replace("\n", " ")
    return "  reproduce: " + style(single_line_command, "cyan", color_enabled=color_enabled)


def format_replay_log_paths(event: dict[str, Any]) -> str | None:
    parts = []
    stdout_path = event.get("stdout_path")
    stderr_path = event.get("stderr_path")
    if stdout_path:
        parts.append(f"stdout={display_path(stdout_path)}")
    if stderr_path:
        parts.append(f"stderr={display_path(stderr_path)}")
    return " ".join(parts) if parts else None


def format_solve_runtime(event: dict[str, Any]) -> str:
    solve_count = event.get("solve_count")

    solve_total = event.get("solve_time_sec_total")
    solve_mean = event.get("solve_time_sec_mean")
    solve_median = event.get("solve_time_sec_median")
    solve_min = event.get("solve_time_sec_min")
    solve_max = event.get("solve_time_sec_max")
    solve_sem = event.get("solve_time_sec_sem")
    solve_relative_sem = event.get("solve_time_sec_relative_sem")
    if solve_count is not None and solve_count > 1:
        if solve_total is not None:
            details = [f"samples={solve_count}", f"avg={format_sec(solve_mean)}"]
            if solve_sem is not None:
                sem_text = f"sem={format_sec(solve_sem)}"
                if solve_relative_sem is not None:
                    sem_text += f" rel_sem={100.0 * solve_relative_sem:.2g}%"
                details.append(sem_text)
            if solve_min is not None and solve_max is not None:
                details.append(f"min-max={format_sec(solve_min)}-{format_sec(solve_max)}")
            return f"{format_sec(solve_total)} ({', '.join(details)})"

    return format_sec(solve_median)


def format_setup_runtime(event: dict[str, Any]) -> str:
    setup_time = event.get("solver_setup_time_sec")
    actual_setup_time = event.get("solver_setup_time_sec_actual")
    cache_hits = event.get("ksp_setup_cache_hits") or 0
    cache_misses = event.get("ksp_setup_cache_misses") or 0
    setup_time_differs = setup_time != actual_setup_time

    if (
        setup_time is not None
        and actual_setup_time is not None
        and (cache_hits > 0 or setup_time_differs)
    ):
        return f"{format_sec(actual_setup_time)} (ksp-cache={cache_hits}/{cache_misses})"
    return format_sec(setup_time)


def format_replay_timing_summary(event: dict[str, Any]) -> str | None:
    parts = format_metric_fields(event, (
        ("subprocess_wall_time_sec", "request", format_sec),
        ("matrix_load_time_sec", "data-load", format_sec),
        ("matrix_prepare_time_sec", "matrix", format_sec),
        ("vector_prepare_time_sec", "vectors", format_sec),
        ("nullspace_time_sec", "nullspace", format_sec),
        ("true_residual_time_sec", "true-residuals", format_sec),
    ))

    for field, label in (
        ("replay_other_time_sec", "replay-other"),
        ("outer_overhead_time_sec", "outer-overhead"),
    ):
        value = event.get(field)
        if value is not None and value > 0.001:
            parts.append(f"{label}={format_sec(value)}")

    soft_timeout = event.get("soft_timeout_sec")
    if soft_timeout is not None and soft_timeout >= 0.0:
        parts.append(f"solve_soft={format_policy_value(soft_timeout)}s")
    effective_timeout = event.get("effective_hard_timeout_sec")
    if effective_timeout is not None:
        timeout_label = (
            "startup_wall_hard" if event.get("used_startup_timeout") else "request_wall_hard"
        )
        parts.append(f"{timeout_label}={format_policy_value(effective_timeout)}s")

    matrix_cache_hits = event.get("matrix_cache_hits")
    matrix_cache_misses = event.get("matrix_cache_misses")
    if matrix_cache_hits is not None or matrix_cache_misses is not None:
        parts.append(
            "matrix-cache="
            f"{format_policy_value(matrix_cache_hits or 0)}/"
            f"{format_policy_value(matrix_cache_misses or 0)}"
        )

    if not parts:
        return None
    return " ".join(parts)


def format_iterations(event: dict[str, Any]) -> str:
    iterations_total = event.get("iterations_total")
    if iterations_total is None:
        return "n/a"

    solve_count = event.get("solve_count")

    details = []
    if solve_count is not None and solve_count > 0:
        details.append(f"samples={solve_count}")
        details.append(f"mean={format_float(event.get('iterations_mean'))}")

    text = format_float(iterations_total)
    return f"{text} ({', '.join(details)})" if details else text


def format_mpi_summary(event: dict[str, Any]) -> str | None:
    parts = format_metric_fields(event, (
        ("solve_mpi_message_count", "msgs", format_float),
        ("solve_mpi_message_bytes", "bytes", format_bytes),
        ("solve_mpi_reduction_count", "reductions", format_float),
    ))
    if not parts:
        return None
    message_count = event.get("solve_mpi_message_count")
    mean_message_bytes = event.get("solve_mpi_message_bytes_mean")
    if mean_message_bytes is not None and message_count is not None and message_count > 0:
        parts.append(f"avg={format_bytes(mean_message_bytes)}/msg")
    return "solve " + " ".join(parts)


def format_memory_summary(event: dict[str, Any]) -> str | None:
    if event.get("rss_request_peak_sample_mb_sum") is None:
        return None
    parts = format_metric_fields(event, (
        ("rss_request_peak_sample_mb_sum", "sample_peak", format_megabytes),
        ("rss_request_delta_mb_sum", "request_delta", format_megabytes),
        ("rss_setup_delta_mb_sum", "pcsetup_delta", format_megabytes),
        ("rss_solve_delta_mb_sum", "solve_delta", format_megabytes),
        ("rss_peak_sample_mb_max_node", "max_node", format_megabytes),
        ("rss_peak_sample_mb_max_rank", "max_rank", format_megabytes),
    ))
    return "rss_sum " + " ".join(parts)


PC_DIAGNOSTIC_METRIC_FIELDS = {
    "levels": "levels",
    "grid_complexity": "grid",
    "operator_complexity": "op",
    "interpolation_complexity": "interp",
    "max_operator_nonzeros_per_row": "max_op_nnz/row",
    "max_interpolation_nonzeros_per_row": "max_interp_nnz/row",
    "min_active_ranks": "min_active_ranks",
    "finest_nonzeros": "fine_nnz",
    "total_nonzeros": "total_nnz",
    "coarsest_rows": "coarse_rows",
    "interpolation_nonzeros": "interp_nnz",
    "max_row_nonzeros": "max_row_nnz",
    "offdiag_nonzeros": "offdiag_nnz",
    "global_blocks": "global_blocks",
    "local_blocks": "local_blocks",
    "global_subdomains": "global_subdomains",
    "local_subdomains": "local_subdomains",
}

PC_DIAGNOSTIC_STRING_FIELDS = {
    "asm_type": "asm",
    "sub_ksp_type": "sub_ksp",
    "sub_pc_type": "sub_pc",
}


def format_pc_diagnostic_record(record: dict[str, Any]) -> str | None:
    parts = [record.get("pc_type") or "unknown"]
    setup_index = record.get("setup_index")
    if setup_index is not None:
        parts.append(f"setup={setup_index}")

    for key, labels, formatter in (
        ("metrics", PC_DIAGNOSTIC_METRIC_FIELDS, format_float),
        ("string_metrics", PC_DIAGNOSTIC_STRING_FIELDS, str),
    ):
        values = record.get(key, {})
        parts.extend(format_metric_fields(values, (
            (name, label, formatter)
            for name, label in ordered_fields(values, labels) if values[name] != ""
        )))
    return " ".join(parts) if len(parts) > 1 else None


def format_pc_diagnostic_summaries(event: dict[str, Any]) -> list[str]:
    return [
        summary for record in event.get("pc_diagnostics", [])
        if (summary := format_pc_diagnostic_record(record)) is not None
    ]


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


def format_trial_label(event: dict[str, Any], *, color_enabled: bool = False) -> str:
    trial_index = format_trial_index(event.get("trial_number"), event.get("trial_count"))
    smac_configuration_tag = event.get("smac_configuration_tag")
    heading = f"trial {trial_index}"
    if smac_configuration_tag:
        heading += (
            f" {format_configuration_tag(smac_configuration_tag, color_enabled=color_enabled)}"
        )
    return heading


def finish_block_lines(event: dict[str, Any], *, color_enabled: bool = False) -> list[str]:
    status = event["status"]
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
            best_line += f" {format_configuration_tag(best_trial['smac_configuration_tag'], color_enabled=color_enabled)}"
        best_line += f"  objective={format_objective(best_event, color_enabled=color_enabled)}"
        lines.append(best_line)
        lines.extend(wrapped_labeled_lines("solver", format_solver_configuration(best_event)))
        lines.append("  PETSc options:")
        lines.extend(wrapped_petsc_options_lines(best_trial.get("petsc_options_text") or ""))
    else:
        lines.append("  best: no successful trial yet")

    lines.append("  files:")
    files = [("summary", "tuning_summary_path"), ("trials", "trial_csv_path"), ("rankings", "ranking_csv_path")]
    if best_trial:
        files.append(("best options", "best_petsc_options_path"))
    for label, field in files:
        lines.extend(wrapped_labeled_lines(
            label, display_path(event[field]), indent="    ",
            valuestyles=("cyan",), color_enabled=color_enabled,
        ))
    return lines


def run_started_lines(event: dict[str, Any], *, color_enabled: bool = False) -> list[str]:
    lines = [style("KSPTune tuning run", "bold", color_enabled=color_enabled)]
    for label, value, styles in (
        ("snapshots", display_path(event["snapshot_directory"]), ("cyan",)),
        ("search space", event["parameter_search_space_name"], ()),
        ("output", display_path(event["output_directory"]), ("cyan",)),
    ):
        lines.extend(wrapped_labeled_lines(
            label, value, valuestyles=styles, color_enabled=color_enabled,
        ))
    mode = (
        style("until stopped", "yellow", color_enabled=color_enabled)
        if event.get("run_until_stopped") else f"{event['trials']} trials"
    )
    lines.append(
        f"  mode: {mode}  workers={event.get('workers', 1)}  "
        f"repeat={event['repeat']}  warmup={event['warmup']}"
    )
    if event.get("run_until_stopped"):
        lines.append(
            "  stop: " + style("Ctrl+C", "yellow", color_enabled=color_enabled)
            + " writes the current summary and prints the best configuration"
        )
    if "sobol_initial_design_configurations" in event:
        additional_count = event.get("additionalinitial_solver_configuration_count", 0)
        startup_count = event.get("startup_initial_solver_configuration_count", 0)
        default_count = 1 if event.get("use_default_solver_configuration") else 0
        sobol_count = event.get("sobol_initial_design_configurations", 0)
        remaining_seeded_count = max(0, additional_count - startup_count)
        initial_design_total = startup_count + remaining_seeded_count + default_count + sobol_count
        lines.append(
            f"  initial design: total={initial_design_total} | "
            f"startup={startup_count} seeded={remaining_seeded_count} "
            f"default={default_count} sobol={sobol_count} (in order) | "
            f"tunable_parameters={event.get('tunable_parameter_count', 'n/a')}"
        )
    lines.append(f"  numerics: {format_numerics_summary(event)}")
    timeout_summary = format_timeout_summary(event)
    if timeout_summary is not None:
        lines.append(f"  timeouts: {timeout_summary}")
    if event.get("fast_fail"):
        lines.append(f"  diagnostic: fast-fail  ksp_max_it={event.get('fast_fail_ksp_max_it', 1)}")
    if event.get("replay_binary"):
        replay_text = display_path(event["replay_binary"])
        if event.get("replay_binary_auto_detected"):
            replay_text += " (auto)"
        lines.extend(wrapped_labeled_lines(
            "replay", replay_text, valuestyles=("cyan",), color_enabled=color_enabled,
        ))
        nullspace_text = format_nullspace_configuration(event.get("nullspace_configuration"))
        if nullspace_text != "none":
            lines.extend(wrapped_labeled_lines("nullspace", nullspace_text))
        if not event.get("dry_run"):
            lines.append("  note: SMAC config IDs match tags in trial headings")
    else:
        lines.append("  replay: dry-run")
    return lines


def trial_finished_lines(event: dict[str, Any], *, color_enabled: bool = False) -> list[str]:
    failure_reason = event.get("failure_reason")
    status = format_status("failed" if failure_reason else "ok", color_enabled=color_enabled)
    lines = [
        f"{format_trial_label(event, color_enabled=color_enabled)} {status}  "
        f"objective: {format_objective(event, color_enabled=color_enabled)}  "
        f"(best: {format_objective(event, best=True, color_enabled=color_enabled)})"
    ]
    lines.extend(wrapped_labeled_lines("solver", format_solver_configuration(event)))
    if failure_reason:
        lines.append(f"  runtime: subprocess={format_sec(event.get('subprocess_wall_time_sec'))}")
        lines.extend(wrapped_labeled_lines(
            "reason", short_text(failure_reason), valuestyles=("red",), color_enabled=color_enabled,
        ))
        log_paths = format_replay_log_paths(event)
        if log_paths is not None:
            lines.extend(wrapped_labeled_lines("logs", log_paths))
    else:
        lines.append(
            f"  runtime: solve={format_solve_runtime(event)}  setup={format_setup_runtime(event)}  "
            f"wall={format_sec(event.get('total_wall_time_sec'))}"
        )
    for label, summary in (
        ("replay", None if failure_reason else format_replay_timing_summary(event)),
        ("memory", format_memory_summary(event)),
    ):
        if summary is not None:
            lines.append(f"  {label}: {summary}")
    lines.extend(f"  pc: {summary}" for summary in format_pc_diagnostic_summaries(event))
    if not failure_reason:
        mpi_summary = format_mpi_summary(event)
        if mpi_summary is not None:
            lines.append(f"  mpi: {mpi_summary}")
    residual_lines = format_residual_summary_lines(event)
    if not failure_reason or residual_lines:
        tolerance_summary = format_ksp_tolerances(event)
        if tolerance_summary is not None:
            lines.append(f"  tolerances: {tolerance_summary}")
        lines.append(f"  diagnostics: {format_diagnostics_summary(event)}")
        lines.extend(residual_lines)
    if should_print_replay_command(event):
        lines.append(format_replay_command_line(
            event["replay_command_text"], color_enabled=color_enabled,
        ))
    return lines


def print_tune_progress(
    event: dict[str, Any], *, color_enabled: bool = False, timestamp: str
) -> None:
    event_name = event.get("event")
    if event_name == "run_started":
        lines = run_started_lines(event, color_enabled=color_enabled)
    elif event_name == "trial_queued":
        line = f"[{timestamp}] {format_trial_label(event, color_enabled=color_enabled)} queued"
        timeout_summary = format_timeout_summary(event)
        if timeout_summary is not None:
            line += f" (timeouts: {timeout_summary})"
        lines = [line]
    elif event_name == "trial_finished":
        lines = trial_finished_lines(event, color_enabled=color_enabled)
    elif event_name == "run_stopping":
        lines = [
            "stopping: " + style("Ctrl+C", "yellow", color_enabled=color_enabled)
            + f" received after {event['trial_count']} completed trials",
            "  writing summary...",
        ]
    elif event_name == "run_finished":
        lines = finish_block_lines(event, color_enabled=color_enabled)
    else:
        return
    print_block(lines, leading_blank=event_name != "run_started")
