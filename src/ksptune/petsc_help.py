from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


HYPRE_SUBTYPES = {"ams", "ads", "boomeramg", "euclid", "ilu", "parasails", "pilut"}


def build_petsc_options_help_command(
    *,
    replay_binary: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
) -> list[str]:
    command = [str(replay_binary), "-replay_options_help", *petsc_options]
    if mpi_processes > 1 or mpiexec_args or mpiexec != "mpiexec":
        return [mpiexec, *(mpiexec_args or []), "-n", str(mpi_processes), *command]
    return command


def normalize_petsc_help_options(
    *,
    ksp_type: str | None = None,
    pc_type: str | None = None,
    pc_hypre_type: str | None = None,
    extra_petsc_options: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    options: list[str] = []
    notes: list[str] = []
    if ksp_type:
        options.extend(["-ksp_type", ksp_type])

    resolved_pc_type = pc_type
    resolved_pc_hypre_type = pc_hypre_type
    if resolved_pc_type in HYPRE_SUBTYPES and resolved_pc_type != "hypre":
        if resolved_pc_hypre_type is None:
            resolved_pc_hypre_type = resolved_pc_type
        resolved_pc_type = "hypre"
        notes.append(
            f"interpreted pc_type={pc_type} as -pc_type hypre "
            f"-pc_hypre_type {resolved_pc_hypre_type}"
        )

    if resolved_pc_type:
        options.extend(["-pc_type", resolved_pc_type])
    if resolved_pc_hypre_type:
        options.extend(["-pc_hypre_type", resolved_pc_hypre_type])

    options.extend(extra_petsc_options or [])
    return options, notes


def value_after_option(options: list[str], option_name: str) -> str | None:
    for index, option in enumerate(options):
        if option == option_name and index + 1 < len(options):
            return options[index + 1]
        prefix = option_name + "="
        if option.startswith(prefix):
            return option[len(prefix) :]
    return None


def infer_option_filters(petsc_options: list[str]) -> list[str]:
    filters: list[str] = []
    ksp_type = value_after_option(petsc_options, "-ksp_type")
    pc_type = value_after_option(petsc_options, "-pc_type")
    hypre_type = value_after_option(petsc_options, "-pc_hypre_type")

    if ksp_type:
        filters.extend(["-ksp_type", f"-ksp_{ksp_type}"])
    if pc_type:
        filters.append("-pc_type")
    if pc_type == "hypre":
        filters.append("-pc_hypre")
        if hypre_type:
            filters.append(f"-pc_hypre_{hypre_type}")
    elif pc_type == "gamg":
        filters.extend(["-pc_gamg", "-pc_mg", "-mg_"])
    elif pc_type:
        filters.append(f"-pc_{pc_type}")

    return deduplicate_preserving_order(filters)


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def extract_petsc_option_lines(output: str) -> list[str]:
    lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") and set(stripped) != {"-"}:
            lines.append(stripped)
    return lines


def filter_petsc_option_lines(
    option_lines: list[str],
    *,
    filters: list[str],
) -> list[str]:
    if not filters:
        return option_lines
    return [
        line
        for line in option_lines
        if any(line.startswith(option_filter) for option_filter in filters)
    ]


def replay_help_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    return environment


def subprocess_output_to_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def combined_subprocess_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    return "\n".join(
        text
        for text in (
            subprocess_output_to_text(stdout),
            subprocess_output_to_text(stderr),
        )
        if text
    )


def build_petsc_options_help_result(
    *,
    command: list[str],
    returncode: int,
    petsc_options: list[str],
    filters: list[str] | None,
    raw: bool,
    output: str,
    timed_out: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    option_lines = extract_petsc_option_lines(output)
    resolved_filters = filters if filters is not None else infer_option_filters(petsc_options)
    filtered_lines = option_lines if raw else filter_petsc_option_lines(
        option_lines,
        filters=resolved_filters,
    )
    result = {
        "command": command,
        "returncode": returncode,
        "petsc_options": petsc_options,
        "filters": resolved_filters,
        "raw_output": output,
        "option_lines": option_lines,
        "filtered_lines": filtered_lines,
        "timed_out": timed_out,
    }
    if error is not None:
        result["error"] = error
    return result


def query_petsc_options_help(
    *,
    replay_binary: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    filters: list[str] | None = None,
    raw: bool = False,
    timeout_sec: float | None = 30.0,
) -> dict[str, Any]:
    command = build_petsc_options_help_command(
        replay_binary=replay_binary,
        petsc_options=petsc_options,
        mpiexec=mpiexec,
        mpiexec_args=mpiexec_args,
        mpi_processes=mpi_processes,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout_sec,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=replay_help_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        error = f"PETSc options help timed out after {timeout_sec:g}s."
        output = combined_subprocess_output(exc.stdout, exc.stderr)
        if output:
            output = f"{output}\n{error}"
        else:
            output = error
        return build_petsc_options_help_result(
            command=command,
            returncode=124,
            petsc_options=petsc_options,
            filters=filters,
            raw=raw,
            output=output,
            timed_out=True,
            error=error,
        )
    except OSError as exc:
        error = f"Could not start PETSc options help command: {exc}"
        return build_petsc_options_help_result(
            command=command,
            returncode=127,
            petsc_options=petsc_options,
            filters=filters,
            raw=raw,
            output=error,
            error=error,
        )

    output = combined_subprocess_output(completed.stdout, completed.stderr)
    return build_petsc_options_help_result(
        command=command,
        returncode=completed.returncode,
        petsc_options=petsc_options,
        filters=filters,
        raw=raw,
        output=output,
    )


def petsc_options_help_to_json(result: dict[str, Any]) -> str:
    data = {
        key: value
        for key, value in result.items()
        if key != "raw_output"
    }
    return json.dumps(data, indent=2, sort_keys=True)
