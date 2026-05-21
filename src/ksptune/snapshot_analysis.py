from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from .nullspaces import parse_nullspace
from .replay import replay_environment
from .snapshot_collections import (
    create_snapshot_collection_from_directory,
    load_snapshots_from_directory,
    summarize_snapshot_directory,
    write_resolved_snapshot_collection,
)

SNAPSHOT_ANALYSIS_BINARY_NAME = "ksptune-petsc-snapshot-analysis"


def find_default_petsc_snapshot_analysis_binary() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    local_binary = (
        project_root
        / "build/cpp/petsc_snapshot_analysis"
        / SNAPSHOT_ANALYSIS_BINARY_NAME
    )
    if local_binary.exists():
        return local_binary.resolve()

    path_binary = shutil.which(SNAPSHOT_ANALYSIS_BINARY_NAME)
    if path_binary:
        return Path(path_binary).resolve()
    return None


def resolve_petsc_snapshot_analysis_binary(
    petsc_snapshot_analysis_binary: str | Path | None,
) -> Path | None:
    if petsc_snapshot_analysis_binary is None:
        return find_default_petsc_snapshot_analysis_binary()

    binary_text = str(petsc_snapshot_analysis_binary)
    if os.sep not in binary_text:
        path_binary = shutil.which(binary_text)
        if path_binary:
            return Path(path_binary).resolve()

    binary_path = Path(binary_text)
    if not binary_path.exists():
        raise ValueError(f"PETSc snapshot analysis binary does not exist: {binary_path}")
    return binary_path.resolve()


def snapshot_analysis_nullspace_options(nullspace: str | None) -> list[str]:
    configuration = parse_nullspace(nullspace)
    if configuration["mode"] == "none":
        return []
    if configuration["mode"] == "constant":
        return ["-analysis_constant_nullspace"]
    if configuration["mode"] == "field":
        return [
            "-analysis_field_nullspace",
            str(configuration["field_index"]),
            "-analysis_field_nullspace_block_size",
            str(configuration["block_size"]),
        ]
    raise ValueError(f"Unsupported nullspace configuration: {configuration['label']}")


def build_petsc_snapshot_analysis_command(
    *,
    petsc_snapshot_analysis_binary: str | Path,
    snapshot_collection_path: str | Path,
    petsc_snapshot_analysis_result_path: str | Path,
    mpiexec: str = "mpiexec",
    mpi_processes: int = 1,
    nullspace: str | None = "none",
    solve_index: int | None = None,
    rhs_compatibility_tolerance: float = 1.0e-10,
    nullspace_residual_tolerance: float = 1.0e-10,
) -> list[str]:
    analysis_command = [
        str(petsc_snapshot_analysis_binary),
        "-snapshot_collection",
        str(snapshot_collection_path),
        "-analysis_json_out",
        str(petsc_snapshot_analysis_result_path),
        "-analysis_rhs_compatibility_tolerance",
        str(rhs_compatibility_tolerance),
        "-analysis_nullspace_residual_tolerance",
        str(nullspace_residual_tolerance),
    ]
    if solve_index is not None:
        analysis_command.extend(["-analysis_solve_target", str(solve_index)])
    analysis_command.extend(snapshot_analysis_nullspace_options(nullspace))

    if mpi_processes > 1:
        return [mpiexec, "-n", str(mpi_processes), *analysis_command]
    return analysis_command


def run_petsc_snapshot_analysis(
    *,
    petsc_snapshot_analysis_binary: str | Path,
    snapshot_collection_path: str | Path,
    output_directory: str | Path,
    mpiexec: str = "mpiexec",
    mpi_processes: int = 1,
    threads_per_rank: int = 1,
    nullspace: str | None = "none",
    solve_index: int | None = None,
    rhs_compatibility_tolerance: float = 1.0e-10,
    nullspace_residual_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    petsc_result_path = output_path / "petsc_snapshot_analysis.json"
    command = build_petsc_snapshot_analysis_command(
        petsc_snapshot_analysis_binary=petsc_snapshot_analysis_binary,
        snapshot_collection_path=snapshot_collection_path,
        petsc_snapshot_analysis_result_path=petsc_result_path,
        mpiexec=mpiexec,
        mpi_processes=mpi_processes,
        nullspace=nullspace,
        solve_index=solve_index,
        rhs_compatibility_tolerance=rhs_compatibility_tolerance,
        nullspace_residual_tolerance=nullspace_residual_tolerance,
    )

    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=replay_environment(threads_per_rank),
    )
    subprocess_wall_time_sec = time.perf_counter() - start_time

    petsc_result: dict[str, Any] = {}
    json_error: str | None = None
    if petsc_result_path.exists():
        try:
            petsc_result = json.loads(petsc_result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            json_error = str(exc)
    else:
        json_error = f"missing PETSc snapshot analysis JSON: {petsc_result_path}"

    status = "ok" if completed.returncode == 0 and json_error is None else "failed"
    return {
        "available": True,
        "status": status,
        "binary": str(Path(petsc_snapshot_analysis_binary).resolve()),
        "command": command,
        "command_text": shlex.join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "subprocess_wall_time_sec": subprocess_wall_time_sec,
        "result_path": str(petsc_result_path.resolve()),
        "json_error": json_error,
        "result": petsc_result,
    }


def analyze_snapshots(
    snapshot_directory: str | Path,
    output_path: str | Path | None = None,
    *,
    petsc_snapshot_analysis_binary: str | Path | None = None,
    mpiexec: str = "mpiexec",
    mpi_processes: int = 1,
    threads_per_rank: int = 1,
    nullspace: str | None = "none",
    solve_index: int | None = None,
    rhs_compatibility_tolerance: float = 1.0e-10,
    nullspace_residual_tolerance: float = 1.0e-10,
    metadata_only: bool = False,
) -> dict[str, Any]:
    directory = Path(snapshot_directory).resolve()
    snapshots = load_snapshots_from_directory(directory)
    summary = summarize_snapshot_directory(directory)
    rows = [snapshot["rows"] for snapshot in snapshots if snapshot["rows"]]
    cols = [snapshot["cols"] for snapshot in snapshots if snapshot["cols"]]

    result = {
        "analysis": "snapshots",
        "snapshot_directory": str(directory),
        **summary,
        "all_declared_matrices_square": all(row == col for row, col in zip(rows, cols)),
        "declared_matrix_row_count_min": min(rows) if rows else None,
        "declared_matrix_row_count_max": max(rows) if rows else None,
    }

    output_file_path = Path(output_path).resolve() if output_path is not None else None
    analysis_output_directory = (
        output_file_path.parent if output_file_path is not None else directory
    )

    if metadata_only:
        result["petsc_snapshot_analysis"] = {
            "available": False,
            "status": "metadata-only",
            "reason": "PETSc snapshot analysis was disabled.",
        }
    else:
        analysis_binary = resolve_petsc_snapshot_analysis_binary(
            petsc_snapshot_analysis_binary,
        )
        if analysis_binary is None:
            result["petsc_snapshot_analysis"] = {
                "available": False,
                "status": "not-found",
                "reason": (
                    f"{SNAPSHOT_ANALYSIS_BINARY_NAME} was not found. Build it with "
                    "`cmake --build build -j` or pass --petsc-analysis-binary."
                ),
            }
        else:
            snapshot_collection_path = create_snapshot_collection_from_directory(
                directory,
                analysis_output_directory / "snapshot_analysis_collection.csv",
                overwrite=True,
            )
            resolved_snapshot_collection_path = write_resolved_snapshot_collection(
                snapshot_collection_path,
                analysis_output_directory / "snapshot_analysis_collection.resolved.csv",
            )
            result["snapshot_collection_path"] = str(snapshot_collection_path.resolve())
            result["resolved_snapshot_collection_path"] = str(
                resolved_snapshot_collection_path.resolve()
            )
            result["petsc_snapshot_analysis"] = run_petsc_snapshot_analysis(
                petsc_snapshot_analysis_binary=analysis_binary,
                snapshot_collection_path=resolved_snapshot_collection_path,
                output_directory=analysis_output_directory,
                mpiexec=mpiexec,
                mpi_processes=mpi_processes,
                threads_per_rank=threads_per_rank,
                nullspace=nullspace,
                solve_index=solve_index,
                rhs_compatibility_tolerance=rhs_compatibility_tolerance,
                nullspace_residual_tolerance=nullspace_residual_tolerance,
            )

    if output_file_path is not None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return result
