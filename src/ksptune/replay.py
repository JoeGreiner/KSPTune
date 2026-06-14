from __future__ import annotations

import json
import math
import os
import queue
import selectors
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REQUIRED_REPLAY_RESULT_FIELDS = {
    "objective_time_sec_median",
    "total_wall_time_sec",
    "converged",
    "reason_code",
    "solve_time_sec_median",
    "solver_setup_time_sec",
    "iterations_total",
    "snapshots",
    "steps",
}

NUMERIC_REPLAY_RESULT_FIELDS = {
    "objective_time_sec_median",
    "total_wall_time_sec",
    "matrix_load_time_sec",
    "matrix_cache_hits",
    "matrix_cache_misses",
    "replay_cache_memory_mb",
    "replay_cache_memory_limit_mb",
    "replay_cache_evictions",
    "solver_setup_time_sec",
    "solver_setup_time_sec_actual",
    "solver_setup_time_sec_logical",
    "ksp_setup_cache_hits",
    "ksp_setup_cache_misses",
    "solve_time_sec_total",
    "solve_time_sec_mean",
    "solve_time_sec_median",
    "solve_time_sec_min",
    "solve_time_sec_max",
    "solve_time_sec_stddev",
    "solve_time_sec_range",
    "solve_mpi_message_count",
    "solve_mpi_message_bytes",
    "solve_mpi_message_bytes_mean",
    "solve_mpi_reduction_count",
    "iterations_total",
    "initial_true_residual_norm_mean",
    "initial_true_relative_residual_mean",
    "final_true_residual_norm_mean",
    "final_true_relative_residual_mean",
    "peak_memory_mb_max_per_rank",
    "peak_memory_mb_mean_per_rank",
    "peak_memory_mb_sum",
    "peak_memory_rank_count",
    "solve_count",
    "field_nullspace_index",
    "field_nullspace_block_size",
    "matrix_nullspace_attached_count",
    "transpose_nullspace_attached_count",
    "near_nullspace_attached_count",
    "rhs_nullspace_removed_count",
    "rhs_nullspace_removed_component_norm_max",
    "rhs_nullspace_removed_component_relative_norm_max",
}

PETSC_RETURN_CODE_HINTS = {
    59: "PETSc caught a signal, commonly a SIGSEGV",
}


def build_replay_command(
    *,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    replay_result_path: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    repeat: int = 1,
    warmup: int = 0,
    extra_replay_options: list[str] | None = None,
) -> list[str]:
    replay_command = [
        str(replay_binary),
        "-snapshot_collection",
        str(snapshot_collection_path),
        "-replay_json_out",
        str(replay_result_path),
        "-replay_repeat",
        str(repeat),
        "-replay_warmup",
        str(warmup),
    ]
    if extra_replay_options:
        replay_command.extend(extra_replay_options)
    replay_command.extend(petsc_options)

    if mpi_processes > 1:
        return [mpiexec, *(mpiexec_args or []), "-n", str(mpi_processes), *replay_command]
    if mpiexec_args:
        return [mpiexec, *mpiexec_args, *replay_command]
    return replay_command


def build_replay_server_command(
    *,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    extra_replay_options: list[str] | None = None,
    cache_memory_mb: float | None = None,
) -> list[str]:
    replay_command = [
        str(replay_binary),
        "-replay_server",
        "-snapshot_collection",
        str(snapshot_collection_path),
    ]
    if cache_memory_mb is not None:
        replay_command.extend(["-replay_cache_memory_mb", str(cache_memory_mb)])
    if extra_replay_options:
        replay_command.extend(extra_replay_options)
    if mpi_processes > 1:
        return [mpiexec, *(mpiexec_args or []), "-n", str(mpi_processes), *replay_command]
    if mpiexec_args:
        return [mpiexec, *mpiexec_args, *replay_command]
    return replay_command


def replay_environment(threads_per_rank: int = 1) -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads_per_rank)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    return environment


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_replay_result(
    replay_result: dict[str, Any],
    *,
    objective_name: str = "solve_time_sec_mean",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(replay_result, dict):
        return ["replay result must be a JSON object"]

    for field in sorted(REQUIRED_REPLAY_RESULT_FIELDS):
        if field not in replay_result:
            errors.append(f"missing replay metric: {field}")

    if objective_name not in replay_result:
        errors.append(f"missing objective metric: {objective_name}")

    for field in sorted(NUMERIC_REPLAY_RESULT_FIELDS | {objective_name}):
        if field in replay_result and not is_number(replay_result[field]):
            errors.append(f"replay metric must be numeric: {field}")

    if "converged" in replay_result and not isinstance(replay_result["converged"], bool):
        errors.append("replay metric must be boolean: converged")
    if "reason_code" in replay_result and not isinstance(replay_result["reason_code"], int):
        errors.append("replay metric must be integer: reason_code")
    if "steps" in replay_result and not isinstance(replay_result["steps"], list):
        errors.append("replay metric must be a list: steps")
    if "peak_memory_mb_per_rank" in replay_result and not isinstance(
        replay_result["peak_memory_mb_per_rank"],
        list,
    ):
        errors.append("replay metric must be a list: peak_memory_mb_per_rank")

    return errors


def replay_failure_reason(
    replay_record: dict[str, Any],
    *,
    objective_name: str = "solve_time_sec_mean",
) -> str | None:
    if replay_record.get("failure_reason"):
        return str(replay_record["failure_reason"])

    returncode = replay_record.get("returncode")
    if returncode != 0:
        hint = PETSC_RETURN_CODE_HINTS.get(returncode)
        if hint:
            return f"replay command returned {returncode} ({hint})"
        return f"replay command returned {returncode}"

    schema_errors = replay_record.get("schema_errors") or validate_replay_result(
        replay_record.get("replay_result", {}),
        objective_name=objective_name,
    )
    if schema_errors:
        return "invalid replay result: " + "; ".join(str(error) for error in schema_errors)

    replay_result = replay_record.get("replay_result", {})
    if replay_result.get("converged") is False:
        reason = replay_result.get("reason", replay_result.get("reason_code", "unknown"))
        return f"not converged: {reason}"

    return None


def replay_objective_value(
    replay_record: dict[str, Any],
    *,
    objective_name: str = "solve_time_sec_mean",
    bad_cost: float,
) -> float:
    if replay_failure_reason(replay_record, objective_name=objective_name):
        return bad_cost
    try:
        return float(replay_record["replay_result"][objective_name])
    except (KeyError, TypeError, ValueError):
        return bad_cost


def solve_step_times(replay_result: dict[str, Any]) -> list[float]:
    steps = replay_result.get("steps")
    if not isinstance(steps, list):
        return []

    solve_times: list[float] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        value = step.get("solve_time_sec")
        if is_number(value):
            solve_times.append(float(value))
    return solve_times


def expected_solve_count(replay_result: dict[str, Any]) -> int | None:
    if is_number(replay_result.get("snapshots")) and is_number(replay_result.get("repeat")):
        return int(replay_result["snapshots"]) * int(replay_result["repeat"])
    return None


def population_stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    values_mean = sum(values) / len(values)
    squared_delta_sum = sum((value - values_mean) ** 2 for value in values)
    return math.sqrt(squared_delta_sum / len(values))


def replay_metric_summary(replay_result: dict[str, Any]) -> dict[str, Any]:
    metric_names = [
        "total_wall_time_sec",
        "matrix_load_time_sec",
        "matrix_cache_hits",
        "matrix_cache_misses",
        "replay_cache_memory_mb",
        "replay_cache_memory_limit_mb",
        "replay_cache_evictions",
        "solver_setup_time_sec",
        "solver_setup_time_sec_actual",
        "solver_setup_time_sec_logical",
        "ksp_setup_cache_hits",
        "ksp_setup_cache_misses",
        "solve_time_sec_total",
        "solve_time_sec_mean",
        "solve_time_sec_median",
        "solve_time_sec_min",
        "solve_time_sec_max",
        "solve_time_sec_stddev",
        "solve_time_sec_range",
        "solve_mpi_message_count",
        "solve_mpi_message_bytes",
        "solve_mpi_message_bytes_mean",
        "solve_mpi_reduction_count",
        "iterations_total",
        "iterations_median",
        "peak_memory_mb_max_per_rank",
        "peak_memory_mb_mean_per_rank",
        "peak_memory_mb_sum",
        "peak_memory_rank_count",
        "initial_true_residual_norm_mean",
        "initial_true_relative_residual_mean",
        "final_true_residual_norm_mean",
        "final_true_relative_residual_mean",
        "converged",
        "reason",
        "reason_code",
        "snapshots",
        "repeat",
        "warmup",
        "solve_count",
        "nullspace",
        "nullspace_source",
        "nullspace_kind",
        "nullspace_actions",
        "field_nullspace_index",
        "field_nullspace_block_size",
        "matrix_nullspace_attached_count",
        "transpose_nullspace_attached_count",
        "near_nullspace_attached_count",
        "rhs_nullspace_removed_count",
        "rhs_nullspace_removed_component_norm_max",
        "rhs_nullspace_removed_component_relative_norm_max",
        "ksp_type",
        "pc_type",
    ]
    summary = {name: replay_result.get(name) for name in metric_names if name in replay_result}

    solve_times = solve_step_times(replay_result)
    if "solve_count" not in summary:
        if solve_times:
            summary["solve_count"] = len(solve_times)
        elif (expected_count := expected_solve_count(replay_result)) is not None:
            summary["solve_count"] = expected_count

    if "solve_time_sec_total" not in summary:
        if solve_times:
            summary["solve_time_sec_total"] = sum(solve_times)
        elif is_number(replay_result.get("solve_time_sec_mean")) and is_number(
            summary.get("solve_count")
        ):
            summary["solve_time_sec_total"] = (
                float(replay_result["solve_time_sec_mean"]) * int(summary["solve_count"])
            )

    if solve_times:
        summary.setdefault("solve_time_sec_min", min(solve_times))
        summary.setdefault("solve_time_sec_max", max(solve_times))
        summary.setdefault("solve_time_sec_stddev", population_stddev(solve_times))
        summary.setdefault("solve_time_sec_range", max(solve_times) - min(solve_times))

    memory_per_rank = replay_result.get("peak_memory_mb_per_rank")
    if isinstance(memory_per_rank, list):
        numeric_memory_values = [
            float(value) for value in memory_per_rank if is_number(value)
        ]
        if numeric_memory_values:
            summary.setdefault("peak_memory_rank_count", len(numeric_memory_values))
            summary.setdefault("peak_memory_mb_sum", sum(numeric_memory_values))
            summary.setdefault("peak_memory_mb_max_per_rank", max(numeric_memory_values))
            summary.setdefault(
                "peak_memory_mb_mean_per_rank",
                sum(numeric_memory_values) / len(numeric_memory_values),
            )
    elif is_number(summary.get("peak_memory_mb_sum")) and is_number(
        summary.get("peak_memory_rank_count")
    ):
        rank_count = int(summary["peak_memory_rank_count"])
        if rank_count > 0:
            summary.setdefault(
                "peak_memory_mb_mean_per_rank",
                float(summary["peak_memory_mb_sum"]) / rank_count,
            )

    return summary


def read_replay_result_file(result_path: Path) -> tuple[dict[str, Any], list[str]]:
    replay_result: dict[str, Any] = {}
    schema_errors: list[str] = []
    if result_path.exists():
        try:
            replay_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            schema_errors.append(f"invalid replay JSON: {exc}")
    else:
        schema_errors.append(f"missing replay JSON: {result_path}")

    if replay_result:
        schema_errors.extend(validate_replay_result(replay_result))
    return replay_result, schema_errors


def run_replay_for_solver_configuration(
    *,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    replay_result_path: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    repeat: int = 1,
    warmup: int = 0,
    timeout_sec: float | None = None,
    threads_per_rank: int = 1,
    extra_replay_options: list[str] | None = None,
) -> dict[str, Any]:
    result_path = Path(replay_result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_replay_command(
        replay_binary=replay_binary,
        snapshot_collection_path=snapshot_collection_path,
        replay_result_path=result_path,
        petsc_options=petsc_options,
        mpiexec=mpiexec,
        mpiexec_args=mpiexec_args,
        mpi_processes=mpi_processes,
        repeat=repeat,
        warmup=warmup,
        extra_replay_options=extra_replay_options,
    )
    start_time = time.perf_counter()
    failure_reason: str | None = None
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout_sec,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=replay_environment(threads_per_rank),
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        failure_reason = f"replay command timed out after {timeout_sec} seconds"
    subprocess_wall_time_sec = time.perf_counter() - start_time

    replay_result: dict[str, Any] = {}
    schema_errors: list[str] = []
    if result_path.exists():
        try:
            replay_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            schema_errors.append(f"invalid replay JSON: {exc}")
    else:
        schema_errors.append(f"missing replay JSON: {result_path}")

    if replay_result:
        schema_errors.extend(validate_replay_result(replay_result))

    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "subprocess_wall_time_sec": subprocess_wall_time_sec,
        "replay_result_path": str(result_path),
        "replay_result": replay_result,
        "schema_errors": schema_errors,
        "failure_reason": failure_reason,
    }


class ReplayServerProcess:
    def __init__(
        self,
        *,
        replay_binary: str | Path,
        snapshot_collection_path: str | Path,
        mpiexec: str = "mpiexec",
        mpiexec_args: list[str] | None = None,
        mpi_processes: int = 1,
        threads_per_rank: int = 1,
        extra_replay_options: list[str] | None = None,
        cache_memory_mb: float | None = None,
    ) -> None:
        self.command = build_replay_server_command(
            replay_binary=replay_binary,
            snapshot_collection_path=snapshot_collection_path,
            mpiexec=mpiexec,
            mpiexec_args=mpiexec_args,
            mpi_processes=mpi_processes,
            extra_replay_options=extra_replay_options,
            cache_memory_mb=cache_memory_mb,
        )
        self.threads_per_rank = threads_per_rank
        self.process: subprocess.Popen[str] | None = None
        self.stderr_file = None
        self.stderr_path: Path | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close(kill=True)
        stderr_file = tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            prefix="ksptune-replay-server-",
            suffix=".stderr",
            delete=False,
        )
        self.stderr_file = stderr_file
        self.stderr_path = Path(stderr_file.name)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            env=replay_environment(self.threads_per_rank),
        )

    def stderr_text(self) -> str:
        if self.stderr_file is not None:
            self.stderr_file.flush()
        if self.stderr_path is None or not self.stderr_path.exists():
            return ""
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")

    def close(self, *, kill: bool = False, timeout_sec: float = 2.0) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            if kill:
                process.kill()
                try:
                    process.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    pass
            else:
                try:
                    if process.stdin is not None:
                        process.stdin.write(
                            json.dumps(
                                {"id": f"shutdown-{uuid.uuid4().hex}", "command": "shutdown"}
                            )
                            + "\n"
                        )
                        process.stdin.flush()
                    process.wait(timeout=timeout_sec)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=timeout_sec)
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None

    def _read_response_line(self, timeout_sec: float | None) -> str | None:
        process = self.process
        if process is None or process.stdout is None:
            return None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout_sec)
            if not events:
                return None
            return process.stdout.readline()
        finally:
            selector.close()

    def run_solver_configuration(
        self,
        *,
        replay_result_path: str | Path,
        petsc_options: list[str],
        repeat: int = 1,
        warmup: int = 0,
        timeout_sec: float | None = None,
        reproduction_command: list[str],
    ) -> dict[str, Any]:
        with self.lock:
            result_path = Path(replay_result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            request_id = uuid.uuid4().hex
            request = {
                "id": request_id,
                "replay_json_out": str(result_path),
                "petsc_options": list(petsc_options),
                "repeat": repeat,
                "warmup": warmup,
            }

            start_time = time.perf_counter()
            failure_reason: str | None = None
            returncode: int | None = None
            response: dict[str, Any] = {}
            stdout = ""
            stderr = ""
            try:
                self.start()
                assert self.process is not None
                if self.process.stdin is None:
                    raise BrokenPipeError("replay server stdin is closed")
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                response_line = self._read_response_line(timeout_sec)
                if not response_line:
                    failure_reason = f"replay server timed out after {timeout_sec} seconds"
                    self.close(kill=True)
                else:
                    stdout = response_line
                    response = json.loads(response_line)
                    returncode = int(response.get("returncode", 1))
                    failure_reason = response.get("failure_reason")
            except (BrokenPipeError, OSError) as exc:
                failure_reason = f"replay server exited before completing request: {exc}"
                self.close(kill=True)
            except json.JSONDecodeError as exc:
                failure_reason = f"invalid replay server response: {exc}"
                self.close(kill=True)

            subprocess_wall_time_sec = time.perf_counter() - start_time
            if returncode is None and self.process is not None:
                returncode = self.process.poll()
            stderr = self.stderr_text()
            replay_result, schema_errors = read_replay_result_file(result_path)
            if failure_reason and not replay_result and "missing replay JSON" in "; ".join(schema_errors):
                schema_errors = []

            return {
                "command": reproduction_command,
                "replay_server_command": self.command,
                "replay_mode": "server",
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "subprocess_wall_time_sec": subprocess_wall_time_sec,
                "replay_result_path": str(result_path),
                "replay_result": replay_result,
                "schema_errors": schema_errors,
                "failure_reason": failure_reason,
                "replay_server_response": response,
            }


class ReplayWorkerPool:
    def __init__(
        self,
        *,
        replay_binary: str | Path,
        snapshot_collection_path: str | Path,
        workers: int,
        mpiexec: str = "mpiexec",
        mpiexec_args: list[str] | None = None,
        mpi_processes: int = 1,
        threads_per_rank: int = 1,
        extra_replay_options: list[str] | None = None,
        cache_memory_mb: float | None = None,
    ) -> None:
        self.workers: list[ReplayServerProcess] = [
            ReplayServerProcess(
                replay_binary=replay_binary,
                snapshot_collection_path=snapshot_collection_path,
                mpiexec=mpiexec,
                mpiexec_args=mpiexec_args,
                mpi_processes=mpi_processes,
                threads_per_rank=threads_per_rank,
                extra_replay_options=extra_replay_options,
                cache_memory_mb=cache_memory_mb,
            )
            for _ in range(max(1, workers))
        ]
        self.available: queue.Queue[ReplayServerProcess] = queue.Queue()
        for worker in self.workers:
            self.available.put(worker)

    def close(self) -> None:
        for worker in self.workers:
            worker.close()

    def run_solver_configuration(self, **kwargs: Any) -> dict[str, Any]:
        worker = self.available.get()
        try:
            return worker.run_solver_configuration(**kwargs)
        finally:
            self.available.put(worker)


def run_replay_server_for_solver_configuration(
    *,
    replay_worker_pool: ReplayWorkerPool,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    replay_result_path: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    repeat: int = 1,
    warmup: int = 0,
    timeout_sec: float | None = None,
    extra_replay_options: list[str] | None = None,
) -> dict[str, Any]:
    reproduction_command = build_replay_command(
        replay_binary=replay_binary,
        snapshot_collection_path=snapshot_collection_path,
        replay_result_path=replay_result_path,
        petsc_options=petsc_options,
        mpiexec=mpiexec,
        mpiexec_args=mpiexec_args,
        mpi_processes=mpi_processes,
        repeat=repeat,
        warmup=warmup,
        extra_replay_options=extra_replay_options,
    )
    return replay_worker_pool.run_solver_configuration(
        replay_result_path=replay_result_path,
        petsc_options=petsc_options,
        repeat=repeat,
        warmup=warmup,
        timeout_sec=timeout_sec,
        reproduction_command=reproduction_command,
    )
