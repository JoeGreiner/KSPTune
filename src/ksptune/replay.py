from __future__ import annotations

import atexit
import json
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .replay_results import OOM_RETURN_CODES, read_replay_result_file


@dataclass(frozen=True)
class ReplayProcessSpec:
    replay_binary: str
    snapshot_collection_path: str
    log_directory: str | None
    mpiexec: str
    mpiexec_args: tuple[str, ...]
    mpi_processes: int
    threads_per_rank: int
    extra_replay_options: tuple[str, ...]
    cache_memory_mb: float | None


_REPLAY_PROCESSES: dict[ReplayProcessSpec, "ReplayServerProcess"] = {}


def replay_process_for_worker(
    *,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    log_directory: str | Path | None = None,
    mpiexec: str,
    mpiexec_args: list[str],
    mpi_processes: int,
    threads_per_rank: int,
    extra_replay_options: list[str],
    cache_memory_mb: float | None,
) -> "ReplayServerProcess":
    spec = ReplayProcessSpec(
        replay_binary=str(replay_binary),
        snapshot_collection_path=str(snapshot_collection_path),
        log_directory=None if log_directory is None else str(log_directory),
        mpiexec=mpiexec,
        mpiexec_args=tuple(mpiexec_args),
        mpi_processes=mpi_processes,
        threads_per_rank=threads_per_rank,
        extra_replay_options=tuple(extra_replay_options),
        cache_memory_mb=cache_memory_mb,
    )
    process = _REPLAY_PROCESSES.get(spec)
    if process is None:
        process = ReplayServerProcess(
            replay_binary=spec.replay_binary,
            snapshot_collection_path=spec.snapshot_collection_path,
            log_directory=spec.log_directory,
            mpiexec=spec.mpiexec,
            mpiexec_args=list(spec.mpiexec_args),
            mpi_processes=spec.mpi_processes,
            threads_per_rank=spec.threads_per_rank,
            extra_replay_options=list(spec.extra_replay_options),
            cache_memory_mb=spec.cache_memory_mb,
        )
        _REPLAY_PROCESSES[spec] = process
    return process


def close_replay_processes() -> None:
    while _REPLAY_PROCESSES:
        _, process = _REPLAY_PROCESSES.popitem()
        process.close()


atexit.register(close_replay_processes)


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
    soft_timeout_sec: float | None = None,
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
    if soft_timeout_sec is not None:
        replay_command.extend(["-replay_soft_timeout_sec", str(soft_timeout_sec)])
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


def kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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
        log_directory: str | Path | None = None,
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
        self.log_directory = Path(log_directory) if log_directory is not None else None
        self.process: subprocess.Popen[str] | None = None
        self.stdout_buffer = b""
        self.stdout_file = None
        self.stdout_path: Path | None = None
        self.stderr_file = None
        self.stderr_path: Path | None = None
        self.lock = threading.Lock()

    def _open_log_file(self, suffix: str):
        if self.log_directory is None:
            return tempfile.NamedTemporaryFile(
                mode="w+",
                encoding="utf-8",
                prefix="ksptune-replay-server-",
                suffix=suffix,
                delete=False,
            )
        self.log_directory.mkdir(parents=True, exist_ok=True)
        path = self.log_directory / f"replay_server_{uuid.uuid4().hex}{suffix}"
        return path.open("w+", encoding="utf-8")

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close(kill=True)
        stdout_file = self._open_log_file(".stdout.log")
        stderr_file = self._open_log_file(".stderr.log")
        self.stdout_file = stdout_file
        self.stdout_path = Path(stdout_file.name)
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
            start_new_session=True,
        )

    def write_stdout_log(self, text: str) -> None:
        if self.stdout_file is None:
            return
        self.stdout_file.write(text)
        self.stdout_file.flush()

    def stdout_log_path(self) -> str | None:
        if self.log_directory is None:
            return None
        return None if self.stdout_path is None else str(self.stdout_path)

    def stderr_log_path(self) -> str | None:
        if self.log_directory is None:
            return None
        return None if self.stderr_path is None else str(self.stderr_path)

    def stderr_text(self) -> str:
        if self.stderr_file is not None:
            self.stderr_file.flush()
        if self.stderr_path is None or not self.stderr_path.exists():
            return ""
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")

    def close(self, *, kill: bool = False, hard_timeout_sec: float = 2.0) -> None:
        process = self.process
        self.process = None
        self.stdout_buffer = b""
        if process is not None and not kill and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(
                        json.dumps({"id": f"shutdown-{uuid.uuid4().hex}", "command": "shutdown"})
                        + "\n"
                    )
                    process.stdin.flush()
                process.wait(timeout=hard_timeout_sec)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                pass
        if process is not None:
            # The launcher may already have exited while its children are still running.
            kill_process_group(process)
            process.wait(timeout=hard_timeout_sec)
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()
        for file_handle in (self.stdout_file, self.stderr_file):
            if file_handle is not None:
                file_handle.close()
        if self.log_directory is None:
            for log_path in (self.stdout_path, self.stderr_path):
                if log_path is not None:
                    try:
                        log_path.unlink(missing_ok=True)
                    except OSError:
                        pass
        self.stdout_file = None
        self.stdout_path = None
        self.stderr_file = None
        self.stderr_path = None

    def _read_response_line(self, hard_timeout_sec: float | None) -> str | None:
        process = self.process
        if process is None or process.stdout is None:
            return None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = None if hard_timeout_sec is None else time.monotonic() + hard_timeout_sec
        try:
            # Read the pipe directly: TextIO.readline may buffer the JSON response
            # behind diagnostic lines while select sees an empty pipe.
            while b"\n" not in self.stdout_buffer:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if not selector.select(remaining):
                    return None
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    line, self.stdout_buffer = self.stdout_buffer, b""
                    return line.decode("utf-8", errors="replace")
                self.stdout_buffer += chunk
            line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
            return line.decode("utf-8", errors="replace") + "\n"
        finally:
            selector.close()

    def _wait_for_returncode_after_stdout_closed(self, hard_timeout_sec: float = 0.25) -> int | None:
        process = self.process
        if process is None:
            return None
        returncode = process.poll()
        if returncode is not None:
            return returncode
        try:
            process.wait(timeout=hard_timeout_sec)
        except subprocess.TimeoutExpired:
            return process.poll()
        return process.returncode

    def run_solver_configuration(
        self,
        *,
        replay_result_path: str | Path,
        petsc_options: list[str],
        repeat: int = 1,
        warmup: int = 0,
        hard_timeout_sec: float | None = None,
        replay_startup_timeout_sec: float | None = None,
        soft_timeout_sec: float | None = None,
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
            if soft_timeout_sec is not None:
                request["soft_timeout_sec"] = soft_timeout_sec

            start_time = time.perf_counter()
            failure_reason: str | None = None
            failure_kind: str | None = None
            returncode: int | None = None
            response: dict[str, Any] = {}
            stdout = ""
            stderr = ""
            stdout_path: str | None = None
            stderr_path: str | None = None
            effective_hard_timeout_sec = hard_timeout_sec
            used_startup_timeout = False
            try:
                server_was_running = self.process is not None and self.process.poll() is None
                self.start()
                stdout_path = self.stdout_log_path()
                stderr_path = self.stderr_log_path()
                if not server_was_running and replay_startup_timeout_sec is not None:
                    effective_hard_timeout_sec = replay_startup_timeout_sec
                    used_startup_timeout = True
                assert self.process is not None
                if self.process.stdin is None:
                    raise BrokenPipeError("replay server stdin is closed")
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                deadline = (
                    time.perf_counter() + effective_hard_timeout_sec
                    if effective_hard_timeout_sec is not None
                    else None
                )
                while True:
                    remaining_timeout = (
                        None if deadline is None else max(0.0, deadline - time.perf_counter())
                    )
                    response_line = self._read_response_line(remaining_timeout)
                    if response_line is None:
                        returncode = self._wait_for_returncode_after_stdout_closed()
                        stderr = self.stderr_text()
                        if returncode is None:
                            failure_kind = "hard_timeout"
                            timeout_label = "startup " if used_startup_timeout else ""
                            failure_reason = (
                                f"replay server {timeout_label}timed out after "
                                f"{effective_hard_timeout_sec} seconds"
                            )
                        else:
                            failure_reason = (
                                f"replay server exited before completing request "
                                f"(returncode={returncode})"
                            )
                            if stderr.strip():
                                failure_reason += f": {stderr.strip()}"
                        self.close(kill=True)
                        break

                    if response_line == "":
                        returncode = self._wait_for_returncode_after_stdout_closed()
                        stderr = self.stderr_text()
                        if returncode is None:
                            failure_reason = "replay server closed stdout before completing request"
                        else:
                            failure_reason = (
                                f"replay server exited before completing request "
                                f"(returncode={returncode})"
                            )
                        if stderr.strip():
                            failure_reason += f": {stderr.strip()}"
                        self.close(kill=True)
                        break

                    self.write_stdout_log(response_line)
                    stdout += response_line
                    try:
                        parsed_response = json.loads(response_line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed_response, dict):
                        continue
                    if parsed_response.get("id") != request_id:
                        if not parsed_response.get("id") and parsed_response.get("failure_reason"):
                            failure_reason = str(parsed_response["failure_reason"])
                            returncode = int(parsed_response.get("returncode", 1))
                            break
                        continue

                    response = parsed_response
                    returncode = int(response.get("returncode", 1))
                    failure_reason = response.get("failure_reason")
                    break
            except (BrokenPipeError, OSError) as exc:
                if self.process is not None:
                    returncode = self.process.poll()
                stdout_path = stdout_path or self.stdout_log_path()
                stderr_path = stderr_path or self.stderr_log_path()
                stderr = self.stderr_text()
                if returncode is None:
                    failure_reason = f"replay server exited before completing request: {exc}"
                else:
                    failure_reason = (
                        f"replay server exited before completing request (returncode={returncode})"
                    )
                    if stderr.strip():
                        failure_reason += f": {stderr.strip()}"
                self.close(kill=True)

            subprocess_wall_time_sec = time.perf_counter() - start_time
            if returncode is None and self.process is not None:
                returncode = self.process.poll()
            if not stderr:
                stderr = self.stderr_text()
            replay_result, schema_errors = read_replay_result_file(result_path)
            if failure_reason and not result_path.exists():
                schema_errors = []
            if not failure_kind and (failure_reason or returncode != 0):
                failure_kind = "oom" if returncode in OOM_RETURN_CODES else "process_error"

            return {
                "command": reproduction_command,
                "replay_server_command": self.command,
                "replay_mode": "server",
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "subprocess_wall_time_sec": subprocess_wall_time_sec,
                "hard_timeout_sec": hard_timeout_sec,
                "replay_startup_timeout_sec": replay_startup_timeout_sec,
                "effective_hard_timeout_sec": effective_hard_timeout_sec,
                "used_startup_timeout": used_startup_timeout,
                "soft_timeout_sec": soft_timeout_sec,
                "replay_result_path": str(result_path),
                "replay_result": replay_result,
                "schema_errors": schema_errors,
                "failure_reason": failure_reason,
                "failure_kind": failure_kind,
                "replay_server_response": response,
            }


def run_replay_server_for_solver_configuration(
    *,
    replay_server: ReplayServerProcess | None = None,
    replay_binary: str | Path,
    snapshot_collection_path: str | Path,
    replay_result_path: str | Path,
    petsc_options: list[str],
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    mpi_processes: int = 1,
    repeat: int = 1,
    warmup: int = 0,
    hard_timeout_sec: float | None = None,
    replay_startup_timeout_sec: float | None = None,
    soft_timeout_sec: float | None = None,
    threads_per_rank: int = 1,
    extra_replay_options: list[str] | None = None,
    log_directory: str | Path | None = None,
    cache_memory_mb: float | None = None,
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
        soft_timeout_sec=soft_timeout_sec,
        extra_replay_options=extra_replay_options,
    )
    replay_server = replay_server or replay_process_for_worker(
        replay_binary=replay_binary,
        snapshot_collection_path=snapshot_collection_path,
        log_directory=log_directory,
        mpiexec=mpiexec,
        mpiexec_args=list(mpiexec_args or []),
        mpi_processes=mpi_processes,
        threads_per_rank=threads_per_rank,
        extra_replay_options=list(extra_replay_options or []),
        cache_memory_mb=cache_memory_mb,
    )
    return replay_server.run_solver_configuration(
        replay_result_path=replay_result_path,
        petsc_options=petsc_options,
        repeat=repeat,
        warmup=warmup,
        hard_timeout_sec=hard_timeout_sec,
        replay_startup_timeout_sec=replay_startup_timeout_sec,
        soft_timeout_sec=soft_timeout_sec,
        reproduction_command=reproduction_command,
    )
