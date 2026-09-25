from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import uuid
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import yaml
from smac.callback import Callback

from .file_io import write_text_atomic
from .trial_records import (
    BAD_COST,
    add_trial_metrics,
    append_json_line,
    best_record_from_trial_records,
    is_successful_trial,
    load_trial_records,
    max_trial_number,
    tuning_output_paths,
    write_tuning_summary_outputs,
    write_yaml_file,
)

from .parameter_search_spaces import (
    get_parameter_search_space_name,
    load_parameter_search_space,
    write_configspace_json,
)
from .nullspaces import parse_nullspace, parse_nullspace_actions
from .petsc_options import (
    petsc_options_to_text,
    render_petsc_options_from_solver_configuration,
)
from .petsc_help import query_petsc_options_help
from .replay_results import (
    DEFAULT_MAX_TRUE_RELATIVE_RESIDUAL,
    DEFAULT_MAX_TRUE_RESIDUAL_NORM,
    evaluate_replay,
    replay_metric_summary,
)
from .replay import (
    close_replay_processes,
    run_replay_server_for_solver_configuration,
)
from .snapshot_collections import (
    load_snapshots_from_directory,
    load_snapshot_collection,
    validate_snapshot_collection,
    summarize_snapshots,
    write_snapshot_collection,
)
from .solver_configurations import (
    default_solver_configuration_from_parameter_search_space,
    parameter_is_active,
    plain_python_value,
    solver_configuration_from_configspace_configuration,
)

RUN_UNTIL_STOPPED_TRIAL_LIMIT = 2_147_483_647
REPLAY_BINARY_NAME = "ksptune-petsc-replay"
REPLAY_BINARY_ENV_VAR = "KSPTUNE_REPLAY_BINARY"
FAST_FAIL_KSP_MAX_IT = 1
HYPRE_HIERARCHY_OPTION = "-pc_hypre_boomeramg_view_hierarchy"
HYPRE_HIERARCHY_REPLAY_OPTION = "-replay_hypre_hierarchy_diagnostics"
HYPRE_HIERARCHY_DIAGNOSTICS_MODES = {"auto", "on", "off"}


def petsc_help_result_has_option(result: dict[str, Any], option_name: str) -> bool:
    for field_name in ("filtered_lines", "option_lines"):
        for line in result.get(field_name) or []:
            if str(line).lstrip().startswith(option_name):
                return True
    return option_name in str(result.get("raw_output") or "")


def resolve_hypre_hierarchy_diagnostics(
    *,
    mode: str | None,
    replay_binary: str | Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    resolved_mode = "auto" if mode is None else str(mode)
    if resolved_mode not in HYPRE_HIERARCHY_DIAGNOSTICS_MODES:
        choices = ", ".join(sorted(HYPRE_HIERARCHY_DIAGNOSTICS_MODES))
        raise ValueError(f"hypre hierarchy diagnostics must be one of: {choices}.")

    diagnostics = {
        "mode": resolved_mode,
        "option": HYPRE_HIERARCHY_OPTION,
        "available": False,
        "enabled": False,
    }
    if resolved_mode == "off":
        diagnostics["reason"] = "disabled by user"
        return diagnostics
    if dry_run or replay_binary is None:
        diagnostics["reason"] = "not checked during dry-run"
        return diagnostics

    help_result = query_petsc_options_help(
        replay_binary=replay_binary,
        petsc_options=["-pc_type", "hypre", "-pc_hypre_type", "boomeramg"],
        filters=[HYPRE_HIERARCHY_OPTION],
        raw=False,
        timeout_sec=10.0,
    )
    available = int(help_result.get("returncode", 1)) == 0 and petsc_help_result_has_option(
        help_result,
        HYPRE_HIERARCHY_OPTION,
    )
    diagnostics.update(
        {
            "available": available,
            "enabled": available,
            "help_returncode": help_result.get("returncode"),
            "help_command": help_result.get("command"),
        }
    )
    if help_result.get("error"):
        diagnostics["help_error"] = help_result["error"]
    if available:
        diagnostics["reason"] = "available in linked PETSc"
    else:
        diagnostics["reason"] = "option not reported by linked PETSc"

    if resolved_mode == "on" and not available:
        raise ValueError(
            f"{HYPRE_HIERARCHY_OPTION} is not available in the linked PETSc replay binary."
        )
    return diagnostics


def petsc_options_with_override(
    petsc_options: list[str],
    option_name: str,
    option_value: Any,
) -> list[str]:
    overridden_options: list[str] = []
    skip_next = False
    for index, option in enumerate(petsc_options):
        if skip_next:
            skip_next = False
            continue
        option_text = str(option)
        if option_text == option_name:
            next_index = index + 1
            if next_index < len(petsc_options) and not str(petsc_options[next_index]).startswith(
                "-"
            ):
                skip_next = True
            continue
        if option_text.startswith(f"{option_name}="):
            continue
        overridden_options.append(option_text)
    overridden_options.extend([option_name, str(option_value)])
    return overridden_options


@dataclass(kw_only=True)
class ReplayServerTargetFunction:
    settings: TuningSettings
    parameter_search_space: Any
    snapshot_collection_path: Path
    extra_replay_options: list[str]

    def __call__(self, configuration: Any, seed: int = 0) -> tuple[float, dict[str, Any]]:  # noqa: ARG002
        configuration_tag = smac_configuration_tag(configuration)
        solver_configuration = solver_configuration_from_configspace_configuration(configuration)
        petsc_options = render_petsc_options_from_solver_configuration(
            self.parameter_search_space,
            solver_configuration,
        )
        if self.settings.fast_fail:
            petsc_options = petsc_options_with_override(
                petsc_options,
                "-ksp_max_it",
                FAST_FAIL_KSP_MAX_IT,
            )
        replay_record = run_replay_server_for_solver_configuration(
            log_directory=self.settings.output_directory / "replay_logs",
            cache_memory_mb=self.settings.replay_cache_memory_mb,
            replay_binary=self.settings.replay_binary,
            snapshot_collection_path=self.snapshot_collection_path,
            replay_result_path=(
                self.settings.output_directory
                / "replay_results"
                / f"candidate_{configuration_tag}_{uuid.uuid4().hex}.json"
            ),
            petsc_options=petsc_options,
            mpiexec=self.settings.mpiexec,
            mpiexec_args=self.settings.mpiexec_args,
            mpi_processes=self.settings.mpi_processes,
            repeat=self.settings.repeat,
            warmup=self.settings.warmup,
            hard_timeout_sec=self.settings.hard_timeout_sec,
            replay_startup_timeout_sec=self.settings.replay_startup_timeout_sec,
            soft_timeout_sec=self.settings.soft_timeout_sec,
            threads_per_rank=self.settings.threads_per_rank,
            extra_replay_options=self.extra_replay_options,
        )
        replay_result = replay_record.get("replay_result", {})
        replay_command = replay_record.get("command") or []
        evaluation = evaluate_replay(
            replay_record,
            objective_name=self.settings.objective_name,
            bad_cost=BAD_COST,
            max_true_relative_residual=self.settings.max_true_relative_residual,
            max_true_residual_norm=self.settings.max_true_residual_norm,
        )
        metrics = replay_metric_summary(replay_result) if evaluation.status in {
            "success", "soft_timeout", "nonconverged", "residual_rejected",
        } else {}
        record = {
            "smac_configuration_tag": configuration_tag,
            "solver_configuration": solver_configuration,
            "petsc_options": petsc_options,
            "petsc_options_text": petsc_options_to_text(petsc_options),
            "replay_command": replay_command,
            "replay_command_text": shlex.join(str(part) for part in replay_command),
            "replay_server_command": replay_record.get("replay_server_command"),
            "replay_server_command_text": shlex.join(
                str(part) for part in replay_record.get("replay_server_command", [])
            ),
            "replay_mode": replay_record.get("replay_mode", "server"),
            "stdout_path": replay_record.get("stdout_path"),
            "stderr_path": replay_record.get("stderr_path"),
            "objective_name": self.settings.objective_name,
            "objective_value": evaluation.objective_value,
            "status": evaluation.status,
            "replay_result_path": replay_record["replay_result_path"],
            "returncode": replay_record["returncode"],
            "failure_reason": evaluation.failure_reason,
            "max_true_relative_residual": self.settings.max_true_relative_residual,
            "max_true_residual_norm": self.settings.max_true_residual_norm,
            "fast_fail": self.settings.fast_fail,
            "fast_fail_ksp_max_it": FAST_FAIL_KSP_MAX_IT if self.settings.fast_fail else None,
            "schema_errors": replay_record.get("schema_errors", []),
            "subprocess_wall_time_sec": replay_record.get("subprocess_wall_time_sec"),
            "soft_timeout_sec": self.settings.soft_timeout_sec,
            "hard_timeout_sec": self.settings.hard_timeout_sec,
            "replay_startup_timeout_sec": replay_record.get("replay_startup_timeout_sec"),
            "effective_hard_timeout_sec": replay_record.get("effective_hard_timeout_sec"),
            "used_startup_timeout": replay_record.get("used_startup_timeout"),
            **metrics,
        }
        return evaluation.objective_value, {"ksptune_trial_record": add_trial_metrics(record)}


def smac_configuration_tag(configuration: Any, chars: int = 6) -> str:
    return hashlib.sha1(str(configuration).encode("utf-8")).hexdigest()[:chars]


def emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def close_smac_runner(smac: Any) -> None:
    runner = getattr(smac, "_runner", None)
    close = getattr(runner, "close", None)
    if callable(close):
        close()


def close_smac_replay_processes(smac: Any) -> None:
    runner = getattr(smac, "_runner", None)
    client = getattr(runner, "_client", None)
    run_on_workers = getattr(client, "run", None)
    if not callable(run_on_workers):
        return

    try:
        run_on_workers(close_replay_processes)
    except Exception as exc:  # pragma: no cover - defensive cleanup path
        warnings.warn(
            f"Could not close replay processes on Dask workers: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def count_tunable_hyperparameters(parameter_search_space: Any) -> int:
    from ConfigSpace import Constant

    return sum(
        1
        for hyperparameter in parameter_search_space.values()
        if not isinstance(hyperparameter, Constant)
    )


def resolve_sobol_initial_design_configuration_count(
    *,
    tunable_parameter_count: int,
    workers: int,
    effective_trials: int,
    default_solver_configuration_count: int,
    additional_solver_configuration_count: int,
) -> int:
    target_initial_design_count = min(40, max(8, tunable_parameter_count, 4 * workers))
    reserved_trials = default_solver_configuration_count + additional_solver_configuration_count
    target_sobol_count = max(0, target_initial_design_count - reserved_trials)
    available_trials = max(0, effective_trials - reserved_trials)
    return min(target_sobol_count, available_trials)


def stable_unique_configurations(configurations: list[Any]) -> list[Any]:
    unique_configurations: list[Any] = []
    for configuration in configurations:
        if configuration not in unique_configurations:
            unique_configurations.append(configuration)
    return unique_configurations


class KSPTuneOrderedInitialDesign:
    def __init__(
        self,
        *,
        base_initial_design: Any,
        leading_configurations: list[Any],
        include_default_configuration: bool,
        configuration_space: Any,
        include_base_configurations: bool = True,
    ) -> None:
        self.base_initial_design = base_initial_design
        self.leading_configurations = list(leading_configurations)
        self.include_default_configuration = include_default_configuration
        self.include_base_configurations = include_base_configurations
        self.configuration_space = configuration_space

    @property
    def meta(self) -> dict[str, Any]:
        base_meta = getattr(self.base_initial_design, "meta", {})
        return {
            "name": self.__class__.__name__,
            "base_initial_design": base_meta,
            "leading_configuration_count": len(self.leading_configurations),
            "include_default_configuration": self.include_default_configuration,
            "include_base_configurations": self.include_base_configurations,
        }

    def select_configurations(self) -> list[Any]:
        base_configurations = (
            list(self.base_initial_design.select_configurations())
            if self.include_base_configurations
            else []
        )
        ordered_configurations = list(self.leading_configurations)

        if self.include_default_configuration:
            default_configuration = self.configuration_space.get_default_configuration()
            matching_default_configuration = next(
                (
                    configuration
                    for configuration in base_configurations
                    if configuration == default_configuration
                ),
                default_configuration,
            )
            if getattr(matching_default_configuration, "origin", None) is None:
                matching_default_configuration.origin = "Initial Design: Default configuration"
            ordered_configurations.append(matching_default_configuration)
            base_configurations = [
                configuration
                for configuration in base_configurations
                if configuration != matching_default_configuration
            ]

        ordered_configurations.extend(base_configurations)
        return stable_unique_configurations(ordered_configurations)


def initial_solver_configuration_records_from_parameter_search_space(
    parameter_search_space: Any,
) -> list[dict[str, Any]]:
    from ConfigSpace import Configuration

    raw_records = list(getattr(parameter_search_space, "_ksptuneinitial_solver_configurations", []))
    records: list[dict[str, Any]] = []
    default_solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, dict):
            raise ValueError("Initial solver configurations must be dictionaries.")
        if "solver_configuration" in raw_record:
            label = str(raw_record.get("label") or f"initial_solver_configuration_{index}")
            solver_configuration_overrides = raw_record["solver_configuration"]
        else:
            label = f"initial_solver_configuration_{index}"
            solver_configuration_overrides = raw_record
        if not isinstance(solver_configuration_overrides, dict):
            raise ValueError(f"Initial solver configuration {label} must be a dictionary.")

        solver_configuration = dict(default_solver_configuration)
        solver_configuration.update(solver_configuration_overrides)
        solver_configuration = {
            name: value
            for name, value in solver_configuration.items()
            if name not in parameter_search_space
            or parameter_is_active(parameter_search_space, name, solver_configuration)
        }
        for name, hyperparameter in parameter_search_space.items():
            if name in solver_configuration:
                continue
            if parameter_is_active(parameter_search_space, name, solver_configuration):
                solver_configuration[name] = plain_python_value(hyperparameter.default_value)
        try:
            configuration = Configuration(parameter_search_space, values=solver_configuration)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Initial solver configuration {label} is not legal: {exc}") from exc
        configuration.origin = f"KSPTune initial solver configuration: {label}"
        records.append(
            {
                "label": label,
                "solver_configuration": solver_configuration_from_configspace_configuration(
                    configuration
                ),
                "configuration": configuration,
            }
        )
    return records


def optimize_smac_with_known_warning_filter(smac: Any) -> Any:
    # SMAC local search can warn on empty conditional-neighbor sets. Filter only that known pair.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice.",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
        )
        return smac.optimize()


def output_directory_has_run_data(output_directory: str | Path) -> bool:
    output_path = Path(output_directory)
    trials_path = output_path / "solver_configuration_trials.jsonl"
    return (
        (output_path / "tuning_run.yaml").exists()
        or ((output_path / "smac").exists() and any((output_path / "smac").iterdir()))
        or (trials_path.exists() and trials_path.read_text(encoding="utf-8").strip() != "")
    )


def clear_restart_outputs(output_directory: str | Path) -> None:
    output_path = Path(output_directory)
    smac_path = output_path / "smac"
    if smac_path.exists():
        shutil.rmtree(smac_path)
    for name in ("parameter_search_space.yaml", "configspace.yaml"):
        (output_path / name).unlink(missing_ok=True)


def remove_best_solver_configuration_outputs(output_directory: str | Path) -> None:
    output_path = Path(output_directory)
    for filename in ("best_solver_configuration.yaml", "best_petsc_options.txt"):
        (output_path / filename).unlink(missing_ok=True)


def local_replay_binary_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / "build/cpp/petsc_replay" / REPLAY_BINARY_NAME,
        project_root / "build" / REPLAY_BINARY_NAME,
        project_root / "cmake-build-release/cpp/petsc_replay" / REPLAY_BINARY_NAME,
        project_root / "cmake-build-debug/cpp/petsc_replay" / REPLAY_BINARY_NAME,
        project_root / "cmake-build-relwithdebinfo/cpp/petsc_replay" / REPLAY_BINARY_NAME,
    ]


def is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def replay_binary_from_environment() -> Path | None:
    replay_binary_from_env = os.environ.get(REPLAY_BINARY_ENV_VAR)
    if not replay_binary_from_env:
        return None

    replay_binary_path = Path(replay_binary_from_env).expanduser()
    if not replay_binary_path.exists():
        raise ValueError(
            f"{REPLAY_BINARY_ENV_VAR} points to a missing replay binary: {replay_binary_path}"
        )
    if not is_executable_file(replay_binary_path):
        raise ValueError(
            f"{REPLAY_BINARY_ENV_VAR} points to a non-executable replay binary: "
            f"{replay_binary_path}"
        )
    return replay_binary_path.resolve()


def validate_replay_binary_path(replay_binary: str | Path, *, source: str) -> Path:
    replay_binary_path = Path(replay_binary).expanduser()
    if not replay_binary_path.exists():
        raise ValueError(f"{source} points to a missing replay binary: {replay_binary_path}")
    if not is_executable_file(replay_binary_path):
        raise ValueError(f"{source} points to a non-executable replay binary: {replay_binary_path}")
    return replay_binary_path.resolve()


def find_default_replay_binary() -> Path | None:
    replay_binary_path = replay_binary_from_environment()
    if replay_binary_path is not None:
        return replay_binary_path

    project_root = Path(__file__).resolve().parents[2]
    for candidate in local_replay_binary_candidates(project_root):
        if is_executable_file(candidate):
            return candidate.resolve()

    path_replay_binary = shutil.which(REPLAY_BINARY_NAME)
    if path_replay_binary:
        return Path(path_replay_binary).resolve()
    return None


def resolve_replay_binary(
    replay_binary: str | Path | None,
    *,
    dry_run: bool,
) -> tuple[Path | None, bool]:
    if replay_binary is not None:
        return validate_replay_binary_path(replay_binary, source="--replay-binary"), False
    if dry_run:
        return None, False

    default_replay_binary = find_default_replay_binary()
    if default_replay_binary is None:
        raise ValueError(
            "--replay-binary was not provided and KSPTune could not find "
            f"{REPLAY_BINARY_NAME}. Install it on PATH, set {REPLAY_BINARY_ENV_VAR}, "
            "build it with `cmake --build build -j`, or pass --replay-binary explicitly."
        )
    return default_replay_binary, True


def find_resume_smac_state_directory(output_directory: str | Path, seed: int) -> Path:
    smac_path = Path(output_directory) / "smac"
    if not smac_path.exists():
        raise ValueError(f"No SMAC state found in {smac_path}.")
    candidates: list[Path] = []
    for scenario_path in smac_path.glob(f"*/{seed}/scenario.json"):
        state_directory = scenario_path.parent
        if (state_directory / "runhistory.json").exists() and (
            state_directory / "intensifier.json"
        ).exists():
            candidates.append(state_directory)
    if not candidates:
        raise ValueError(f"No complete SMAC state found in {smac_path} for seed {seed}.")
    if len(candidates) > 1:
        candidate_text = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple SMAC states found for seed {seed}: {candidate_text}")
    return candidates[0]


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def write_json_mapping(path: str | Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, indent=4, allow_nan=True) + "\n")


def write_solver_configuration_outputs(
    *,
    output_directory: str | Path,
    parameter_search_space: Any,
    solver_configuration: dict[str, Any],
) -> None:
    directory = Path(output_directory)
    petsc_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    write_yaml_file(directory / "best_solver_configuration.yaml", solver_configuration)
    write_text_atomic(
        directory / "best_petsc_options.txt",
        petsc_options_to_text(petsc_options) + "\n",
    )


@dataclass(frozen=True, kw_only=True)
class TuningSettings:
    snapshot_directory: str | Path
    output_directory: str | Path
    replay_binary: str | Path | None = None
    mpiexec: str = "mpiexec"
    mpiexec_args: list[str] = field(default_factory=list)
    mpi_processes: int = 1
    threads_per_rank: int = 1
    workers: int = 1
    trials: int | None = None
    repeat: int = 1
    warmup: int = 0
    soft_timeout_sec: float | None = None
    hard_timeout_sec: float | None = None
    replay_startup_timeout_sec: float | None = None
    objective_name: str = "solve_time_sec_mean"
    seed: int = 1
    dry_run: bool = False
    run_until_stopped: bool = False
    nullspace: str | None = "from-metadata"
    nullspace_actions: str | None = None
    deduplicate_matrices: bool = True
    reuse_ksp_setup: bool = True
    use_initial_guess: bool = True
    replay_cache_memory_mb: float | None = None
    hypre_hierarchy_diagnostics: str | None = "auto"
    max_true_relative_residual: float | None = DEFAULT_MAX_TRUE_RELATIVE_RESIDUAL
    max_true_residual_norm: float | None = DEFAULT_MAX_TRUE_RESIDUAL_NORM
    fast_fail: bool = False
    force_restart: bool = False

    def __post_init__(self) -> None:
        if self.trials is None:
            object.__setattr__(self, "run_until_stopped", True)
        if self.hard_timeout_sec is None and self.soft_timeout_sec is not None:
            object.__setattr__(self, "hard_timeout_sec", 1.5 * self.soft_timeout_sec)


def settings_to_dict(settings: TuningSettings) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(settings).items()
    }


@dataclass(kw_only=True)
class PreparedRun:
    settings: TuningSettings
    parameter_search_space: Any
    metadata: dict[str, Any]
    snapshot_collection_path: Path
    replay_snapshot_collection_path: Path
    replay_options: list[str]
    initial_configurations: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    existing_records: list[dict[str, Any]] = field(default_factory=list)
    smac_state: Path | None = None

    @property
    def trials_path(self) -> Path:
        return self.settings.output_directory / "solver_configuration_trials.jsonl"

    @property
    def completed_trials(self) -> int:
        return max_trial_number(self.existing_records)


def prepare_tuning(
    settings: TuningSettings,
    parameter_search_space: Any,
    *,
    previous_run: dict[str, Any] | None = None,
    existing_records: list[dict[str, Any]] | None = None,
    smac_state: Path | None = None,
) -> PreparedRun:
    resume = previous_run is not None
    if settings.workers < 1:
        raise ValueError("workers must be at least 1.")
    if resume and settings.force_restart:
        raise ValueError("resume and force_restart cannot be used together.")
    if not settings.run_until_stopped and (settings.trials is None or settings.trials < 1):
        raise ValueError("trials must be at least 1 unless run_until_stopped is true.")
    output_path = Path(settings.output_directory).resolve()
    if not resume and output_directory_has_run_data(output_path) and not settings.force_restart:
        raise ValueError(
            f"{output_path} already contains a KSPTune run. "
            "Use `ksptune resume RUN_DIR` to continue it or pass "
            "`--force-restart` to overwrite it."
        )
    replay_binary, auto_detected = resolve_replay_binary(
        settings.replay_binary, dry_run=settings.dry_run
    )
    settings = replace(
        settings,
        output_directory=output_path,
        snapshot_directory=Path(settings.snapshot_directory).resolve(),
        replay_binary=replay_binary,
        mpiexec_args=list(settings.mpiexec_args),
    )
    effective_trials = (
        RUN_UNTIL_STOPPED_TRIAL_LIMIT if settings.run_until_stopped else settings.trials
    )
    nullspace = parse_nullspace(settings.nullspace)
    actions = parse_nullspace_actions(settings.nullspace_actions, nullspace=settings.nullspace)
    diagnostics = resolve_hypre_hierarchy_diagnostics(
        mode=settings.hypre_hierarchy_diagnostics,
        replay_binary=replay_binary,
        dry_run=settings.dry_run,
    )
    replay_options = [*nullspace["replay_options"], *actions["replay_options"]]
    if not settings.reuse_ksp_setup:
        replay_options.extend(["-replay_reuse_ksp_setup", "false"])
    if not settings.use_initial_guess:
        replay_options.extend(["-replay_use_initial_guess", "false"])
    if diagnostics["enabled"]:
        replay_options.extend([HYPRE_HIERARCHY_REPLAY_OPTION, "true"])

    if resume:
        collection = Path(previous_run["snapshot_collection_path"]).resolve()
        replay_collection = Path(previous_run["replay_snapshot_collection_path"]).resolve()
        for path in (collection, replay_collection):
            errors = validate_snapshot_collection(path)
            if errors:
                raise ValueError("Cannot resume: " + "\n".join(errors))
        snapshots = load_snapshot_collection(collection)
    else:
        collection = output_path / "snapshot_collection.csv"
        replay_collection = output_path / "snapshot_collection.resolved.csv"
        snapshots = load_snapshots_from_directory(settings.snapshot_directory)

    initial_configurations = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    tunable_count = count_tunable_hyperparameters(parameter_search_space)
    if resume:
        use_default = previous_run["use_default_solver_configuration"]
        sobol_count = previous_run["sobol_initial_design_configurations"]
    else:
        initial_configurations = initial_configurations[:effective_trials]
        use_default = effective_trials > len(initial_configurations)
        sobol_count = resolve_sobol_initial_design_configuration_count(
            tunable_parameter_count=tunable_count,
            workers=settings.workers,
            effective_trials=effective_trials,
            default_solver_configuration_count=int(use_default),
            additional_solver_configuration_count=len(initial_configurations),
        )
    labels = [record["label"] for record in initial_configurations]
    startup_labels = labels[: settings.workers]
    metadata = dict(
        replay_binary_auto_detected=auto_detected,
        snapshot_collection_path=str(collection),
        replay_snapshot_collection_path=str(replay_collection),
        parameter_search_space=get_parameter_search_space_name(parameter_search_space),
        parameter_search_space_source=getattr(parameter_search_space, "_ksptune_source", None),
        snapshot_summary=summarize_snapshots(snapshots),
        effective_trials=effective_trials,
        nullspace_configuration=nullspace,
        nullspace_action_configuration=actions,
        hypre_hierarchy_diagnostics=diagnostics,
        fast_fail_ksp_max_it=FAST_FAIL_KSP_MAX_IT if settings.fast_fail else None,
        resume=resume,
        resumed_completed_trials=max_trial_number(existing_records or []),
        failed_replay_objective_value=BAD_COST,
        smac_deterministic=True,
        initial_design_order="seeded-first",
        use_default_solver_configuration=use_default,
        tunable_parameter_count=tunable_count,
        sobol_initial_design_configurations=sobol_count,
        startup_initial_solver_configuration_count=len(startup_labels),
        startup_initial_solver_configurations=startup_labels,
        additionalinitial_solver_configuration_count=len(labels),
        additionalinitial_solver_configurations=labels,
        initial_solver_configurations=[
            {"label": record["label"], "solver_configuration": record["solver_configuration"]}
            for record in initial_configurations
        ],
    )
    if resume:
        for key in (
            "parameter_search_space",
            "parameter_search_space_source",
            "replay_binary_auto_detected",
        ):
            metadata[key] = previous_run[key]
    return PreparedRun(
        settings=settings,
        parameter_search_space=parameter_search_space,
        metadata=metadata,
        snapshot_collection_path=collection,
        replay_snapshot_collection_path=replay_collection,
        replay_options=replay_options,
        initial_configurations=initial_configurations,
        snapshots=snapshots,
        existing_records=existing_records or [],
        smac_state=smac_state,
    )


def save_run_state(run: PreparedRun) -> None:
    output_path = run.settings.output_directory
    if run.smac_state is not None:
        load_trial_records(run.trials_path, repair=True)
    else:
        write_snapshot_collection(run.snapshots, run.snapshot_collection_path, relative_paths=True)
        write_snapshot_collection(
            run.snapshots,
            run.replay_snapshot_collection_path,
            deduplicate_matrices=run.settings.deduplicate_matrices,
        )
        write_configspace_json(run.parameter_search_space, output_path / "configspace.json")
        write_text_atomic(run.trials_path, "")
        remove_best_solver_configuration_outputs(output_path)
    write_yaml_file(output_path / "tuning_run.yaml", {
        "schema_version": 2,
        "settings": settings_to_dict(run.settings),
        "metadata": run.metadata,
    })


def run_started_event(run: PreparedRun) -> dict[str, Any]:
    return {
        **settings_to_dict(run.settings),
        **run.metadata,
        "event": "run_started",
        "output_directory": str(run.settings.output_directory),
        "parameter_search_space_name": run.metadata["parameter_search_space"],
        "default_solver_configuration": default_solver_configuration_from_parameter_search_space(
            run.parameter_search_space
        ),
        "completed_trials": run.completed_trials,
        "remaining_trials": (
            None
            if run.settings.run_until_stopped
            else max(0, run.metadata["effective_trials"] - run.completed_trials)
        ),
    }


class TuningProgressCallback(Callback):
    def __init__(self, run: PreparedRun, progress_callback: Callable | None) -> None:
        self.run = run
        self.progress_callback = progress_callback
        self.completed_trials = run.completed_trials
        self.queued_trials = run.completed_trials
        self.best_record = best_record_from_trial_records(run.existing_records)

    def on_ask_end(self, smbo: Any, info: Any) -> None:  # noqa: ARG002
        self.queued_trials += 1
        settings = self.run.settings
        emit_progress(
            self.progress_callback,
            {
                "event": "trial_queued",
                "trial_number": self.queued_trials,
                "trial_count": None if settings.run_until_stopped else settings.trials,
                "run_until_stopped": settings.run_until_stopped,
                "smac_configuration_tag": smac_configuration_tag(info.config),
                "solver_configuration": solver_configuration_from_configspace_configuration(
                    info.config
                ),
                "soft_timeout_sec": settings.soft_timeout_sec,
                "hard_timeout_sec": settings.hard_timeout_sec,
                "replay_startup_timeout_sec": settings.replay_startup_timeout_sec,
            },
        )

    def on_tell_end(self, smbo: Any, info: Any, value: Any) -> None:  # noqa: ARG002
        self.completed_trials += 1
        record = dict(value.additional_info.get("ksptune_trial_record") or {})
        if not record:
            solver_configuration = solver_configuration_from_configspace_configuration(info.config)
            record = {
                "smac_configuration_tag": smac_configuration_tag(info.config),
                "solver_configuration": solver_configuration,
                "petsc_options": render_petsc_options_from_solver_configuration(
                    self.run.parameter_search_space,
                    solver_configuration,
                ),
                "objective_name": self.run.settings.objective_name,
                "failure_reason": value.additional_info.get("error", "target function failed"),
            }
        record.update(trial_number=self.completed_trials, objective_value=value.cost)
        append_json_line(self.run.trials_path, record)
        improved = is_successful_trial(record) and (
            self.best_record is None
            or float(value.cost) < float(self.best_record["objective_value"])
        )
        if improved:
            self.best_record = record
            write_solver_configuration_outputs(
                output_directory=self.run.settings.output_directory,
                parameter_search_space=self.run.parameter_search_space,
                solver_configuration=record["solver_configuration"],
            )
        best = self.best_record or {}
        emit_progress(
            self.progress_callback,
            {
                "event": "trial_finished",
                "trial_count": None
                if self.run.settings.run_until_stopped
                else self.run.settings.trials,
                "run_until_stopped": self.run.settings.run_until_stopped,
                "best_objective_value_so_far": best.get("objective_value"),
                "best_trial_number_so_far": best.get("trial_number"),
                "best_smac_configuration_tag_so_far": best.get("smac_configuration_tag"),
                "is_new_best_so_far": improved,
                **record,
            },
        )


def create_smac(run: PreparedRun, callback: TuningProgressCallback) -> Any:
    from smac import HyperparameterOptimizationFacade, Scenario

    settings = run.settings
    replay_target_function = ReplayServerTargetFunction(
        settings=settings,
        parameter_search_space=run.parameter_search_space,
        snapshot_collection_path=run.replay_snapshot_collection_path,
        extra_replay_options=run.replay_options,
    )

    def target_function(configuration: Any, seed: int = 0) -> tuple[float, dict[str, Any]]:
        return replay_target_function(configuration, seed=seed)

    scenario_kwargs = {
        "configspace": run.parameter_search_space,
        "output_directory": settings.output_directory / "smac",
        "n_trials": run.metadata["effective_trials"],
        "seed": settings.seed,
        "deterministic": True,
        "crash_cost": BAD_COST,
        "n_workers": settings.workers,
        "use_default_config": run.metadata["use_default_solver_configuration"],
    }
    if run.smac_state is not None:
        scenario_kwargs["name"] = load_json_mapping(run.smac_state / "scenario.json")["name"]
    scenario = Scenario(**scenario_kwargs)
    sobol_count = run.metadata["sobol_initial_design_configurations"]
    use_default = run.metadata["use_default_solver_configuration"]
    # Preserve the saved design even when resuming an open-ended run with a small budget.
    initial_count = sobol_count + int(use_default) + len(run.initial_configurations)
    design_scenario = (
        replace(scenario, n_trials=initial_count)
        if initial_count > run.metadata["effective_trials"]
        else scenario
    )
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        design_scenario,
        n_configs=sobol_count,
        max_ratio=1.0,
        additional_configs=[],
    )
    if run.initial_configurations:
        initial_design = KSPTuneOrderedInitialDesign(
            base_initial_design=initial_design,
            leading_configurations=[
                record["configuration"] for record in run.initial_configurations
            ],
            include_default_configuration=use_default,
            configuration_space=run.parameter_search_space,
            include_base_configurations=use_default or sobol_count > 0,
        )
    original_files = {}
    updated_files = {}
    if run.smac_state is not None:
        scenario_path = run.smac_state / "scenario.json"
        original_files[scenario_path] = scenario_path.read_text(encoding="utf-8")
        scenario_data = json.loads(original_files[scenario_path])
        previous_workers = scenario_data["n_workers"]
        scenario_data.update(n_trials=run.metadata["effective_trials"], n_workers=settings.workers)
        saved_design = scenario_data.get("_meta", {}).get("initial_design")
        if saved_design is not None and saved_design != initial_design.meta:
            raise ValueError(
                "Cannot resume: saved SMAC initial design differs from the run metadata."
            )
        if previous_workers != settings.workers and "_meta" in scenario_data:
            from smac.runner.target_function_runner import TargetFunctionRunner

            scenario_data["_meta"]["runner"] = (
                {"name": "DaskParallelRunner"}
                if settings.workers > 1
                else TargetFunctionRunner(
                    scenario, target_function, required_arguments=["seed"]
                ).meta
            )
        updated_files[scenario_path] = scenario_data
        optimization_path = run.smac_state / "optimization.json"
        if optimization_path.exists():
            original_files[optimization_path] = optimization_path.read_text(encoding="utf-8")
            optimization_data = json.loads(original_files[optimization_path])
            # SMAC otherwise returns immediately despite an increased trial budget.
            optimization_data["finished"] = False
            updated_files[optimization_path] = optimization_data
        for name in ("runhistory.json", "intensifier.json"):
            load_json_mapping(run.smac_state / name)
    try:
        for path, data in updated_files.items():
            write_json_mapping(path, data)
        return HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=target_function,
            callbacks=[callback],
            initial_design=initial_design,
            overwrite=run.smac_state is None,
        )
    except BaseException:
        for path, text in original_files.items():
            write_text_atomic(path, text)
        raise


def finish_tuning(
    run: PreparedRun, status: str, progress_callback: Callable | None
) -> dict[str, Any]:
    summary = write_tuning_summary_outputs(
        output_directory=run.settings.output_directory,
        objective_name=run.settings.objective_name,
        status=status,
    )
    best = summary.get("best_trial") or {}
    solver_configuration = (
        default_solver_configuration_from_parameter_search_space(run.parameter_search_space)
        if status == "dry-run"
        else best.get("solver_configuration")
    )
    if solver_configuration is not None:
        write_solver_configuration_outputs(
            output_directory=run.settings.output_directory,
            parameter_search_space=run.parameter_search_space,
            solver_configuration=solver_configuration,
        )
    result = {
        "status": status,
        "output_directory": str(run.settings.output_directory),
        "summary": summary,
        **tuning_output_paths(run.settings.output_directory),
    }
    if status == "dry-run":
        result["default_solver_configuration"] = solver_configuration
    else:
        result.update(
            stopped_by_user=status == "stopped",
            best_cost=best.get("objective_value"),
            best_solver_configuration=solver_configuration,
            best_petsc_options=best.get("petsc_options"),
        )
    emit_progress(progress_callback, {"event": "run_finished", **result})
    return result


def execute_tuning(run: PreparedRun, progress_callback: Callable | None) -> dict[str, Any]:
    emit_progress(progress_callback, run_started_event(run))
    if run.settings.force_restart:
        clear_restart_outputs(run.settings.output_directory)
    status = "dry-run" if run.settings.dry_run else "completed"
    budget_reached = (
        run.smac_state is not None and run.completed_trials >= run.metadata["effective_trials"]
    )
    if run.settings.dry_run or budget_reached:
        save_run_state(run)
    else:
        callback = TuningProgressCallback(run, progress_callback)
        smac = create_smac(run, callback)
        try:
            save_run_state(run)
            optimize_smac_with_known_warning_filter(smac)
        except KeyboardInterrupt:
            status = "stopped"
            emit_progress(
                progress_callback,
                {
                    "event": "run_stopping",
                    "output_directory": str(run.settings.output_directory),
                    "trial_count": callback.completed_trials,
                },
            )
        finally:
            try:
                close_smac_replay_processes(smac)
            finally:
                try:
                    close_replay_processes()
                finally:
                    close_smac_runner(smac)
    return finish_tuning(run, status, progress_callback)


def run_tuning(
    settings: TuningSettings,
    *,
    parameter_search_space: Any,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return execute_tuning(prepare_tuning(settings, parameter_search_space), progress_callback)


RESUME_OVERRIDE_FIELDS = {
    "trials",
    "run_until_stopped",
    "workers",
    "soft_timeout_sec",
    "hard_timeout_sec",
    "replay_startup_timeout_sec",
    "hypre_hierarchy_diagnostics",
    "max_true_relative_residual",
    "max_true_residual_norm",
    "fast_fail",
}


def resume_tuning(
    run_directory: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    overrides = dict(overrides or {})
    unknown = overrides.keys() - RESUME_OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"Unknown resume overrides: {', '.join(sorted(unknown))}")
    overrides = {key: value for key, value in overrides.items() if value is not None}
    output_path = Path(run_directory).resolve()
    tuning_run_path = output_path / "tuning_run.yaml"
    trials_path = output_path / "solver_configuration_trials.jsonl"
    for path in (tuning_run_path, trials_path):
        if not path.is_file():
            raise ValueError(f"Cannot resume: {path.name} not found in {output_path}.")
    saved_run = load_yaml_mapping(tuning_run_path)
    if saved_run.get("schema_version") != 2:
        raise ValueError("Cannot resume: unsupported tuning run format; start a new run (schema version 2).")
    settings_values = saved_run["settings"]
    previous_run = saved_run["metadata"]
    if settings_values["dry_run"]:
        raise ValueError("Cannot resume a dry-run tuning directory.")
    settings_values.update(output_directory=output_path, force_restart=False)
    if previous_run["replay_binary_auto_detected"]:
        settings_values["replay_binary"] = None
    if "trials" in overrides:
        overrides.setdefault("run_until_stopped", False)
    settings_values.update(overrides)
    settings = TuningSettings(**settings_values)
    space = load_parameter_search_space(output_path / "configspace.json", seed=settings.seed)
    space._ksptuneinitial_solver_configurations = previous_run["initial_solver_configurations"]
    run = prepare_tuning(
        settings,
        space,
        previous_run=previous_run,
        existing_records=load_trial_records(trials_path),
        smac_state=find_resume_smac_state_directory(output_path, settings.seed),
    )
    return execute_tuning(run, progress_callback)
