from __future__ import annotations

import csv
import json
import sys
import warnings
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from ksptune.parameter_search_spaces import load_parameter_search_space
from ksptune.tuning_runs import (
    BAD_COST,
    count_tunable_hyperparameters,
    initial_solver_configuration_records_from_parameter_search_space,
    optimize_smac_with_known_warning_filter,
    resolve_sobol_initial_design_configuration_count,
    run_tuning,
)


def write_minimal_snapshot(directory: Path, solve_index: int = 0) -> None:
    file_stem = f"rmtest_6x5x4_trimmed_veri_nosmall__solve_{solve_index:06d}"
    (directory / f"{file_stem}__A.bin").write_bytes(b"A")
    (directory / f"{file_stem}__b.bin").write_bytes(b"b")
    (directory / f"{file_stem}__x0.bin").write_bytes(b"x0")


def test_initial_design_helpers_count_only_tunable_hyperparameters() -> None:
    parameter_search_space = load_parameter_search_space("petsc.hypre-basic")

    assert len(parameter_search_space) == 12
    assert count_tunable_hyperparameters(parameter_search_space) == 8
    assert resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=8,
        workers=1,
        effective_trials=100,
        default_solver_configuration_count=1,
        additional_solver_configuration_count=4,
    ) == 16
    assert resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=8,
        workers=8,
        effective_trials=100,
        default_solver_configuration_count=1,
        additional_solver_configuration_count=4,
    ) == 32
    assert resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=8,
        workers=4,
        effective_trials=12,
        default_solver_configuration_count=1,
        additional_solver_configuration_count=4,
    ) == 7


def test_hypre_basicinitial_solver_configurations_are_legal() -> None:
    parameter_search_space = load_parameter_search_space("petsc.hypre-basic")

    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    labels = [record["label"] for record in records]

    assert labels == [
        "joe_hypre_opts8",
        "joe_hypre_opts9_projected",
        "joe_hypre_opts4_projected",
        "joe_hypre_opts3_projected",
    ]
    assert "joe_hypre_opts11" not in labels
    for record in records:
        assert record["solver_configuration"]["ksp_type"] == "fgmres"
        assert record["solver_configuration"]["pc_type"] == "hypre"
        assert record["configuration"].origin.startswith(
            "KSPTune initial solver configuration:"
        )


def test_tuning_dry_run_writes_reproducible_files(tmp_path: Path) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    snapshot_directory = dump_directory

    output_directory = tmp_path / "run"
    result = run_tuning(
        snapshot_directory=snapshot_directory,
        parameter_search_space=load_parameter_search_space("petsc.hypre-basic"),
        output_directory=output_directory,
        dry_run=True,
    )

    assert result["status"] == "dry-run"
    assert (output_directory / "tuning_run.yaml").exists()
    assert (output_directory / "parameter_search_space.yaml").exists()
    assert (output_directory / "configspace.yaml").exists()
    assert (output_directory / "configspace.json").exists()
    assert (output_directory / "best_solver_configuration.yaml").exists()
    assert (output_directory / "best_petsc_options.txt").read_text(encoding="utf-8")
    assert (output_directory / "solver_configuration_trials.jsonl").exists()
    assert (output_directory / "solver_configuration_trials.jsonl").read_text(encoding="utf-8") == ""
    assert Path(result["tuning_summary_path"]).exists()
    assert Path(result["trial_csv_path"]).exists()
    assert Path(result["ranking_csv_path"]).exists()

    summary = yaml.safe_load((output_directory / "tuning_summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "dry-run"
    assert summary["trial_count"] == 0
    assert summary["best_trial"] is None

    rows = list(csv.DictReader((output_directory / "solver_configuration_trials.csv").open()))
    assert rows == []


def test_run_until_stopped_writes_best_so_far_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    snapshot_directory = dump_directory

    class DummyScenario:
        def __init__(
            self,
            *,
            configspace,
            output_directory,
            n_trials,
            seed,
            deterministic,
            crash_cost,
            n_workers,
            use_default_config,
        ):
            self.configspace = configspace
            self.output_directory = output_directory
            self.n_trials = n_trials
            self.seed = seed
            self.deterministic = deterministic
            self.crash_cost = crash_cost
            self.n_workers = n_workers
            self.use_default_config = use_default_config

    class DummyFacade:
        @staticmethod
        def get_initial_design(scenario, **kwargs):
            return SimpleNamespace(scenario=scenario, **kwargs)

        def __init__(self, *, scenario, target_function, callbacks, initial_design, overwrite):
            self.scenario = scenario
            self.target_function = target_function
            self.callbacks = callbacks
            self.initial_design = initial_design
            self.overwrite = overwrite

        def optimize(self):
            cost, additional_info = self.target_function(
                self.scenario.configspace.get_default_configuration()
            )
            trial_info = SimpleNamespace(config=self.scenario.configspace.get_default_configuration())
            trial_value = SimpleNamespace(cost=cost, additional_info=additional_info)
            for callback in self.callbacks:
                callback.on_tell_end(None, trial_info, trial_value)
            raise KeyboardInterrupt

    monkeypatch.setitem(
        sys.modules,
        "smac",
        SimpleNamespace(
            HyperparameterOptimizationFacade=DummyFacade,
            Scenario=DummyScenario,
        ),
    )

    def fake_run_replay_for_solver_configuration(**kwargs):
        return {
            "returncode": 0,
            "replay_result_path": str(kwargs["replay_result_path"]),
            "schema_errors": [],
            "subprocess_wall_time_sec": 0.2,
            "failure_reason": None,
            "replay_result": {
                "objective_time_sec_median": 0.1,
                "total_wall_time_sec": 0.3,
                "matrix_load_time_sec": 0.01,
                "solver_setup_time_sec": 0.02,
                "solve_time_sec_mean": 0.1,
                "solve_time_sec_median": 0.1,
                "iterations_total": 5,
                "snapshots": 1,
                "steps": [],
                "converged": True,
                "reason": "KSP_CONVERGED_RTOL",
                "reason_code": 2,
                "peak_memory_mb_max_per_rank": 64.0,
                "final_true_relative_residual_mean": 1.0e-9,
                "final_true_residual_norm_mean": 1.0e-10,
            },
        }

    monkeypatch.setattr(
        "ksptune.tuning_runs.run_replay_for_solver_configuration",
        fake_run_replay_for_solver_configuration,
    )

    events = []
    output_directory = tmp_path / "run"
    result = run_tuning(
        snapshot_directory=snapshot_directory,
        parameter_search_space=load_parameter_search_space("petsc.hypre-basic"),
        output_directory=output_directory,
        replay_binary="unused",
        run_until_stopped=True,
        progress_callback=events.append,
    )

    assert result["status"] == "stopped"
    assert result["stopped_by_user"] is True
    assert result["best_cost"] == 0.1
    assert (output_directory / "best_petsc_options.txt").exists()
    assert (output_directory / "best_petsc_options.txt").read_text(encoding="utf-8")

    summary = yaml.safe_load((output_directory / "tuning_summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "stopped"
    assert summary["trial_count"] == 1
    assert summary["best_trial"]["objective_value"] == 0.1

    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["run_until_stopped"] is True
    assert tuning_run["effective_trials"] > tuning_run["trials"]
    assert tuning_run["failed_replay_objective_value"] == BAD_COST
    assert tuning_run["smac_deterministic"] is True
    assert tuning_run["workers"] == 1
    trial_finished = next(event for event in events if event["event"] == "trial_finished")
    assert trial_finished["best_objective_value_so_far"] == 0.1
    assert trial_finished["best_trial_number_so_far"] == 1
    assert trial_finished["best_smac_configuration_tag_so_far"]
    assert trial_finished["is_new_best_so_far"] is True
    assert any(event["event"] == "run_stopping" for event in events)


def test_tuning_uses_auto_detected_replay_binary_when_not_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    snapshot_directory = dump_directory
    auto_replay_binary = tmp_path / "ksptune-petsc-replay"

    class DummyScenario:
        def __init__(
            self,
            *,
            configspace,
            output_directory,
            n_trials,
            seed,
            deterministic,
            crash_cost,
            n_workers,
            use_default_config,
        ):
            self.configspace = configspace
            self.output_directory = output_directory
            self.n_trials = n_trials
            self.seed = seed
            self.deterministic = deterministic
            self.crash_cost = crash_cost
            self.n_workers = n_workers
            self.use_default_config = use_default_config

    class DummyFacade:
        @staticmethod
        def get_initial_design(scenario, **kwargs):
            return SimpleNamespace(scenario=scenario, **kwargs)

        def __init__(self, *, scenario, target_function, callbacks, initial_design, overwrite):
            self.scenario = scenario
            self.target_function = target_function
            self.callbacks = callbacks
            self.initial_design = initial_design
            self.overwrite = overwrite

        def optimize(self):
            cost, additional_info = self.target_function(
                self.scenario.configspace.get_default_configuration()
            )
            trial_info = SimpleNamespace(config=self.scenario.configspace.get_default_configuration())
            trial_value = SimpleNamespace(cost=cost, additional_info=additional_info)
            for callback in self.callbacks:
                callback.on_tell_end(None, trial_info, trial_value)

    monkeypatch.setitem(
        sys.modules,
        "smac",
        SimpleNamespace(
            HyperparameterOptimizationFacade=DummyFacade,
            Scenario=DummyScenario,
        ),
    )
    monkeypatch.setattr(
        "ksptune.tuning_runs.find_default_replay_binary",
        lambda: auto_replay_binary,
    )

    seen_replay_binaries = []
    seen_extra_replay_options = []

    def fake_run_replay_for_solver_configuration(**kwargs):
        seen_replay_binaries.append(kwargs["replay_binary"])
        seen_extra_replay_options.append(kwargs["extra_replay_options"])
        return {
            "command": [
                str(kwargs["replay_binary"]),
                "-snapshot_collection",
                str(kwargs["snapshot_collection_path"]),
                *kwargs["extra_replay_options"],
                "-ksp_type",
                "cg",
            ],
            "returncode": 0,
            "replay_result_path": str(kwargs["replay_result_path"]),
            "schema_errors": [],
            "subprocess_wall_time_sec": 0.2,
            "failure_reason": None,
            "replay_result": {
                "objective_time_sec_median": 0.1,
                "total_wall_time_sec": 0.3,
                "matrix_load_time_sec": 0.01,
                "solver_setup_time_sec": 0.02,
                "solve_time_sec_mean": 0.1,
                "solve_time_sec_median": 0.1,
                "iterations_total": 5,
                "snapshots": 1,
                "steps": [],
                "converged": True,
                "reason": "KSP_CONVERGED_RTOL",
                "reason_code": 2,
                "peak_memory_mb_max_per_rank": 64.0,
                "final_true_relative_residual_mean": 1.0e-9,
                "final_true_residual_norm_mean": 1.0e-10,
            },
        }

    monkeypatch.setattr(
        "ksptune.tuning_runs.run_replay_for_solver_configuration",
        fake_run_replay_for_solver_configuration,
    )

    events = []
    output_directory = tmp_path / "run"
    result = run_tuning(
        snapshot_directory=snapshot_directory,
        parameter_search_space=load_parameter_search_space("petsc.hypre-basic"),
        output_directory=output_directory,
        trials=1,
        workers=3,
        nullspace="field:1,block_size=2",
        progress_callback=events.append,
    )

    assert result["status"] == "completed"
    assert seen_replay_binaries == [auto_replay_binary]
    assert seen_extra_replay_options == [
        [
            "-replay_nullspace",
            "field",
            "-replay_field_nullspace",
            "1",
            "-replay_field_nullspace_block_size",
            "2",
        ]
    ]
    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["replay_binary"] == str(auto_replay_binary)
    assert tuning_run["replay_binary_auto_detected"] is True
    assert tuning_run["nullspace"]["label"] == "field:1"
    assert tuning_run["workers"] == 3
    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["replay_binary"] == str(auto_replay_binary)
    assert run_started["replay_binary_auto_detected"] is True
    assert run_started["nullspace_configuration"]["label"] == "field:1"
    assert run_started["workers"] == 3

    trial_record = json.loads(
        (output_directory / "solver_configuration_trials.jsonl").read_text(encoding="utf-8")
    )
    assert trial_record["replay_command"][:2] == [str(auto_replay_binary), "-snapshot_collection"]
    assert "-replay_nullspace" in trial_record["replay_command"]
    assert "field" in trial_record["replay_command"]
    assert trial_record["replay_command_text"].startswith(str(auto_replay_binary))


def test_tuning_configures_default_and_seeded_initial_design(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    snapshot_directory = dump_directory

    captured = {}

    class DummyScenario:
        def __init__(
            self,
            *,
            configspace,
            output_directory,
            n_trials,
            seed,
            deterministic,
            crash_cost,
            n_workers,
            use_default_config,
        ):
            self.configspace = configspace
            self.output_directory = output_directory
            self.n_trials = n_trials
            self.seed = seed
            self.deterministic = deterministic
            self.crash_cost = crash_cost
            self.n_workers = n_workers
            self.use_default_config = use_default_config
            captured["scenario"] = self

    class DummyFacade:
        @staticmethod
        def get_initial_design(scenario, **kwargs):
            captured["initial_design_kwargs"] = kwargs
            return SimpleNamespace(scenario=scenario, **kwargs)

        def __init__(self, *, scenario, target_function, callbacks, initial_design, overwrite):
            self.scenario = scenario
            self.target_function = target_function
            self.callbacks = callbacks
            self.initial_design = initial_design
            self.overwrite = overwrite
            captured["initial_design"] = initial_design

        def optimize(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "smac",
        SimpleNamespace(
            HyperparameterOptimizationFacade=DummyFacade,
            Scenario=DummyScenario,
        ),
    )

    events = []
    output_directory = tmp_path / "run"
    run_tuning(
        snapshot_directory=snapshot_directory,
        parameter_search_space=load_parameter_search_space("petsc.hypre-basic"),
        output_directory=output_directory,
        replay_binary="unused",
        trials=12,
        workers=4,
        progress_callback=events.append,
    )

    assert captured["scenario"].use_default_config is True
    assert captured["initial_design"] is not None
    assert captured["initial_design_kwargs"]["n_configs"] == 7
    assert captured["initial_design_kwargs"]["max_ratio"] == 1.0
    assert len(captured["initial_design_kwargs"]["additional_configs"]) == 4

    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["use_default_solver_configuration"] is True
    assert tuning_run["tunable_parameter_count"] == 8
    assert tuning_run["sobol_initial_design_configurations"] == 7
    assert tuning_run["additionalinitial_solver_configuration_count"] == 4
    assert tuning_run["additionalinitial_solver_configurations"][0] == "joe_hypre_opts8"

    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["sobol_initial_design_configurations"] == 7
    assert run_started["additionalinitial_solver_configuration_count"] == 4


def test_known_smac_empty_neighbor_runtime_warning_is_filtered() -> None:
    class DummySmac:
        def optimize(self):
            warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)
            warnings.warn(
                "invalid value encountered in scalar divide",
                RuntimeWarning,
                stacklevel=2,
            )
            return "done"

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = optimize_smac_with_known_warning_filter(DummySmac())

    assert result == "done"
    assert captured_warnings == []


def test_unrelated_runtime_warnings_are_not_filtered() -> None:
    class DummySmac:
        def optimize(self):
            warnings.warn("different runtime warning", RuntimeWarning, stacklevel=2)

    with pytest.raises(RuntimeWarning, match="different runtime warning"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            optimize_smac_with_known_warning_filter(DummySmac())
