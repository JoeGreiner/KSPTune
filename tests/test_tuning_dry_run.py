from __future__ import annotations

import csv
import json
import sys
import warnings
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from test_replay import valid_replay_result

from ksptune.parameter_search_spaces import load_parameter_search_space, write_configspace_json
from ksptune.snapshot_collections import load_snapshots_from_directory, write_snapshot_collection
from ksptune.trial_records import (
    BAD_COST,
    load_trial_records,
)
from ksptune.tuning_runs import (
    TuningSettings,
    settings_to_dict,
    HYPRE_HIERARCHY_REPLAY_OPTION,
    KSPTuneOrderedInitialDesign,
    REPLAY_BINARY_ENV_VAR,
    count_tunable_hyperparameters,
    find_default_replay_binary,
    initial_solver_configuration_records_from_parameter_search_space,
    optimize_smac_with_known_warning_filter,
    petsc_options_with_override,
    resolve_hypre_hierarchy_diagnostics,
    resolve_replay_binary,
    resolve_sobol_initial_design_configuration_count,
    resume_tuning,
    run_tuning,
)


def test_petsc_options_with_override_replaces_existing_option() -> None:
    assert petsc_options_with_override(
        ["-ksp_type", "gmres", "-ksp_max_it", "10000", "-pc_type", "hypre"],
        "-ksp_max_it",
        1,
    ) == ["-ksp_type", "gmres", "-pc_type", "hypre", "-ksp_max_it", "1"]


def test_petsc_options_with_override_replaces_equals_form() -> None:
    assert petsc_options_with_override(
        ["-ksp_type", "gmres", "-ksp_max_it=10000"],
        "-ksp_max_it",
        1,
    ) == ["-ksp_type", "gmres", "-ksp_max_it", "1"]


def write_minimal_snapshot(directory: Path, solve_index: int = 0) -> None:
    snapshot_id = f"example_{solve_index:06d}"
    (directory / f"A_{snapshot_id}.bin").write_bytes(b"A")
    (directory / f"A_{snapshot_id}.txt").write_text("rows=12\ncols=12\nmpi_size=1\n")
    (directory / f"b_{snapshot_id}.bin").write_bytes(b"b")
    (directory / f"x0_{snapshot_id}.bin").write_bytes(b"x0")
    (directory / f"s_{snapshot_id}.txt").write_text(
        f"solve_index={solve_index}\n"
        f"A=A_{snapshot_id}.bin\nmatrix_metadata=A_{snapshot_id}.txt\n"
        f"b=b_{snapshot_id}.bin\nx0=x0_{snapshot_id}.bin\nrhs_size=12\n"
    )


def write_executable_file(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_hypre_hierarchy_diagnostics_auto_enables_reported_option(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_binary = write_executable_file(tmp_path / "dummy-replay")
    seen = {}

    def fake_query_petsc_options_help(**kwargs):
        seen.update(kwargs)
        return {
            "returncode": 0,
            "filtered_lines": ["-pc_hypre_boomeramg_view_hierarchy <bool>"],
            "option_lines": [],
            "raw_output": "",
            "command": [str(replay_binary), "-replay_options_help"],
        }

    monkeypatch.setattr(
        "ksptune.tuning_runs.query_petsc_options_help",
        fake_query_petsc_options_help,
    )

    result = resolve_hypre_hierarchy_diagnostics(
        mode="auto",
        replay_binary=replay_binary,
        dry_run=False,
    )

    assert result["available"] is True
    assert result["enabled"] is True
    assert seen["petsc_options"] == ["-pc_type", "hypre", "-pc_hypre_type", "boomeramg"]


def test_resolve_hypre_hierarchy_diagnostics_auto_disables_missing_option(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_binary = write_executable_file(tmp_path / "dummy-replay")

    monkeypatch.setattr(
        "ksptune.tuning_runs.query_petsc_options_help",
        lambda **kwargs: {
            "returncode": 0,
            "filtered_lines": [],
            "option_lines": [],
            "raw_output": "",
            "command": [str(replay_binary), "-replay_options_help"],
        },
    )

    result = resolve_hypre_hierarchy_diagnostics(
        mode="auto",
        replay_binary=replay_binary,
        dry_run=False,
    )

    assert result["available"] is False
    assert result["enabled"] is False
    assert result["reason"] == "option not reported by linked PETSc"


def test_resolve_hypre_hierarchy_diagnostics_on_rejects_missing_option(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_binary = write_executable_file(tmp_path / "dummy-replay")

    monkeypatch.setattr(
        "ksptune.tuning_runs.query_petsc_options_help",
        lambda **kwargs: {
            "returncode": 0,
            "filtered_lines": [],
            "option_lines": [],
            "raw_output": "",
            "command": [str(replay_binary), "-replay_options_help"],
        },
    )

    with pytest.raises(ValueError, match="not available"):
        resolve_hypre_hierarchy_diagnostics(
            mode="on",
            replay_binary=replay_binary,
            dry_run=False,
        )


def test_find_default_replay_binary_uses_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_binary = write_executable_file(tmp_path / "custom-replay")
    monkeypatch.setenv(REPLAY_BINARY_ENV_VAR, str(replay_binary))

    assert find_default_replay_binary() == replay_binary.resolve()


def test_find_default_replay_binary_rejects_missing_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_binary = tmp_path / "missing-replay"
    monkeypatch.setenv(REPLAY_BINARY_ENV_VAR, str(replay_binary))

    with pytest.raises(ValueError, match="missing replay binary"):
        find_default_replay_binary()


def test_find_default_replay_binary_rejects_non_executable_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_binary = tmp_path / "custom-replay"
    replay_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    replay_binary.chmod(0o644)
    monkeypatch.setenv(REPLAY_BINARY_ENV_VAR, str(replay_binary))

    with pytest.raises(ValueError, match="non-executable replay binary"):
        find_default_replay_binary()


def test_resolve_replay_binary_error_mentions_default_locations(monkeypatch) -> None:
    monkeypatch.delenv(REPLAY_BINARY_ENV_VAR, raising=False)
    monkeypatch.setattr("ksptune.tuning_runs.find_default_replay_binary", lambda: None)

    with pytest.raises(ValueError) as error:
        resolve_replay_binary(None, dry_run=False)

    assert REPLAY_BINARY_ENV_VAR in str(error.value)
    assert "PATH" in str(error.value)


def test_resolve_replay_binary_rejects_missing_explicit_path(tmp_path: Path) -> None:
    replay_binary = tmp_path / "missing-replay"

    with pytest.raises(ValueError, match="--replay-binary points to a missing replay binary"):
        resolve_replay_binary(replay_binary, dry_run=False)


def write_resume_run(
    output_directory: Path,
    snapshot_directory: Path,
    *,
    replay_binary: Path,
    trials: int | None = 3,
    run_until_stopped: bool = False,
    workers: int = 1,
    soft_timeout_sec: float | None = None,
    hard_timeout_sec: float | None = None,
    replay_startup_timeout_sec: float | None = 1200.0,
    max_true_relative_residual: float = 2.0e-4,
    max_true_residual_norm: float = 3.0e-4,
) -> None:
    output_directory.mkdir()
    snapshot_collection_path = output_directory / "snapshot_collection.csv"
    replay_snapshot_collection_path = output_directory / "snapshot_collection.resolved.csv"
    snapshots = load_snapshots_from_directory(snapshot_directory)
    write_snapshot_collection(snapshots, snapshot_collection_path, relative_paths=True)
    write_snapshot_collection(snapshots, replay_snapshot_collection_path)
    write_configspace_json(load_parameter_search_space("petsc.boomeramg_basic"),
                           output_directory / "configspace.json")
    yaml.safe_dump(
        {
            "schema_version": 2,
            "settings": settings_to_dict(TuningSettings(
                snapshot_directory=snapshot_directory, output_directory=output_directory,
                replay_binary=replay_binary, trials=trials, workers=workers,
                run_until_stopped=run_until_stopped, soft_timeout_sec=soft_timeout_sec,
                hard_timeout_sec=hard_timeout_sec, replay_startup_timeout_sec=replay_startup_timeout_sec,
                hypre_hierarchy_diagnostics="off", max_true_relative_residual=max_true_relative_residual,
                max_true_residual_norm=max_true_residual_norm,
            )),
            "metadata": {
                "snapshot_collection_path": str(snapshot_collection_path),
                "replay_snapshot_collection_path": str(replay_snapshot_collection_path),
                "parameter_search_space": "petsc.boomeramg_basic",
                "parameter_search_space_source": "builtin:petsc.boomeramg_basic",
                "use_default_solver_configuration": True,
                "sobol_initial_design_configurations": 1,
                "initial_solver_configurations": [],
                "replay_binary_auto_detected": False,
            },
        },
        (output_directory / "tuning_run.yaml").open("w", encoding="utf-8"),
        sort_keys=False,
    )
    (output_directory / "solver_configuration_trials.jsonl").write_text(
        json.dumps(
            {
                "trial_number": 1,
                "smac_configuration_tag": "old001",
                "solver_configuration": {"ksp_type": "cg", "pc_type": "jacobi"},
                "petsc_options": ["-ksp_type", "cg", "-pc_type", "jacobi"],
                "objective_name": "solve_time_sec_mean",
                "objective_value": 0.5,
                "failure_reason": None,
                "converged": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    smac_state_directory = output_directory / "smac" / "oldsmac" / "1"
    smac_state_directory.mkdir(parents=True)
    (smac_state_directory / "scenario.json").write_text(
        json.dumps(
            {
                "name": "oldsmac",
                "n_trials": trials or 2_147_483_647,
                "n_workers": workers,
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )
    (smac_state_directory / "runhistory.json").write_text("{}", encoding="utf-8")
    (smac_state_directory / "intensifier.json").write_text("{}", encoding="utf-8")


def test_initial_design_helpers_count_only_tunable_hyperparameters() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")

    assert len(parameter_search_space) == 15
    assert count_tunable_hyperparameters(parameter_search_space) == 8
    assert resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=8,
        workers=1,
        effective_trials=100,
        default_solver_configuration_count=1,
        additional_solver_configuration_count=4,
    ) == 3
    assert resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=8,
        workers=8,
        effective_trials=100,
        default_solver_configuration_count=1,
        additional_solver_configuration_count=4,
    ) == 27
    assert resolve_sobol_initial_design_configuration_count(
        tunable_parameter_count=8,
        workers=4,
        effective_trials=12,
        default_solver_configuration_count=1,
        additional_solver_configuration_count=4,
    ) == 7


def test_ordered_initial_design_puts_seeded_configurations_before_default_and_sobol() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")
    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    default_configuration = parameter_search_space.get_default_configuration()
    default_configuration.origin = "Initial Design: Default configuration"
    sobol_like_configuration = records[4]["configuration"]

    base_initial_design = SimpleNamespace(
        meta={"name": "DummySobol"},
        select_configurations=lambda: [
            sobol_like_configuration,
            default_configuration,
            records[0]["configuration"],
        ],
    )
    initial_design = KSPTuneOrderedInitialDesign(
        base_initial_design=base_initial_design,
        leading_configurations=[
            records[0]["configuration"],
            records[1]["configuration"],
        ],
        include_default_configuration=True,
        configuration_space=parameter_search_space,
    )

    configurations = initial_design.select_configurations()

    assert configurations == [
        records[0]["configuration"],
        records[1]["configuration"],
        default_configuration,
        sobol_like_configuration,
    ]


def test_boomeramg_basic_initial_solver_configurations_are_legal() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")

    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    labels = [record["label"] for record in records]

    assert labels == [
        "boomeramg_exti_threshold_0p05",
        "boomeramg_ff_agg_1_paths_2",
        "boomeramg_hmis_exti_pmax_3",
        "boomeramg_exti_pmax_8",
        "boomeramg_pmis_exticc_agg_0",
        "boomeramg_pmis_ff_pmax_3_paths_1",
        "boomeramg_pmis_extimm_pmax_3_paths_1",
        "boomeramg_sor_pmis_extemm_pmax_15_paths_3",
        "boomeramg_hmis_exti_paths_2",
        "boomeramg_pmis_exti_paths_2",
        "boomeramg_hmis_extimm_paths_2",
        "boomeramg_pmis_exti_paths_3",
        "boomeramg_hmis_ff_paths_1",
        "boomeramg_pmis_extimm_agg_0",
        "boomeramg_pmis_ff_agg_0",
        "boomeramg_sor_pmis_exti_pmax_0_paths_3",
        "boomeramg_sor_pmis_ext_pmax_12_paths_3",
        "boomeramg_sor_hmis_extemm_pmax_10_paths_3",
    ]
    for record in records:
        assert record["solver_configuration"]["ksp_type"] == "gmres"
        assert record["solver_configuration"]["ksp_pc_side"] == "left"
        assert record["solver_configuration"]["pc_type"] == "hypre"
        assert record["configuration"].origin.startswith(
            "KSPTune initial solver configuration:"
        )


def test_boomeramg_extended_initial_solver_configurations_are_legal() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_extended")

    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    labels = [record["label"] for record in records]

    assert "boomeramg_pmis_exticc_agg_0" in labels
    assert labels[-7:] == [
        "boomeramg_hmis_exti_paths_2",
        "boomeramg_pmis_exti_paths_2",
        "boomeramg_hmis_extimm_paths_2",
        "boomeramg_pmis_exti_paths_3",
        "boomeramg_hmis_ff_paths_1",
        "boomeramg_pmis_extimm_agg_0",
        "boomeramg_pmis_ff_agg_0",
    ]
    for record in records:
        assert record["solver_configuration"]["ksp_type"] == "gmres"
        assert record["solver_configuration"]["ksp_pc_side"] == "left"
        assert record["solver_configuration"]["pc_type"] == "hypre"
        assert record["configuration"].origin.startswith(
            "KSPTune initial solver configuration:"
        )


def test_gamg_initial_solver_configurations_are_legal() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_basic")

    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    labels = [record["label"] for record in records]

    assert labels == [
        "gamg_mis_chebyshev_jacobi_compact",
        "gamg_mis_parallel_coarse_compact",
        "gamg_mis_low_peak_compact",
        "gamg_mis_repartition_spread",
        "gamg_mis_sor_low_mpi",
        "gamg_misk_startup_seed",
        "gamg_mis_threshold_0_baseline",
    ]
    for record in records:
        solver_configuration = record["solver_configuration"]
        assert solver_configuration["ksp_type"] == "gmres"
        assert solver_configuration["ksp_pc_side"] == "left"
        assert solver_configuration["pc_type"] == "gamg"
        assert record["configuration"].origin.startswith(
            "KSPTune initial solver configuration:"
        )

    ok_mis = next(
        record
        for record in records
        if record["label"] == "gamg_mis_threshold_0_baseline"
    )
    assert ok_mis["solver_configuration"]["pc_gamg_mat_coarsen_type"] == "mis"
    assert ok_mis["solver_configuration"]["pc_gamg_aggressive_square_graph"] is False
    assert ok_mis["solver_configuration"]["pc_gamg_aggressive_mis_k"] == 2
    assert "pc_gamg_mat_coarsen_misk_distance" not in ok_mis["solver_configuration"]
    ok_misk = next(
        record
        for record in records
        if record["label"]
        == "gamg_misk_startup_seed"
    )
    assert ok_misk["solver_configuration"]["mg_levels_ksp_type"] == "chebyshev"
    assert ok_misk["solver_configuration"]["mg_levels_pc_type"] == "jacobi"
    assert ok_misk["solver_configuration"]["pc_gamg_mat_coarsen_type"] == "misk"
    assert ok_misk["solver_configuration"]["pc_gamg_mat_coarsen_misk_distance"] == 1
    best_mis = next(
        record
        for record in records
        if record["label"] == "gamg_mis_chebyshev_jacobi_compact"
    )
    assert best_mis["solver_configuration"]["pc_gamg_mat_coarsen_type"] == "mis"
    assert best_mis["solver_configuration"]["pc_gamg_aggressive_square_graph"] is False
    assert best_mis["solver_configuration"]["pc_gamg_threshold_scale"] == 1.0
    assert best_mis["solver_configuration"]["pc_gamg_parallel_coarse_grid_solver"] is False
    low_mpi = next(
        record
        for record in records
        if record["label"] == "gamg_mis_sor_low_mpi"
    )
    assert low_mpi["solver_configuration"]["mg_levels_pc_type"] == "sor"
    assert low_mpi["solver_configuration"]["pc_gamg_agg_nsmooths"] == 2
    repartition = next(
        record
        for record in records
        if record["label"] == "gamg_mis_repartition_spread"
    )
    assert repartition["solver_configuration"]["pc_gamg_repartition"] is True
    assert repartition["solver_configuration"]["pc_gamg_parallel_coarse_grid_solver"] is True


def test_gamg_mono_initial_solver_configurations_are_legal() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_extended_mono")

    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )
    labels = [record["label"] for record in records]

    assert labels[-1] == "gamg_chebyshev_rowl1"
    assert "gamg_mis_low_memory_filter_probe" not in labels
    for record in records:
        solver_configuration = record["solver_configuration"]
        assert solver_configuration["ksp_type"] == "gmres"
        assert solver_configuration["ksp_pc_side"] == "left"
        assert solver_configuration["pc_type"] == "gamg"
        assert solver_configuration["ksp_rtol"] == 1.0e-8
        assert solver_configuration["mg_coarse_pc_type"] != "ilu"
        assert solver_configuration["pc_gamg_low_memory_threshold_filter"] is False
        if solver_configuration["mg_levels_ksp_type"] == "chebyshev":
            assert solver_configuration["mg_levels_ksp_chebyshev_esteig_noisy"] is True
        assert not any(
            "nullspace" in parameter_name or "block_size" in parameter_name
            for parameter_name in solver_configuration
        )
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
        TuningSettings(
            snapshot_directory=snapshot_directory,
            output_directory=output_directory,
            soft_timeout_sec=100.0,
            replay_startup_timeout_sec=123.0,
            dry_run=True,
        ),
        parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
    )

    assert result["status"] == "dry-run"
    assert (output_directory / "tuning_run.yaml").exists()
    assert not (output_directory / "parameter_search_space.yaml").exists()
    assert not (output_directory / "configspace.yaml").exists()
    assert (output_directory / "configspace.json").exists()
    assert (output_directory / "best_solver_configuration.yaml").exists()
    assert (output_directory / "best_petsc_options.txt").read_text(encoding="utf-8")
    assert (output_directory / "solver_configuration_trials.jsonl").exists()
    assert (output_directory / "solver_configuration_trials.jsonl").read_text(
        encoding="utf-8"
    ) == ""
    assert Path(result["tuning_summary_path"]).exists()
    assert Path(result["trial_csv_path"]).exists()
    assert Path(result["ranking_csv_path"]).exists()

    summary = yaml.safe_load((output_directory / "tuning_summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "dry-run"
    assert summary["trial_count"] == 0
    assert summary["best_trial"] is None

    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["settings"]["deduplicate_matrices"] is True
    assert tuning_run["settings"]["reuse_ksp_setup"] is True
    assert tuning_run["settings"]["use_initial_guess"] is True
    assert tuning_run["settings"]["soft_timeout_sec"] == 100.0
    assert tuning_run["settings"]["hard_timeout_sec"] == 150.0
    assert tuning_run["settings"]["replay_startup_timeout_sec"] == 123.0
    assert tuning_run["settings"]["max_true_relative_residual"] == 1.0e-4
    assert tuning_run["settings"]["max_true_residual_norm"] == 1.0e-4
    assert tuning_run["settings"]["fast_fail"] is False

    rows = list(csv.DictReader((output_directory / "solver_configuration_trials.csv").open()))
    assert rows == []


def test_tuning_dry_run_records_fast_fail_mode(tmp_path: Path) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)

    output_directory = tmp_path / "run"
    run_tuning(
        TuningSettings(
            snapshot_directory=dump_directory,
            output_directory=output_directory,
            dry_run=True,
            fast_fail=True,
        ),
        parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
    )

    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["settings"]["fast_fail"] is True
    assert tuning_run["metadata"]["fast_fail_ksp_max_it"] == 1


def test_tuning_refuses_existing_run_directory_without_force_restart(tmp_path: Path) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    output_directory = tmp_path / "run"
    output_directory.mkdir()
    (output_directory / "tuning_run.yaml").write_text("status: old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already contains a KSPTune run"):
        run_tuning(
            TuningSettings(
                snapshot_directory=dump_directory,
                output_directory=output_directory,
                dry_run=True,
            ),
            parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
        )


def test_tuning_force_restart_allows_existing_run_directory(tmp_path: Path) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    output_directory = tmp_path / "run"
    output_directory.mkdir()
    (output_directory / "tuning_run.yaml").write_text("status: old\n", encoding="utf-8")
    (output_directory / "solver_configuration_trials.jsonl").write_text(
        '{"trial_number": 99}\n',
        encoding="utf-8",
    )

    result = run_tuning(
        TuningSettings(
            snapshot_directory=dump_directory,
            output_directory=output_directory,
            dry_run=True,
            force_restart=True,
        ),
        parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
    )

    assert result["status"] == "dry-run"
    assert (output_directory / "solver_configuration_trials.jsonl").read_text(
        encoding="utf-8"
    ) == ""
    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["settings"]["force_restart"] is True


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
            assert hasattr(target_function, "__code__")
            self.scenario = scenario
            self.target_function = target_function
            self.callbacks = callbacks
            self.initial_design = initial_design
            self.overwrite = overwrite

        def optimize(self):
            cost, additional_info = self.target_function(
                self.scenario.configspace.get_default_configuration()
            )
            trial_info = SimpleNamespace(
                config=self.scenario.configspace.get_default_configuration()
            )
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

    def fake_run_replay_server_for_solver_configuration(**kwargs):
        return {
            "returncode": 0,
            "replay_result_path": str(kwargs["replay_result_path"]),
            "schema_errors": [],
            "subprocess_wall_time_sec": 0.2,
            "failure_reason": None,
            "replay_result": {
                **valid_replay_result(),
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
                "rss_request_peak_sample_mb_sum": 128.0,
                "rss_peak_sample_mb_max_rank": 64.0,
                "rss_peak_sample_mb_max_node": 128.0,
                "final_true_relative_residual_mean": 1.0e-9,
                "final_true_residual_norm_mean": 1.0e-10,
            },
        }

    monkeypatch.setattr(
        "ksptune.tuning_runs.run_replay_server_for_solver_configuration",
        fake_run_replay_server_for_solver_configuration,
    )

    events = []
    output_directory = tmp_path / "run"
    replay_binary = write_executable_file(tmp_path / "dummy-replay")
    result = run_tuning(
        TuningSettings(
            snapshot_directory=snapshot_directory,
            output_directory=output_directory,
            replay_binary=replay_binary,
            run_until_stopped=True,
        ),
        parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
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
    assert tuning_run["settings"]["run_until_stopped"] is True
    assert tuning_run["metadata"]["effective_trials"] > 0
    assert tuning_run["settings"]["trials"] is None
    assert tuning_run["metadata"]["failed_replay_objective_value"] == BAD_COST
    assert tuning_run["metadata"]["smac_deterministic"] is True
    assert tuning_run["settings"]["workers"] == 1
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
            trial_info = SimpleNamespace(
                config=self.scenario.configspace.get_default_configuration()
            )
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
    monkeypatch.setattr(
        "ksptune.tuning_runs.query_petsc_options_help",
        lambda **kwargs: {
            "returncode": 0,
            "filtered_lines": ["-pc_hypre_boomeramg_view_hierarchy <bool>"],
            "option_lines": [],
            "raw_output": "",
            "command": [str(auto_replay_binary), "-replay_options_help"],
        },
    )

    seen_replay_binaries = []
    seen_extra_replay_options = []
    seen_petsc_options = []

    def fake_run_replay_server_for_solver_configuration(**kwargs):
        seen_replay_binaries.append(kwargs["replay_binary"])
        seen_extra_replay_options.append(kwargs["extra_replay_options"])
        seen_petsc_options.append(kwargs["petsc_options"])
        return {
            "command": [
                str(kwargs["replay_binary"]),
                "-snapshot_collection",
                str(kwargs["snapshot_collection_path"]),
                *kwargs["extra_replay_options"],
                *kwargs["petsc_options"],
                "-ksp_type",
                "cg",
            ],
            "returncode": 0,
            "replay_result_path": str(kwargs["replay_result_path"]),
            "schema_errors": [],
            "subprocess_wall_time_sec": 0.2,
            "failure_reason": None,
            "replay_result": {
                **valid_replay_result(),
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
                "rss_request_peak_sample_mb_sum": 128.0,
                "rss_peak_sample_mb_max_rank": 64.0,
                "rss_peak_sample_mb_max_node": 128.0,
                "final_true_relative_residual_mean": 1.0e-9,
                "final_true_residual_norm_mean": 1.0e-10,
            },
        }

    monkeypatch.setattr(
        "ksptune.tuning_runs.run_replay_server_for_solver_configuration",
        fake_run_replay_server_for_solver_configuration,
    )

    events = []
    output_directory = tmp_path / "run"
    result = run_tuning(
        TuningSettings(
            snapshot_directory=snapshot_directory,
            output_directory=output_directory,
            trials=1,
            workers=3,
            nullspace="field:1,block_size=2",
            reuse_ksp_setup=False,
            use_initial_guess=False,
            fast_fail=True,
        ),
        parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
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
            "-replay_reuse_ksp_setup",
            "false",
            "-replay_use_initial_guess",
            "false",
            HYPRE_HIERARCHY_REPLAY_OPTION,
            "true",
        ]
    ]
    assert seen_petsc_options[0][-2:] == ["-ksp_max_it", "1"]
    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["settings"]["replay_binary"] == str(auto_replay_binary)
    assert tuning_run["metadata"]["replay_binary_auto_detected"] is True
    assert tuning_run["metadata"]["nullspace_configuration"]["label"] == "field:1"
    assert tuning_run["settings"]["reuse_ksp_setup"] is False
    assert tuning_run["settings"]["use_initial_guess"] is False
    assert tuning_run["metadata"]["hypre_hierarchy_diagnostics"]["enabled"] is True
    assert tuning_run["settings"]["workers"] == 3
    assert tuning_run["settings"]["fast_fail"] is True
    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["replay_binary"] == str(auto_replay_binary)
    assert run_started["replay_binary_auto_detected"] is True
    assert run_started["nullspace_configuration"]["label"] == "field:1"
    assert run_started["reuse_ksp_setup"] is False
    assert run_started["use_initial_guess"] is False
    assert run_started["workers"] == 3
    assert run_started["fast_fail"] is True

    trial_record = json.loads(
        (output_directory / "solver_configuration_trials.jsonl").read_text(encoding="utf-8")
    )
    assert trial_record["replay_command"][:2] == [str(auto_replay_binary), "-snapshot_collection"]
    assert "-replay_nullspace" in trial_record["replay_command"]
    assert "field" in trial_record["replay_command"]
    assert "-replay_reuse_ksp_setup" in trial_record["replay_command"]
    assert "-replay_use_initial_guess" in trial_record["replay_command"]
    assert "-ksp_max_it" in trial_record["replay_command"]
    assert trial_record["fast_fail"] is True
    assert trial_record["fast_fail_ksp_max_it"] == 1
    assert trial_record["replay_command_text"].startswith(str(auto_replay_binary))


def test_resume_tuning_appends_trials_and_reuses_smac_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    replay_binary = write_executable_file(tmp_path / "ksptune-petsc-replay")
    output_directory = tmp_path / "run"
    write_resume_run(
        output_directory,
        dump_directory,
        replay_binary=replay_binary,
        trials=3,
        soft_timeout_sec=90.0,
        hard_timeout_sec=135.0,
    )

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
            name=None,
        ):
            self.configspace = configspace
            self.output_directory = output_directory
            self.n_trials = n_trials
            self.seed = seed
            self.deterministic = deterministic
            self.crash_cost = crash_cost
            self.n_workers = n_workers
            self.use_default_config = use_default_config
            self.name = name
            captured["scenario"] = self

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
            captured["overwrite"] = overwrite

        def optimize(self):
            cost, additional_info = self.target_function(
                self.scenario.configspace.get_default_configuration()
            )
            trial_info = SimpleNamespace(
                config=self.scenario.configspace.get_default_configuration()
            )
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

    def fake_run_replay_server_for_solver_configuration(**kwargs):
        return {
            "command": [
                str(kwargs["replay_binary"]),
                "-snapshot_collection",
                str(kwargs["snapshot_collection_path"]),
            ],
            "returncode": 0,
            "replay_result_path": str(kwargs["replay_result_path"]),
            "schema_errors": [],
            "subprocess_wall_time_sec": 0.2,
            "failure_reason": None,
            "replay_result": {
                **valid_replay_result(),
                "objective_time_sec_median": 0.6,
                "total_wall_time_sec": 0.7,
                "matrix_load_time_sec": 0.01,
                "solver_setup_time_sec": 0.02,
                "solve_time_sec_mean": 0.6,
                "solve_time_sec_median": 0.6,
                "iterations_total": 5,
                "snapshots": 1,
                "steps": [],
                "converged": True,
                "reason": "KSP_CONVERGED_RTOL",
                "reason_code": 2,
                "rss_request_peak_sample_mb_sum": 128.0,
                "rss_peak_sample_mb_max_rank": 64.0,
                "rss_peak_sample_mb_max_node": 128.0,
                "final_true_relative_residual_mean": 1.0e-9,
                "final_true_residual_norm_mean": 1.0e-10,
            },
        }

    monkeypatch.setattr(
        "ksptune.tuning_runs.run_replay_server_for_solver_configuration",
        fake_run_replay_server_for_solver_configuration,
    )

    events = []
    result = resume_tuning(
        output_directory,
        overrides={
            "trials": 3,
            "workers": 2,
        },
        progress_callback=events.append,
    )

    assert result["status"] == "completed"
    assert captured["overwrite"] is False
    assert captured["scenario"].name == "oldsmac"
    assert captured["scenario"].n_trials == 3
    assert captured["scenario"].n_workers == 2
    records = [
        json.loads(line)
        for line in (output_directory / "solver_configuration_trials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["trial_number"] for record in records] == [1, 2]
    assert records[0]["smac_configuration_tag"] == "old001"
    assert records[1]["objective_value"] == 0.6
    assert result["best_cost"] == 0.5
    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["settings"]["soft_timeout_sec"] == 90.0
    assert tuning_run["settings"]["hard_timeout_sec"] == 135.0
    assert tuning_run["settings"]["replay_startup_timeout_sec"] == 1200.0
    assert tuning_run["settings"]["max_true_relative_residual"] == 2.0e-4
    assert tuning_run["settings"]["max_true_residual_norm"] == 3.0e-4
    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["resume"] is True
    assert run_started["completed_trials"] == 1
    assert run_started["remaining_trials"] == 2
    scenario_json = json.loads(
        (output_directory / "smac" / "oldsmac" / "1" / "scenario.json").read_text(encoding="utf-8")
    )
    assert scenario_json["n_trials"] == 3
    assert scenario_json["n_workers"] == 2


def test_resume_tuning_refreshes_outputs_when_target_is_already_reached(
    tmp_path: Path,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_minimal_snapshot(dump_directory)
    replay_binary = write_executable_file(tmp_path / "ksptune-petsc-replay")
    output_directory = tmp_path / "run"
    write_resume_run(
        output_directory,
        dump_directory,
        replay_binary=replay_binary,
        trials=1,
        soft_timeout_sec=90.0,
        hard_timeout_sec=135.0,
    )

    result = resume_tuning(
        output_directory,
        overrides={
            "trials": 1,
            "soft_timeout_sec": 80.0,
            "hard_timeout_sec": 120.0,
            "replay_startup_timeout_sec": 1800.0,
            "max_true_relative_residual": 1.0e-3,
            "max_true_residual_norm": 2.0e-3,
        },
    )

    assert result["status"] == "completed"
    assert result["best_cost"] == 0.5
    records = load_trial_records(output_directory / "solver_configuration_trials.jsonl")
    assert len(records) == 1
    summary = yaml.safe_load((output_directory / "tuning_summary.yaml").read_text(encoding="utf-8"))
    assert summary["trial_count"] == 1
    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["settings"]["soft_timeout_sec"] == 80.0
    assert tuning_run["settings"]["hard_timeout_sec"] == 120.0
    assert tuning_run["settings"]["replay_startup_timeout_sec"] == 1800.0
    assert tuning_run["settings"]["max_true_relative_residual"] == 1.0e-3
    assert tuning_run["settings"]["max_true_residual_norm"] == 2.0e-3


def test_tuning_configures_seeded_first_initial_design(
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
    replay_binary = write_executable_file(tmp_path / "dummy-replay")
    run_tuning(
        TuningSettings(
            snapshot_directory=snapshot_directory,
            output_directory=output_directory,
            replay_binary=replay_binary,
            trials=12,
            workers=4,
        ),
        parameter_search_space=load_parameter_search_space("petsc.boomeramg_basic"),
        progress_callback=events.append,
    )

    assert captured["scenario"].use_default_config is False
    assert isinstance(captured["initial_design"], KSPTuneOrderedInitialDesign)
    assert captured["initial_design_kwargs"]["n_configs"] == 0
    assert captured["initial_design_kwargs"]["max_ratio"] == 1.0
    assert captured["initial_design_kwargs"]["additional_configs"] == []
    assert len(captured["initial_design"].leading_configurations) == 12

    tuning_run = yaml.safe_load((output_directory / "tuning_run.yaml").read_text(encoding="utf-8"))
    assert tuning_run["metadata"]["initial_design_order"] == "seeded-first"
    assert tuning_run["metadata"]["use_default_solver_configuration"] is False
    assert tuning_run["metadata"]["tunable_parameter_count"] == 8
    assert tuning_run["metadata"]["sobol_initial_design_configurations"] == 0
    assert tuning_run["metadata"]["startup_initial_solver_configuration_count"] == 4
    assert tuning_run["metadata"]["startup_initial_solver_configurations"] == [
        "boomeramg_exti_threshold_0p05",
        "boomeramg_ff_agg_1_paths_2",
        "boomeramg_hmis_exti_pmax_3",
        "boomeramg_exti_pmax_8",
    ]
    assert tuning_run["metadata"]["additionalinitial_solver_configuration_count"] == 12
    assert tuning_run["metadata"]["additionalinitial_solver_configurations"][0] == "boomeramg_exti_threshold_0p05"
    assert "boomeramg_pmis_exticc_agg_0" in tuning_run["metadata"]["additionalinitial_solver_configurations"]

    run_started = next(event for event in events if event["event"] == "run_started")
    assert run_started["sobol_initial_design_configurations"] == 0
    assert run_started["startup_initial_solver_configuration_count"] == 4
    assert run_started["additionalinitial_solver_configuration_count"] == 12


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
