from __future__ import annotations

import pytest

from ksptune.parameter_search_spaces import (
    build_configspace_from_parameter_search_space,
    list_builtin_parameter_search_spaces,
    load_parameter_search_space,
    parameter_search_space_to_yaml,
    validate_parameter_search_space,
    write_configspace_yaml,
)
from ksptune.petsc_options import render_petsc_options_from_solver_configuration
from ksptune.solver_configurations import default_solver_configuration_from_parameter_search_space


def test_builtin_parameter_search_spaces_are_loadable() -> None:
    names = list_builtin_parameter_search_spaces()
    assert names == [
        "petsc.boomeramg_basic",
        "petsc.boomeramg_extended",
        "petsc.boomeramg_extended_mono",
        "petsc.direct_like_mono",
        "petsc.direct_mono",
        "petsc.gamg_basic",
        "petsc.gamg_extended",
        "petsc.gamg_extended_mono",
        "petsc.hypre_ilu_mono",
        "petsc.parasails_mono",
        "petsc.pilut_mono",
        "petsc.solver_comparison_mono",
        "petsc.spai_mono",
    ]

    for name in names:
        parameter_search_space = load_parameter_search_space(name)
        assert parameter_search_space.name == name
        assert validate_parameter_search_space(parameter_search_space) == []

    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")
    configuration_space = build_configspace_from_parameter_search_space(parameter_search_space)

    assert len(configuration_space) > 0
    assert "hyperparameters:" in parameter_search_space_to_yaml(parameter_search_space)


def test_petsc_boomeramg_basic_defaults_match_tuned_baseline() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    expected_values = {
        "ksp_type": "gmres",
        "ksp_pc_side": "left",
        "ksp_norm_type": "preconditioned",
        "ksp_atol": 1.0e-8,
        "ksp_rtol": 1.0e-8,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_coarsen_type": "PMIS",
        "pc_hypre_boomeramg_interp_type": "ext+i-mm",
        "pc_hypre_boomeramg_relax_type_all": "l1-Gauss-Seidel",
        "pc_hypre_boomeramg_P_max": 4,
        "pc_hypre_boomeramg_agg_nl": 1,
        "pc_hypre_boomeramg_agg_num_paths": 1,
        "pc_hypre_boomeramg_truncfactor": 0.0,
        "pc_hypre_boomeramg_strong_threshold": 0.1,
    }
    for name, value in expected_values.items():
        assert solver_configuration[name] == value

    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    assert "-pc_type" in options
    assert "hypre" in options
    assert "-ksp_norm_type" in options
    assert "preconditioned" in options
    assert "-ksp_rtol" in options
    assert "1e-08" in options
    assert "-ksp_atol" in options
    assert "-pc_hypre_boomeramg_P_max" in options
    assert "4" in options
    assert "-pc_hypre_boomeramg_interp_type" in options
    assert "ext+i-mm" in options
    assert "-pc_hypre_boomeramg_agg_num_paths" in options
    assert "-pc_hypre_boomeramg_cycle_type" not in options
    assert "-pc_hypre_boomeramg_measure_type" not in options
    assert "-pc_hypre_boomeramg_nodal_coarsen" not in options
    assert "-pc_hypre_boomeramg_truncfactor" not in options
    assert not parameter_search_space["ksp_type"].legal_value("cg")


def test_petsc_gamg_basic_renders_fixed_tolerances() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )

    assert solver_configuration["ksp_type"] == "gmres"
    assert solver_configuration["ksp_pc_side"] == "left"
    assert solver_configuration["ksp_norm_type"] == "preconditioned"
    assert solver_configuration["ksp_atol"] == 1.0e-8
    assert solver_configuration["ksp_rtol"] == 1.0e-7
    assert solver_configuration["pc_gamg_threshold_scale"] == 1.0
    assert solver_configuration["pc_gamg_low_memory_threshold_filter"] is False
    assert "-ksp_atol" in options
    assert "-ksp_rtol" in options
    assert "-pc_gamg_threshold_scale" in options
    assert "-pc_gamg_low_memory_threshold_filter" in options
    assert "false" in options
    assert "1e-07" in options


def test_petsc_parasails_mono_renders_regular_hypre_options() -> None:
    parameter_search_space = load_parameter_search_space("petsc.parasails_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert solver_configuration["ksp_type"] == "cg"
    assert solver_configuration["pc_type"] == "hypre"
    assert solver_configuration["pc_hypre_type"] == "parasails"
    assert parameter_search_space["pc_hypre_parasails_reuse"].value is False
    assert not parameter_search_space["pc_hypre_parasails_reuse"].legal_value(True)
    assert "-pc_hypre_parasails_nlevels 1" in options_text
    assert "-pc_hypre_parasails_thresh 0.4" in options_text
    assert "-pc_hypre_parasails_filter 0.1" in options_text
    assert "-pc_hypre_parasails_logging false" in options_text
    assert "-pc_hypre_parasails_reuse" not in options_text
    assert "-pc_hypre_parasails_sym" not in options_text


def test_petsc_parasails_mono_renders_auto_drop_threshold_and_zero_filter() -> None:
    parameter_search_space = load_parameter_search_space("petsc.parasails_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    solver_configuration.update(
        {
            "pc_hypre_parasails_thresh_mode": "auto_drop",
            "pc_hypre_parasails_thresh_auto": -0.9,
            "pc_hypre_parasails_filter_mode": "zero",
            "pc_hypre_parasails_filter_zero": 0.0,
        }
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert "-pc_hypre_parasails_thresh -0.9" in options_text
    assert "-pc_hypre_parasails_filter 0.0" in options_text
    assert "pc_hypre_parasails_thresh_auto" not in options_text


def test_petsc_pilut_mono_is_wide_hypre_pilut_space() -> None:
    parameter_search_space = load_parameter_search_space("petsc.pilut_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert solver_configuration["ksp_type"] == "gmres"
    assert solver_configuration["pc_type"] == "hypre"
    assert solver_configuration["pc_hypre_type"] == "pilut"
    assert parameter_search_space["ksp_type"].value == "gmres"
    assert parameter_search_space["ksp_gmres_restart"].value == 30

    factorrowsize = parameter_search_space["pc_hypre_pilut_factorrowsize"]
    drop_tolerance = parameter_search_space["pc_hypre_pilut_tol"]
    maxiter = parameter_search_space["pc_hypre_pilut_maxiter"]
    assert factorrowsize.lower == 20
    assert factorrowsize.upper == 8000
    assert drop_tolerance.lower == pytest.approx(1.0e-8)
    assert drop_tolerance.upper == pytest.approx(1.0)
    assert drop_tolerance.log
    assert maxiter.lower == 1
    assert maxiter.upper == 50
    assert maxiter.default_value == 1

    assert "-ksp_type gmres" in options_text
    assert "-ksp_pc_side left" in options_text
    assert "-pc_type hypre" in options_text
    assert "-pc_hypre_type pilut" in options_text
    assert "-pc_hypre_pilut_factorrowsize 150" in options_text
    assert "-pc_hypre_pilut_tol 0.0001" in options_text
    assert "-pc_hypre_pilut_maxiter 1" in options_text
    assert "-ksp_gmres_restart" not in options_text

    initial_configurations = getattr(
        parameter_search_space,
        "_ksptune_initial_solver_configurations",
    )
    labels = {configuration["label"] for configuration in initial_configurations}
    assert "pilut_factor_rows_5780" in labels
    assert "pilut_factor_rows_477" in labels


def test_petsc_hypre_ilu_mono_uses_petsc_hypre_ilu_options() -> None:
    parameter_search_space = load_parameter_search_space("petsc.hypre_ilu_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert solver_configuration["ksp_type"] == "gmres"
    assert solver_configuration["pc_type"] == "hypre"
    assert solver_configuration["pc_hypre_type"] == "ilu"
    assert parameter_search_space["pc_hypre_ilu_type"].legal_value("GMRES-ILUT")
    assert parameter_search_space["pc_hypre_ilu_type"].legal_value("RAS-ILUk")
    assert parameter_search_space["pc_hypre_ilu_level"].lower == 0
    assert parameter_search_space["pc_hypre_ilu_level"].upper == 5
    assert parameter_search_space["pc_hypre_ilu_max_nnz_per_row"].lower == 10
    assert parameter_search_space["pc_hypre_ilu_max_nnz_per_row"].upper == 2000
    assert parameter_search_space["pc_hypre_ilu_drop_threshold"].log
    assert parameter_search_space["pc_hypre_ilu_tri_solve"].default_value is True
    assert parameter_search_space["pc_hypre_ilu_lower_jacobi_iters"].lower == 1
    assert parameter_search_space["pc_hypre_ilu_lower_jacobi_iters"].upper == 5
    assert parameter_search_space["pc_hypre_ilu_upper_jacobi_iters"].lower == 1
    assert parameter_search_space["pc_hypre_ilu_upper_jacobi_iters"].upper == 5

    for expected_fragment in [
        "-ksp_type gmres",
        "-pc_type hypre",
        "-pc_hypre_type ilu",
        "-pc_hypre_ilu_type Block-Jacobi-ILUT",
        "-pc_hypre_ilu_level 1",
        "-pc_hypre_ilu_max_nnz_per_row 657",
        "-pc_hypre_ilu_tol 0.0",
        "-pc_hypre_ilu_maxiter 1",
        "-pc_hypre_ilu_drop_threshold 3.593594e-07",
        "-pc_hypre_ilu_tri_solve true",
        "-pc_hypre_ilu_local_reordering false",
    ]:
        assert expected_fragment in options_text

    assert "-pc_hypre_ilu_lower_jacobi_iters" not in options_text
    assert "-pc_hypre_ilu_upper_jacobi_iters" not in options_text
    assert "-pc_hypre_ilu_logging" not in options_text
    assert "-pc_hypre_ilu_print_level" not in options_text


def test_petsc_solver_comparison_mono_contains_fixed_benchmark_candidates() -> None:
    from ksptune.tuning_runs import initial_solver_configuration_records_from_parameter_search_space

    parameter_search_space = load_parameter_search_space("petsc.solver_comparison_mono")
    records = initial_solver_configuration_records_from_parameter_search_space(
        parameter_search_space
    )

    assert [record["label"] for record in records] == [
        "spai_cg",
        "gamg_gmres",
        "parasails_gmres",
        "pilut_gmres",
        "boomeramg_gmres",
        "bjacobi_ilu_cg_baseline",
    ]

    rendered_by_label = {
        record["label"]: " ".join(
            render_petsc_options_from_solver_configuration(
                parameter_search_space,
                record["solver_configuration"],
            )
        )
        for record in records
    }

    assert "-pc_type spai" in rendered_by_label["spai_cg"]
    assert "-pc_spai_epsilon 0.0300007666235" in rendered_by_label["spai_cg"]
    assert "-pc_spai_max 68672" in rendered_by_label["spai_cg"]
    assert "-pc_spai_maxnew 50" in rendered_by_label["spai_cg"]
    assert "-pc_spai_nbsteps 8" in rendered_by_label["spai_cg"]
    assert "-pc_type gamg" in rendered_by_label["gamg_gmres"]
    assert "-ksp_max_it 1000" in rendered_by_label["gamg_gmres"]
    assert "-ksp_gmres_restart 30" in rendered_by_label["gamg_gmres"]
    assert "-pc_gamg_threshold 0.1141898203948" in rendered_by_label["gamg_gmres"]
    assert "-pc_gamg_threshold_scale 0.2534279319386" in rendered_by_label["gamg_gmres"]
    assert "-pc_gamg_process_eq_limit 349" in rendered_by_label["gamg_gmres"]
    assert "-pc_mg_type kaskade" in rendered_by_label["gamg_gmres"]
    assert "-pc_gamg_esteig_ksp_type" not in rendered_by_label["gamg_gmres"]
    assert "-pc_hypre_type parasails" in rendered_by_label["parasails_gmres"]
    assert "-ksp_gmres_restart 30" in rendered_by_label["parasails_gmres"]
    assert "-pc_hypre_parasails_loadbal 0.1170590021484" in rendered_by_label[
        "parasails_gmres"
    ]
    assert "-pc_hypre_parasails_thresh 0.0028680776115" in rendered_by_label[
        "parasails_gmres"
    ]
    assert "-pc_hypre_parasails_filter 0.0" in rendered_by_label[
        "parasails_gmres"
    ]
    assert "-pc_hypre_parasails_reuse" not in rendered_by_label["parasails_gmres"]
    assert "-pc_hypre_type pilut" in rendered_by_label["pilut_gmres"]
    assert "-pc_hypre_pilut_factorrowsize 1837" in rendered_by_label["pilut_gmres"]
    assert "-pc_hypre_pilut_maxiter 17" in rendered_by_label["pilut_gmres"]
    assert "-pc_hypre_pilut_tol 9.8176496e-06" in rendered_by_label["pilut_gmres"]
    assert "-pc_hypre_type boomeramg" in rendered_by_label["boomeramg_gmres"]
    assert "-pc_hypre_boomeramg_nodal_coarsen" not in rendered_by_label[
        "boomeramg_gmres"
    ]
    assert "-ksp_type cg" in rendered_by_label["bjacobi_ilu_cg_baseline"]
    assert "-pc_type bjacobi" in rendered_by_label["bjacobi_ilu_cg_baseline"]
    assert "-sub_pc_type ilu" in rendered_by_label["bjacobi_ilu_cg_baseline"]


def test_petsc_gamg_basic_uses_conservative_search_bounds() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_basic")

    agg_nsmooths = parameter_search_space["pc_gamg_agg_nsmooths"]
    aggressive_coarsening = parameter_search_space["pc_gamg_aggressive_coarsening"]
    threshold = parameter_search_space["pc_gamg_threshold"]
    threshold_scale = parameter_search_space["pc_gamg_threshold_scale"]
    aggressive_mis_k = parameter_search_space["pc_gamg_aggressive_mis_k"]
    low_memory_filter = parameter_search_space["pc_gamg_low_memory_threshold_filter"]
    mat_coarsen_type = parameter_search_space["pc_gamg_mat_coarsen_type"]
    mat_coarsen_misk_distance = parameter_search_space["pc_gamg_mat_coarsen_misk_distance"]

    assert agg_nsmooths.lower == 1
    assert agg_nsmooths.upper == 2
    assert aggressive_coarsening.lower == 2
    assert aggressive_coarsening.upper == 6
    assert threshold.lower == pytest.approx(0.0)
    assert threshold.upper == pytest.approx(0.02)
    assert threshold_scale.lower == pytest.approx(0.5)
    assert threshold_scale.upper == pytest.approx(1.25)
    assert threshold_scale.default_value == pytest.approx(1.0)
    assert aggressive_mis_k.upper == 4
    assert low_memory_filter.legal_value(False)
    assert low_memory_filter.legal_value(True)
    assert low_memory_filter.default_value is False
    assert mat_coarsen_type.default_value == "mis"
    assert mat_coarsen_misk_distance.value == 1


def test_petsc_gamg_basic_aggressive_mis_k_tracks_expanded_coarsening() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    solver_configuration.update(
        {
            "pc_gamg_aggressive_coarsening": 6,
            "pc_gamg_aggressive_square_graph": False,
            "pc_gamg_aggressive_mis_k": 3,
        }
    )

    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )

    assert "-pc_gamg_aggressive_coarsening" in options
    assert "6" in options
    assert "-pc_gamg_aggressive_mis_k" in options
    assert "3" in options


def test_petsc_gamg_basic_renders_coarse_eq_limit_only_for_parallel_coarse_solver() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    solver_configuration.update(
        {
            "pc_gamg_parallel_coarse_grid_solver": False,
            "pc_gamg_coarse_eq_limit": 3000,
        }
    )

    inactive_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    assert "-pc_gamg_coarse_eq_limit" not in inactive_options

    solver_configuration.update(
        {
            "pc_gamg_parallel_coarse_grid_solver": True,
            "pc_gamg_coarse_eq_limit": 3000,
        }
    )
    active_options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert "-pc_gamg_coarse_eq_limit 3000" in active_options_text


def test_petsc_gamg_extended_contains_esteig_inventory() -> None:
    parameter_search_space = load_parameter_search_space("petsc.gamg_extended")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert parameter_search_space["pc_gamg_esteig_ksp_type"].legal_value("cg")
    assert parameter_search_space["mg_levels_esteig_ksp_type"].legal_value("cg")
    assert parameter_search_space["mg_levels_esteig_ksp_max_it"].legal_value(10)
    assert parameter_search_space["mg_levels_esteig_ksp_rtol"].legal_value(1.0e-2)
    assert parameter_search_space["pc_gamg_prolongator_filter"].default_value is False
    assert "-pc_gamg_esteig_ksp_type cg" in options_text
    assert "-mg_levels_esteig_ksp_type cg" in options_text
    assert "-mg_levels_esteig_ksp_max_it 10" in options_text
    assert "-pc_gamg_prolongator_filter" not in options_text


def test_petsc_gamg_extended_mono_is_broad_scalar_space() -> None:
    extended_space = load_parameter_search_space("petsc.gamg_extended")
    mono_space = load_parameter_search_space("petsc.gamg_extended_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        mono_space
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            mono_space,
            solver_configuration,
        )
    )

    assert set(mono_space.keys()) == set(extended_space.keys())
    assert solver_configuration["ksp_type"] == "gmres"
    assert solver_configuration["pc_type"] == "gamg"
    assert solver_configuration["ksp_atol"] == 1.0e-8
    assert solver_configuration["ksp_rtol"] == 1.0e-8
    assert mono_space["pc_gamg_threshold"].upper == pytest.approx(0.3)
    assert mono_space["pc_gamg_threshold_scale"].lower == pytest.approx(0.05)
    assert mono_space["pc_gamg_mat_coarsen_misk_distance"].upper == 4
    assert mono_space["pc_gamg_process_eq_limit"].upper == 1000
    assert mono_space["mg_levels_pc_type"].legal_value("asm")
    assert mono_space["mg_coarse_pc_type"].legal_value("bjacobi")
    assert mono_space["mg_coarse_pc_type"].legal_value("lu")
    assert mono_space["mg_coarse_pc_type"].legal_value("jacobi")
    assert not mono_space["mg_coarse_pc_type"].legal_value("ilu")
    assert mono_space["pc_gamg_low_memory_threshold_filter"].value is False
    assert mono_space["mg_levels_ksp_chebyshev_esteig_noisy"].value is True

    excluded_fragments = ("nullspace", "near_nullspace", "near_null_space", "block_size")
    assert not any(
        any(fragment in parameter_name for fragment in excluded_fragments)
        for parameter_name in mono_space.keys()
    )
    for option_fragment in [
        "-replay_nullspace",
        "-pc_gamg_block_size",
        "-pc_gamg_near_nullspace",
        "-pc_gamg_near_null_space",
    ]:
        assert option_fragment not in options_text


def test_petsc_spai_mono_is_wide_solve_only_space() -> None:
    parameter_search_space = load_parameter_search_space("petsc.spai_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert solver_configuration["ksp_type"] == "cg"
    assert solver_configuration["pc_type"] == "spai"
    assert solver_configuration["ksp_pc_side"] == "left"
    assert solver_configuration["ksp_norm_type"] == "preconditioned"
    assert solver_configuration["ksp_atol"] == 1.0e-8
    assert solver_configuration["ksp_rtol"] == 1.0e-8

    epsilon = parameter_search_space["pc_spai_epsilon"]
    nbsteps = parameter_search_space["pc_spai_nbsteps"]
    maxnew = parameter_search_space["pc_spai_maxnew"]
    max_buffer = parameter_search_space["pc_spai_max"]

    assert epsilon.lower == pytest.approx(0.005)
    assert epsilon.upper == pytest.approx(0.8)
    assert epsilon.log
    assert epsilon.legal_value(0.0372778643732)
    assert nbsteps.lower == 1
    assert nbsteps.upper == 100
    assert maxnew.lower == 1
    assert maxnew.upper == 64
    assert max_buffer.lower == 5000
    assert max_buffer.upper == 100000
    assert parameter_search_space["pc_spai_block_size"].legal_value(0)
    assert parameter_search_space["pc_spai_block_size"].legal_value(2)
    assert parameter_search_space["pc_spai_cache_size"].value == 5
    assert parameter_search_space["pc_spai_sp"].legal_value(0)

    for expected_fragment in [
        "-ksp_type cg",
        "-ksp_pc_side left",
        "-pc_type spai",
        "-pc_spai_epsilon 0.0372778643732",
        "-pc_spai_nbsteps 10",
        "-pc_spai_maxnew 2",
        "-pc_spai_max 10000",
        "-pc_spai_block_size 1",
        "-pc_spai_cache_size 5",
        "-pc_spai_sp 1",
    ]:
        assert expected_fragment in options_text

    initial_configurations = getattr(
        parameter_search_space,
        "_ksptune_initial_solver_configurations",
    )
    labels = {configuration["label"] for configuration in initial_configurations}
    assert "spai_steps_10" in labels
    assert "spai_steps_30" in labels


def test_petsc_boomeramg_basic_contains_initial_configuration_option_ranges() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")

    categorical_boomeramg_option_values = {
        "pc_hypre_boomeramg_coarsen_type": ["PMIS", "HMIS"],
        "pc_hypre_boomeramg_interp_type": [
            "ext+i",
            "ext+i-cc",
            "ext+i-mm",
            "ext",
            "ext+e-mm",
            "FF",
        ],
        "pc_hypre_boomeramg_relax_type_all": [
            "l1-Gauss-Seidel",
            "SOR/Jacobi",
            "l1scaled-Jacobi",
        ],
        "pc_hypre_boomeramg_P_max": [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16],
        "pc_hypre_boomeramg_agg_nl": [0, 1, 2],
        "pc_hypre_boomeramg_agg_num_paths": [1, 2, 3],
    }
    for parameter_name, values in categorical_boomeramg_option_values.items():
        for value in values:
            assert parameter_search_space[parameter_name].legal_value(value)

    assert not parameter_search_space["pc_hypre_boomeramg_coarsen_type"].legal_value("Falgout")
    assert not parameter_search_space["pc_hypre_boomeramg_agg_nl"].legal_value(4)
    assert not parameter_search_space["pc_hypre_boomeramg_agg_num_paths"].legal_value(0)
    for removed_relax_type in ["symmetric-SOR/Jacobi", "Jacobi", "FCF-Jacobi"]:
        assert not parameter_search_space["pc_hypre_boomeramg_relax_type_all"].legal_value(
            removed_relax_type
        )
    for removed_parameter in [
        "pc_hypre_boomeramg_cycle_type",
        "pc_hypre_boomeramg_measure_type",
        "pc_hypre_boomeramg_nodal_coarsen",
        "pc_hypre_boomeramg_relax_type_coarse",
        "pc_hypre_boomeramg_relax_weight_all",
    ]:
        assert removed_parameter not in parameter_search_space

    strong_threshold = parameter_search_space["pc_hypre_boomeramg_strong_threshold"]
    truncfactor_values = list(parameter_search_space["pc_hypre_boomeramg_truncfactor"].sequence)

    assert strong_threshold.lower == pytest.approx(0.0001)
    assert strong_threshold.upper == pytest.approx(0.7)
    assert strong_threshold.default_value == pytest.approx(0.1)
    assert strong_threshold.legal_value(0.0388801205714)
    assert strong_threshold.legal_value(0.0001)
    assert not strong_threshold.legal_value(0.0)
    assert truncfactor_values == sorted(
        [
            *[round(index * 0.025, 3) for index in range(0, 17)],
                0.0021247466225,
                0.0024114874969,
                0.0044798272768,
                0.015826322068,
            ]
        )


def test_petsc_boomeramg_basic_agg_num_paths_is_conditional() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    solver_configuration.update(
        {
            "pc_hypre_boomeramg_agg_nl": 0,
            "pc_hypre_boomeramg_agg_num_paths": 2,
        }
    )
    inactive_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    assert "-pc_hypre_boomeramg_agg_num_paths" not in inactive_options

    solver_configuration.update(
        {
            "pc_hypre_boomeramg_agg_nl": 1,
            "pc_hypre_boomeramg_agg_num_paths": 2,
        }
    )
    active_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    assert "-pc_hypre_boomeramg_agg_num_paths" in active_options
    assert "2" in active_options


def test_petsc_boomeramg_extended_is_basic_superset_with_fixed_v_cycle() -> None:
    basic_space = load_parameter_search_space("petsc.boomeramg_basic")
    extended_space = load_parameter_search_space("petsc.boomeramg_extended")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        extended_space
    )

    assert set(basic_space.keys()) <= set(extended_space.keys())
    assert solver_configuration["ksp_type"] == "gmres"
    assert solver_configuration["ksp_pc_side"] == "left"
    assert solver_configuration["ksp_norm_type"] == "preconditioned"
    assert solver_configuration["ksp_atol"] == 1.0e-8
    assert solver_configuration["ksp_rtol"] == 1.0e-8
    assert solver_configuration["pc_hypre_boomeramg_cycle_type"] == "V"
    assert solver_configuration["pc_hypre_boomeramg_max_iter"] == 1
    assert solver_configuration["pc_hypre_boomeramg_tol"] == 0.0

    options = render_petsc_options_from_solver_configuration(
        extended_space,
        solver_configuration,
    )
    options_text = " ".join(options)

    assert "-pc_hypre_boomeramg_cycle_type V" in options_text
    assert "-pc_hypre_boomeramg_max_iter 1" in options_text
    assert "-pc_hypre_boomeramg_tol 0.0" in options_text
    assert "-pc_hypre_boomeramg_agg_num_paths" not in options
    assert "-pc_hypre_boomeramg_nodal_coarsen" not in options
    assert "-pc_hypre_boomeramg_no_CF" not in options
    assert "-pc_hypre_boomeramg_keeptranspose" not in options


def test_petsc_boomeramg_extended_ranges_and_categories() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_extended")

    assert parameter_search_space["pc_hypre_boomeramg_strong_threshold"].lower == 0.0
    assert parameter_search_space["pc_hypre_boomeramg_strong_threshold"].upper == 0.9
    assert parameter_search_space["pc_hypre_boomeramg_truncfactor"].lower == 0.0
    assert parameter_search_space["pc_hypre_boomeramg_truncfactor"].upper == 0.5
    assert parameter_search_space["pc_hypre_boomeramg_max_row_sum"].lower == 0.0
    assert parameter_search_space["pc_hypre_boomeramg_max_row_sum"].upper == 1.0
    assert parameter_search_space["pc_hypre_boomeramg_P_max"].lower == 0
    assert parameter_search_space["pc_hypre_boomeramg_P_max"].upper == 16
    assert parameter_search_space["pc_hypre_boomeramg_agg_nl"].upper == 4
    assert parameter_search_space["pc_hypre_boomeramg_agg_num_paths"].upper == 4
    assert not parameter_search_space["pc_hypre_boomeramg_coarsen_type"].legal_value(
        "CLJP"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_coarsen_type"].legal_value(
        "Falgout"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_coarsen_type"].legal_value(
        "Ruge-Stueben"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_coarsen_type"].legal_value(
        "modifiedRuge-Stueben"
    )
    assert parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "ext+i-mm"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "classical"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "direct"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "ad-wts"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "multipass"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "multipass-wts"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "standard"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "standard-wts"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "block"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_interp_type"].legal_value(
        "block-wtd"
    )
    assert parameter_search_space["pc_hypre_boomeramg_relax_type_all"].legal_value(
        "Chebyshev"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_relax_type_all"].legal_value(
        "Jacobi"
    )
    assert not parameter_search_space["pc_hypre_boomeramg_relax_type_all"].legal_value(
        "FCF-Jacobi"
    )


def test_petsc_boomeramg_extended_excludes_local_patch_parameters() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_extended")

    excluded_parameters = {
        "pc_hypre_boomeramg_nongalerkin_tol",
        "pc_hypre_boomeramg_agg_interp_type",
        "pc_hypre_boomeramg_agg_truncfactor",
        "pc_hypre_boomeramg_agg_P_max",
        "pc_hypre_boomeramg_agg_P12_truncfactor",
        "pc_hypre_boomeramg_agg_P12_max",
        "pc_hypre_boomeramg_sabs",
        "pc_hypre_boomeramg_vec_interp_qtrunc",
        "pc_hypre_boomeramg_view_hierarchy",
        "pc_hypre_boomeramg_interp_refine",
        "pc_hypre_boomeramg_no_CF",
        "pc_hypre_boomeramg_nodal_coarsen",
        "pc_hypre_boomeramg_nodal_coarsen_diag",
        "pc_hypre_boomeramg_vec_interp_qmax",
        "pc_hypre_boomeramg_vec_interp_smooth",
        "pc_hypre_boomeramg_vec_interp_variant",
    }
    assert excluded_parameters.isdisjoint(parameter_search_space.keys())


def test_petsc_boomeramg_extended_mono_excludes_system_amg_parameters() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_extended_mono")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    excluded_parameters = {
        "pc_hypre_boomeramg_nodal_coarsen",
        "pc_hypre_boomeramg_nodal_coarsen_diag",
        "pc_hypre_boomeramg_nodal_relaxation",
        "pc_hypre_boomeramg_vec_interp_qmax",
        "pc_hypre_boomeramg_vec_interp_smooth",
        "pc_hypre_boomeramg_vec_interp_variant",
        "pc_hypre_boomeramg_numfunctions",
        "pc_hypre_boomeramg_outer_relax_weight_all",
        "pc_hypre_boomeramg_relax_weight_all",
        "pc_hypre_boomeramg_restriction_type",
        "pc_hypre_boomeramg_use_smooth_type",
    }
    assert excluded_parameters.isdisjoint(parameter_search_space.keys())
    for option_name in excluded_parameters:
        assert f"-{option_name}" not in options_text
    assert (
        parameter_search_space["pc_hypre_boomeramg_agg_nl"].upper
        <= parameter_search_space["pc_hypre_boomeramg_max_levels"].lower
    )
    assert parameter_search_space["pc_hypre_boomeramg_max_row_sum"].upper == pytest.approx(
        0.85
    )
    assert not parameter_search_space["pc_hypre_boomeramg_relax_type_all"].legal_value(
        "l1scaled-Jacobi"
    )
    assert parameter_search_space["pc_hypre_boomeramg_relax_type_down"].legal_value(
        "l1scaled-Jacobi"
    )


def test_petsc_boomeramg_extended_conditionals_render_only_agg_num_paths_when_active() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_extended")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    default_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    assert "-pc_hypre_boomeramg_agg_num_paths" not in default_options

    solver_configuration.update(
        {
            "pc_hypre_boomeramg_agg_nl": 2,
            "pc_hypre_boomeramg_agg_num_paths": 4,
        }
    )
    active_options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert "-pc_hypre_boomeramg_agg_num_paths 4" in active_options_text


def test_petsc_options_render_readable_float_values() -> None:
    parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    solver_configuration.update(
        {
            "pc_hypre_boomeramg_strong_threshold": 0.1,
            "pc_hypre_boomeramg_truncfactor": 0.05,
        }
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert "-pc_hypre_boomeramg_strong_threshold 0.1" in options_text
    assert "-pc_hypre_boomeramg_truncfactor 0.05" in options_text
    assert "0.10000000000000001" not in options_text
    assert "0.050000000000000003" not in options_text


def test_inactive_parameters_are_not_rendered() -> None:
    from ConfigSpace import Categorical, ConfigurationSpace, Constant
    from ConfigSpace.conditions import EqualsCondition

    parameter_search_space = ConfigurationSpace(name="conditional", seed=1)
    ksp_type = Categorical("ksp_type", ["cg", "fgmres"], default="cg")
    restart = Constant(
        "ksp_gmres_restart",
        80,
        meta={"petsc_option": "-ksp_gmres_restart"},
    )
    pc_type = Constant("pc_type", "hypre", meta={"petsc_option": "-pc_type"})
    parameter_search_space.add([ksp_type, restart, pc_type])
    parameter_search_space.add(EqualsCondition(restart, ksp_type, "fgmres"))

    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        {"ksp_type": "cg", "ksp_gmres_restart": 80, "pc_type": "hypre"},
    )

    assert "-pc_type" in options
    assert "hypre" in options
    assert "-ksp_gmres_restart" not in options


def test_invalid_cg_right_parameter_search_space_is_rejected() -> None:
    from ConfigSpace import Categorical, ConfigurationSpace, Constant

    parameter_search_space = ConfigurationSpace(name="invalid.cg-right", seed=1)
    ksp_type = Categorical(
        "ksp_type",
        ["cg", "fgmres"],
        default="cg",
        meta={"petsc_option": "-ksp_type"},
    )
    ksp_pc_side = Categorical(
        "ksp_pc_side",
        ["left", "right"],
        default="right",
        meta={"petsc_option": "-ksp_pc_side"},
    )
    pc_type = Constant("pc_type", "jacobi", meta={"petsc_option": "-pc_type"})
    parameter_search_space.add([ksp_type, ksp_pc_side, pc_type])

    errors = validate_parameter_search_space(parameter_search_space)

    assert errors == [
        "KSP CG does not support right preconditioning side. "
        "Make ksp_pc_side inactive for ksp_type=cg or restrict CG to left side."
    ]


def test_invalid_cg_right_python_parameter_search_space_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid_cg_right.py"
    path.write_text(
        """
from ConfigSpace import Categorical, ConfigurationSpace, Constant


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="invalid.cg-right", seed=seed)
    parameter_search_space.add(
        [
            Categorical(
                "ksp_type",
                ["cg", "fgmres"],
                default="cg",
                meta={"petsc_option": "-ksp_type"},
            ),
            Categorical(
                "ksp_pc_side",
                ["left", "right"],
                default="right",
                meta={"petsc_option": "-ksp_pc_side"},
            ),
            Constant("pc_type", "jacobi", meta={"petsc_option": "-pc_type"}),
        ]
    )
    return parameter_search_space
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="KSP CG does not support right"):
        load_parameter_search_space(path)


def test_native_configspace_yaml_can_be_loaded(tmp_path) -> None:
    source_parameter_search_space = load_parameter_search_space("petsc.boomeramg_basic")
    path = tmp_path / "configspace.yaml"
    write_configspace_yaml(source_parameter_search_space, path)

    loaded_parameter_search_space = load_parameter_search_space(path)
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        loaded_parameter_search_space
    )
    options = render_petsc_options_from_solver_configuration(
        loaded_parameter_search_space,
        solver_configuration,
    )

    assert "-pc_type" in options
    assert "hypre" in options


def test_python_parameter_search_space_file_can_be_loaded(tmp_path) -> None:
    path = tmp_path / "asm_ilu.py"
    path.write_text(
        """
from ConfigSpace import Categorical, ConfigurationSpace, Constant, Integer
from ConfigSpace.conditions import EqualsCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="asm_ilu", seed=seed)
    ksp_type = Categorical(
        "ksp_type",
        ["cg", "fgmres"],
        default="fgmres",
        meta={"petsc_option": "-ksp_type"},
    )
    pc_type = Constant("pc_type", "asm", meta={"petsc_option": "-pc_type"})
    pc_asm_overlap = Integer(
        "pc_asm_overlap",
        (0, 4),
        default=1,
        meta={"petsc_option": "-pc_asm_overlap"},
    )
    ksp_gmres_restart = Integer(
        "ksp_gmres_restart",
        (30, 200),
        default=100,
        meta={"petsc_option": "-ksp_gmres_restart"},
    )
    parameter_search_space.add([ksp_type, pc_type, pc_asm_overlap, ksp_gmres_restart])
    parameter_search_space.add(EqualsCondition(ksp_gmres_restart, ksp_type, "fgmres"))
    return parameter_search_space
""".lstrip(),
        encoding="utf-8",
    )

    parameter_search_space = load_parameter_search_space(path)
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )

    assert "-pc_type" in options
    assert "asm" in options
