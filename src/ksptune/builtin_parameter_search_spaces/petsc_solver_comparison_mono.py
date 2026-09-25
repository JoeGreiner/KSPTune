from __future__ import annotations

from ..search_space_helpers import initial_solver_configuration, petsc_meta

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Constant
from ConfigSpace.conditions import AndConjunction, EqualsCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause


def hypre_subtype_condition(child: Any, pc_type: Any, pc_hypre_type: Any, subtype: str) -> Any:
    return AndConjunction(
        EqualsCondition(child, pc_type, "hypre"),
        EqualsCondition(child, pc_hypre_type, subtype),
    )


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    space = ConfigurationSpace(name="petsc.solver_comparison_mono", seed=seed)

    ksp_type = Categorical(
        "ksp_type",
        ["cg", "gmres"],
        default="cg",
    )
    ksp_gmres_restart = Categorical(
        "ksp_gmres_restart",
        [20, 30, 50, 100],
        default=30,
    )
    ksp_pc_side = Constant("ksp_pc_side", "left")
    ksp_norm_type = Constant(
        "ksp_norm_type",
        "preconditioned",
    )
    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-8)
    ksp_max_it = Categorical(
        "ksp_max_it",
        [1000],
        default=1000,
    )

    pc_type = Categorical(
        "pc_type",
        ["spai", "gamg", "hypre", "bjacobi"],
        default="spai",
    )
    pc_hypre_type = Categorical(
        "pc_hypre_type",
        ["parasails", "pilut", "boomeramg"],
        default="parasails",
    )

    pc_spai_block_size = Constant(
        "pc_spai_block_size",
        1,
    )
    pc_spai_cache_size = Constant(
        "pc_spai_cache_size",
        5,
    )
    pc_spai_epsilon = Constant(
        "pc_spai_epsilon",
        0.0300007666235,
    )
    pc_spai_max = Constant("pc_spai_max", 68672)
    pc_spai_maxnew = Constant("pc_spai_maxnew", 50)
    pc_spai_nbsteps = Constant("pc_spai_nbsteps", 8)
    pc_spai_sp = Constant("pc_spai_sp", 1)

    ksp_gmres_cgs_refinement_type = Constant(
        "ksp_gmres_cgs_refinement_type",
        "refine_never",
    )
    ksp_gmres_modifiedgramschmidt = Constant(
        "ksp_gmres_modifiedgramschmidt",
        False,
        meta=petsc_meta("-ksp_gmres_modifiedgramschmidt", flag=True),
    )
    ksp_gmres_preallocate = Constant(
        "ksp_gmres_preallocate",
        False,
        meta=petsc_meta("-ksp_gmres_preallocate", flag=True),
    )
    mg_coarse_ksp_type = Constant(
        "mg_coarse_ksp_type",
        "gmres",
    )
    mg_coarse_pc_type = Constant(
        "mg_coarse_pc_type",
        "bjacobi",
    )
    mg_coarse_ksp_max_it = Constant(
        "mg_coarse_ksp_max_it",
        1,
    )
    mg_coarse_sub_pc_type = Constant(
        "mg_coarse_sub_pc_type",
        "ilu",
    )
    mg_coarse_sub_pc_factor_mat_ordering_type = Constant(
        "mg_coarse_sub_pc_factor_mat_ordering_type",
        "natural",
    )
    mg_levels_ksp_max_it = Constant(
        "mg_levels_ksp_max_it",
        1,
    )
    mg_levels_ksp_type = Constant(
        "mg_levels_ksp_type",
        "richardson",
    )
    mg_levels_pc_type = Constant(
        "mg_levels_pc_type",
        "jacobi",
    )
    pc_gamg_agg_nsmooths = Constant(
        "pc_gamg_agg_nsmooths",
        2,
    )
    pc_gamg_coarse_grid_layout_type = Constant(
        "pc_gamg_coarse_grid_layout_type",
        "spread",
    )
    pc_gamg_cpu_pin_coarse_grids = Constant(
        "pc_gamg_cpu_pin_coarse_grids",
        False,
    )
    pc_gamg_graph_symmetrize = Constant(
        "pc_gamg_graph_symmetrize",
        True,
    )
    pc_gamg_low_memory_threshold_filter = Constant(
        "pc_gamg_low_memory_threshold_filter",
        False,
    )
    pc_gamg_mat_coarsen_type = Constant(
        "pc_gamg_mat_coarsen_type",
        "misk",
    )
    pc_gamg_parallel_coarse_grid_solver = Constant(
        "pc_gamg_parallel_coarse_grid_solver",
        False,
    )
    pc_gamg_process_eq_limit = Constant(
        "pc_gamg_process_eq_limit",
        349,
    )
    pc_gamg_recompute_esteig = Constant(
        "pc_gamg_recompute_esteig",
        False,
    )
    pc_gamg_repartition = Constant(
        "pc_gamg_repartition",
        False,
    )
    pc_gamg_reuse_interpolation = Constant(
        "pc_gamg_reuse_interpolation",
        True,
    )
    pc_gamg_type = Constant("pc_gamg_type", "agg")
    pc_gamg_use_sa_esteig = Constant(
        "pc_gamg_use_sa_esteig",
        False,
    )
    pc_mg_cycle_type = Constant("pc_mg_cycle_type", "v")
    pc_mg_type = Constant("pc_mg_type", "kaskade")
    mg_levels_pc_jacobi_abs = Constant(
        "mg_levels_pc_jacobi_abs",
        True,
    )
    mg_levels_pc_jacobi_fixdiagonal = Constant(
        "mg_levels_pc_jacobi_fixdiagonal",
        False,
    )
    mg_levels_pc_jacobi_type = Constant(
        "mg_levels_pc_jacobi_type",
        "diagonal",
    )
    pc_gamg_aggressive_coarsening = Constant(
        "pc_gamg_aggressive_coarsening",
        0,
    )
    pc_gamg_mat_coarsen_misk_distance = Constant(
        "pc_gamg_mat_coarsen_misk_distance",
        3,
    )
    pc_gamg_mis_k_minimum_degree_ordering = Constant(
        "pc_gamg_mis_k_minimum_degree_ordering",
        False,
    )
    pc_gamg_threshold = Constant(
        "pc_gamg_threshold",
        0.1141898203948,
    )
    pc_gamg_threshold_scale = Constant(
        "pc_gamg_threshold_scale",
        0.2534279319386,
    )
    pc_hypre_parasails_loadbal = Constant(
        "pc_hypre_parasails_loadbal",
        0.1170590021484,
    )
    pc_hypre_parasails_logging = Constant(
        "pc_hypre_parasails_logging",
        False,
    )
    pc_hypre_parasails_nlevels = Constant(
        "pc_hypre_parasails_nlevels",
        0,
    )
    pc_hypre_parasails_reuse = Constant(
        "pc_hypre_parasails_reuse",
        False,
        meta=petsc_meta("-pc_hypre_parasails_reuse", skip_values=[False]),
    )
    pc_hypre_parasails_filter = Constant(
        "pc_hypre_parasails_filter",
        0.0,
    )
    pc_hypre_parasails_thresh = Constant(
        "pc_hypre_parasails_thresh",
        0.0028680776115,
    )

    pc_hypre_pilut_factorrowsize = Constant(
        "pc_hypre_pilut_factorrowsize",
        1837,
    )
    pc_hypre_pilut_maxiter = Constant(
        "pc_hypre_pilut_maxiter",
        17,
    )
    pc_hypre_pilut_tol = Constant(
        "pc_hypre_pilut_tol",
        9.8176496e-06,
    )

    sub_pc_type = Constant(
        "sub_pc_type",
        "ilu",
    )

    pc_hypre_boomeramg_p_max = Constant(
        "pc_hypre_boomeramg_P_max",
        2,
    )
    pc_hypre_boomeramg_agg_nl = Constant(
        "pc_hypre_boomeramg_agg_nl",
        5,
    )
    pc_hypre_boomeramg_coarsen_type = Constant(
        "pc_hypre_boomeramg_coarsen_type",
        "Falgout",
    )
    pc_hypre_boomeramg_cycle_type = Constant(
        "pc_hypre_boomeramg_cycle_type",
        "W",
    )
    pc_hypre_boomeramg_grid_sweeps_coarse = Constant(
        "pc_hypre_boomeramg_grid_sweeps_coarse",
        3,
    )
    pc_hypre_boomeramg_grid_sweeps_down = Constant(
        "pc_hypre_boomeramg_grid_sweeps_down",
        1,
    )
    pc_hypre_boomeramg_grid_sweeps_up = Constant(
        "pc_hypre_boomeramg_grid_sweeps_up",
        4,
    )
    pc_hypre_boomeramg_interp_refine = Constant(
        "pc_hypre_boomeramg_interp_refine",
        3,
    )
    pc_hypre_boomeramg_interp_type = Constant(
        "pc_hypre_boomeramg_interp_type",
        "standard-wts",
    )
    pc_hypre_boomeramg_max_coarse_size = Constant(
        "pc_hypre_boomeramg_max_coarse_size",
        2455,
    )
    pc_hypre_boomeramg_max_iter = Constant(
        "pc_hypre_boomeramg_max_iter",
        1,
    )
    pc_hypre_boomeramg_max_levels = Constant(
        "pc_hypre_boomeramg_max_levels",
        6,
    )
    pc_hypre_boomeramg_max_row_sum = Constant(
        "pc_hypre_boomeramg_max_row_sum",
        0.3240801122047,
    )
    pc_hypre_boomeramg_measure_type = Constant(
        "pc_hypre_boomeramg_measure_type",
        "global",
    )
    pc_hypre_boomeramg_min_coarse_size = Constant(
        "pc_hypre_boomeramg_min_coarse_size",
        33,
    )
    pc_hypre_boomeramg_relax_type_all = Constant(
        "pc_hypre_boomeramg_relax_type_all",
        "backward-l1-Gauss-Seidel",
    )
    pc_hypre_boomeramg_strong_threshold = Constant(
        "pc_hypre_boomeramg_strong_threshold",
        0.041958722133,
    )
    pc_hypre_boomeramg_tol = Constant(
        "pc_hypre_boomeramg_tol",
        0.0,
    )
    pc_hypre_boomeramg_truncfactor = Constant(
        "pc_hypre_boomeramg_truncfactor",
        0.9363566990464,
    )
    pc_hypre_boomeramg_agg_num_paths = Constant(
        "pc_hypre_boomeramg_agg_num_paths",
        4,
    )
    pc_hypre_boomeramg_relax_type_coarse = Constant(
        "pc_hypre_boomeramg_relax_type_coarse",
        "seqboundary-Gauss-Seidel",
    )
    pc_hypre_boomeramg_relax_type_down = Constant(
        "pc_hypre_boomeramg_relax_type_down",
        "symmetric-SOR/Jacobi",
    )
    pc_hypre_boomeramg_relax_type_up = Constant(
        "pc_hypre_boomeramg_relax_type_up",
        "backward-SOR/Jacobi",
    )

    spai_parameters = [
        pc_spai_block_size,
        pc_spai_cache_size,
        pc_spai_epsilon,
        pc_spai_max,
        pc_spai_maxnew,
        pc_spai_nbsteps,
        pc_spai_sp,
    ]
    gamg_parameters = [
        ksp_gmres_cgs_refinement_type,
        ksp_gmres_modifiedgramschmidt,
        ksp_gmres_preallocate,
        mg_coarse_ksp_type,
        mg_coarse_pc_type,
        mg_coarse_ksp_max_it,
        mg_coarse_sub_pc_type,
        mg_coarse_sub_pc_factor_mat_ordering_type,
        mg_levels_ksp_max_it,
        mg_levels_ksp_type,
        mg_levels_pc_type,
        pc_gamg_agg_nsmooths,
        pc_gamg_coarse_grid_layout_type,
        pc_gamg_cpu_pin_coarse_grids,
        pc_gamg_graph_symmetrize,
        pc_gamg_low_memory_threshold_filter,
        pc_gamg_mat_coarsen_type,
        pc_gamg_parallel_coarse_grid_solver,
        pc_gamg_process_eq_limit,
        pc_gamg_recompute_esteig,
        pc_gamg_repartition,
        pc_gamg_reuse_interpolation,
        pc_gamg_type,
        pc_gamg_use_sa_esteig,
        pc_mg_cycle_type,
        pc_mg_type,
        mg_levels_pc_jacobi_abs,
        mg_levels_pc_jacobi_fixdiagonal,
        mg_levels_pc_jacobi_type,
        pc_gamg_aggressive_coarsening,
        pc_gamg_mat_coarsen_misk_distance,
        pc_gamg_mis_k_minimum_degree_ordering,
        pc_gamg_threshold,
        pc_gamg_threshold_scale,
    ]
    parasails_parameters = [
        pc_hypre_parasails_loadbal,
        pc_hypre_parasails_logging,
        pc_hypre_parasails_nlevels,
        pc_hypre_parasails_reuse,
        pc_hypre_parasails_filter,
        pc_hypre_parasails_thresh,
    ]
    pilut_parameters = [
        pc_hypre_pilut_factorrowsize,
        pc_hypre_pilut_maxiter,
        pc_hypre_pilut_tol,
    ]
    bjacobi_parameters = [
        sub_pc_type,
    ]
    boomeramg_parameters = [
        pc_hypre_boomeramg_p_max,
        pc_hypre_boomeramg_agg_nl,
        pc_hypre_boomeramg_coarsen_type,
        pc_hypre_boomeramg_cycle_type,
        pc_hypre_boomeramg_grid_sweeps_coarse,
        pc_hypre_boomeramg_grid_sweeps_down,
        pc_hypre_boomeramg_grid_sweeps_up,
        pc_hypre_boomeramg_interp_refine,
        pc_hypre_boomeramg_interp_type,
        pc_hypre_boomeramg_max_coarse_size,
        pc_hypre_boomeramg_max_iter,
        pc_hypre_boomeramg_max_levels,
        pc_hypre_boomeramg_max_row_sum,
        pc_hypre_boomeramg_measure_type,
        pc_hypre_boomeramg_min_coarse_size,
        pc_hypre_boomeramg_relax_type_all,
        pc_hypre_boomeramg_strong_threshold,
        pc_hypre_boomeramg_tol,
        pc_hypre_boomeramg_truncfactor,
        pc_hypre_boomeramg_agg_num_paths,
        pc_hypre_boomeramg_relax_type_coarse,
        pc_hypre_boomeramg_relax_type_down,
        pc_hypre_boomeramg_relax_type_up,
    ]

    space.add(
        [
            ksp_type,
            ksp_gmres_restart,
            ksp_pc_side,
            ksp_norm_type,
            ksp_atol,
            ksp_rtol,
            ksp_max_it,
            pc_type,
            pc_hypre_type,
            *spai_parameters,
            *gamg_parameters,
            *parasails_parameters,
            *pilut_parameters,
            *boomeramg_parameters,
            *bjacobi_parameters,
        ]
    )

    space.add(
        [
            EqualsCondition(ksp_gmres_restart, ksp_type, "gmres"),
            EqualsCondition(pc_hypre_type, pc_type, "hypre"),
            *[EqualsCondition(parameter, pc_type, "spai") for parameter in spai_parameters],
            *[EqualsCondition(parameter, pc_type, "gamg") for parameter in gamg_parameters],
            *[
                hypre_subtype_condition(parameter, pc_type, pc_hypre_type, "parasails")
                for parameter in parasails_parameters
            ],
            *[
                hypre_subtype_condition(parameter, pc_type, pc_hypre_type, "pilut")
                for parameter in pilut_parameters
            ],
            *[
                hypre_subtype_condition(parameter, pc_type, pc_hypre_type, "boomeramg")
                for parameter in boomeramg_parameters
            ],
            *[EqualsCondition(parameter, pc_type, "bjacobi") for parameter in bjacobi_parameters],
        ]
    )

    space.add(
        [
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(pc_type, "spai"),
                ForbiddenEqualsClause(ksp_type, "gmres"),
            ),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(pc_type, "gamg"),
                ForbiddenEqualsClause(ksp_type, "cg"),
            ),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(pc_type, "hypre"),
                ForbiddenEqualsClause(ksp_type, "cg"),
            ),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(pc_type, "bjacobi"),
                ForbiddenEqualsClause(ksp_type, "gmres"),
            ),
        ]
    )

    space._ksptune_initial_solver_configurations = [
        initial_solver_configuration("spai_cg", ksp_type="cg", pc_type="spai"),
        initial_solver_configuration(
            "gamg_gmres",
            ksp_type="gmres",
            ksp_max_it=1000,
            ksp_gmres_restart=30,
            pc_type="gamg",
        ),
        initial_solver_configuration(
            "parasails_gmres",
            ksp_type="gmres",
            ksp_gmres_restart=30,
            pc_type="hypre",
            pc_hypre_type="parasails",
        ),
        initial_solver_configuration(
            "pilut_gmres",
            ksp_type="gmres",
            pc_type="hypre",
            pc_hypre_type="pilut",
        ),
        initial_solver_configuration(
            "boomeramg_gmres",
            ksp_type="gmres",
            ksp_gmres_restart=100,
            pc_type="hypre",
            pc_hypre_type="boomeramg",
        ),
        initial_solver_configuration(
            "bjacobi_ilu_cg_baseline",
            ksp_type="cg",
            pc_type="bjacobi",
        ),
    ]

    return space
