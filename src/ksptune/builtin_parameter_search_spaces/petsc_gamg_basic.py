from __future__ import annotations

from ..search_space_helpers import bool_choice, initial_solver_configuration

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import AndConjunction, EqualsCondition, InCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.gamg_basic", seed=seed)

    # Keep the solver identity explicit even where PETSc would choose the same defaults.
    ksp_type = Constant("ksp_type", "gmres")
    ksp_pc_side = Constant("ksp_pc_side", "left")
    pc_type = Constant("pc_type", "gamg")

    # Keep the numerical target fixed while tuning algorithmic choices.
    ksp_norm_type = Constant(
        "ksp_norm_type",
        "preconditioned",
    )
    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-7)
    ksp_reuse_preconditioner = Constant(
        "ksp_reuse_preconditioner",
        True,
    )

    pc_gamg_agg_nsmooths = UniformIntegerHyperparameter(
        "pc_gamg_agg_nsmooths",
        lower=1,
        upper=2,
        default_value=1,
    )
    pc_gamg_aggressive_coarsening = UniformIntegerHyperparameter(
        "pc_gamg_aggressive_coarsening",
        lower=2,
        upper=6,
        default_value=3,
    )
    pc_gamg_aggressive_square_graph = bool_choice(
        "pc_gamg_aggressive_square_graph",
        True,
        "-pc_gamg_aggressive_square_graph",
    )
    pc_gamg_aggressive_mis_k = UniformIntegerHyperparameter(
        "pc_gamg_aggressive_mis_k",
        lower=2,
        upper=4,
        default_value=2,
    )
    pc_gamg_threshold = UniformFloatHyperparameter(
        "pc_gamg_threshold",
        lower=0.0,
        upper=0.02,
        default_value=0.01,
    )
    pc_gamg_threshold_scale = UniformFloatHyperparameter(
        "pc_gamg_threshold_scale",
        lower=0.5,
        upper=1.25,
        default_value=1.0,
    )
    pc_gamg_low_memory_threshold_filter = bool_choice(
        "pc_gamg_low_memory_threshold_filter",
        False,
        "-pc_gamg_low_memory_threshold_filter",
    )
    pc_gamg_mat_coarsen_type = Categorical(
        "pc_gamg_mat_coarsen_type",
        ["misk", "mis"],
        default="mis",
    )
    pc_gamg_mat_coarsen_misk_distance = Constant(
        "pc_gamg_mat_coarsen_misk_distance",
        1,
    )
    mg_levels_ksp_type = Constant(
        "mg_levels_ksp_type",
        "chebyshev",
    )
    mg_levels_ksp_max_it = UniformIntegerHyperparameter(
        "mg_levels_ksp_max_it",
        lower=1,
        upper=2,
        default_value=1,
    )
    mg_levels_pc_type = Categorical(
        "mg_levels_pc_type",
        ["jacobi", "sor"],
        default="jacobi",
    )

    pc_gamg_repartition = bool_choice("pc_gamg_repartition", False, "-pc_gamg_repartition")
    pc_gamg_coarse_grid_layout_type = Categorical(
        "pc_gamg_coarse_grid_layout_type",
        ["spread", "compact"],
        default="spread",
    )
    pc_gamg_process_eq_limit = UniformIntegerHyperparameter(
        "pc_gamg_process_eq_limit",
        lower=25,
        upper=300,
        default_value=50,
    )
    pc_gamg_parallel_coarse_grid_solver = bool_choice(
        "pc_gamg_parallel_coarse_grid_solver",
        False,
        "-pc_gamg_parallel_coarse_grid_solver",
    )
    pc_gamg_coarse_eq_limit = UniformIntegerHyperparameter(
        "pc_gamg_coarse_eq_limit",
        lower=500,
        upper=12000,
        default_value=3000,
    )
    pc_gamg_recompute_esteig = Constant(
        "pc_gamg_recompute_esteig",
        False,
    )
    pc_gamg_reuse_interpolation = Constant(
        "pc_gamg_reuse_interpolation",
        True,
    )

    parameter_search_space.add(
        [
            ksp_type,
            ksp_pc_side,
            pc_type,
            ksp_norm_type,
            ksp_atol,
            ksp_rtol,
            ksp_reuse_preconditioner,
            pc_gamg_agg_nsmooths,
            pc_gamg_aggressive_coarsening,
            pc_gamg_aggressive_square_graph,
            pc_gamg_aggressive_mis_k,
            pc_gamg_threshold,
            pc_gamg_threshold_scale,
            pc_gamg_low_memory_threshold_filter,
            pc_gamg_mat_coarsen_type,
            pc_gamg_mat_coarsen_misk_distance,
            mg_levels_ksp_type,
            mg_levels_ksp_max_it,
            mg_levels_pc_type,
            pc_gamg_repartition,
            pc_gamg_coarse_grid_layout_type,
            pc_gamg_process_eq_limit,
            pc_gamg_parallel_coarse_grid_solver,
            pc_gamg_coarse_eq_limit,
            pc_gamg_recompute_esteig,
            pc_gamg_reuse_interpolation,
        ]
    )

    parameter_search_space.add(
        [
            AndConjunction(
                InCondition(
                    pc_gamg_aggressive_mis_k,
                    pc_gamg_aggressive_coarsening,
                    [2, 3, 4, 5, 6],
                ),
                EqualsCondition(
                    pc_gamg_aggressive_mis_k,
                    pc_gamg_aggressive_square_graph,
                    False,
                ),
            ),
            EqualsCondition(
                pc_gamg_mat_coarsen_misk_distance,
                pc_gamg_mat_coarsen_type,
                "misk",
            ),
            EqualsCondition(
                pc_gamg_coarse_eq_limit,
                pc_gamg_parallel_coarse_grid_solver,
                True,
            ),
        ]
    )

    parameter_search_space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "gamg_mis_chebyshev_jacobi_compact",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="jacobi",
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=4,
            pc_gamg_aggressive_square_graph=False,
            pc_gamg_aggressive_mis_k=4,
            pc_gamg_mat_coarsen_type="mis",
            pc_gamg_threshold=0.0084794943117,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_process_eq_limit=145,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_parallel_coarse_grid_solver=False,
            pc_gamg_repartition=False,
        ),
        initial_solver_configuration(
            "gamg_mis_parallel_coarse_compact",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="jacobi",
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=5,
            pc_gamg_aggressive_square_graph=True,
            pc_gamg_mat_coarsen_type="mis",
            pc_gamg_threshold=0.0091676848579,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_coarse_eq_limit=5781,
            pc_gamg_process_eq_limit=101,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_parallel_coarse_grid_solver=True,
            pc_gamg_repartition=False,
        ),
        initial_solver_configuration(
            "gamg_mis_low_peak_compact",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="jacobi",
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=6,
            pc_gamg_aggressive_square_graph=True,
            pc_gamg_mat_coarsen_type="mis",
            pc_gamg_threshold=0.0080952794789,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_process_eq_limit=94,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_parallel_coarse_grid_solver=False,
            pc_gamg_repartition=False,
        ),
        initial_solver_configuration(
            "gamg_mis_repartition_spread",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="jacobi",
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=6,
            pc_gamg_aggressive_square_graph=True,
            pc_gamg_mat_coarsen_type="mis",
            pc_gamg_threshold=0.0085392709523,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_coarse_eq_limit=5700,
            pc_gamg_process_eq_limit=215,
            pc_gamg_coarse_grid_layout_type="spread",
            pc_gamg_parallel_coarse_grid_solver=True,
            pc_gamg_repartition=True,
        ),
        initial_solver_configuration(
            "gamg_mis_sor_low_mpi",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="sor",
            pc_gamg_agg_nsmooths=2,
            pc_gamg_aggressive_coarsening=4,
            pc_gamg_aggressive_square_graph=True,
            pc_gamg_mat_coarsen_type="mis",
            pc_gamg_threshold=0.0052718151205,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_process_eq_limit=201,
            pc_gamg_coarse_grid_layout_type="spread",
            pc_gamg_parallel_coarse_grid_solver=False,
            pc_gamg_repartition=False,
        ),
        initial_solver_configuration(
            "gamg_misk_startup_seed",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="jacobi",
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=3,
            pc_gamg_aggressive_square_graph=True,
            pc_gamg_mat_coarsen_type="misk",
            pc_gamg_mat_coarsen_misk_distance=1,
            pc_gamg_threshold=0.02,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_coarse_eq_limit=3000,
            pc_gamg_process_eq_limit=100,
            pc_gamg_coarse_grid_layout_type="spread",
            pc_gamg_parallel_coarse_grid_solver=True,
            pc_gamg_repartition=False,
        ),
        initial_solver_configuration(
            "gamg_mis_threshold_0_baseline",
            mg_levels_ksp_max_it=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_pc_type="jacobi",
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=3,
            pc_gamg_aggressive_square_graph=False,
            pc_gamg_aggressive_mis_k=2,
            pc_gamg_mat_coarsen_type="mis",
            pc_gamg_threshold=0.0,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_process_eq_limit=50,
            pc_gamg_coarse_grid_layout_type="spread",
            pc_gamg_parallel_coarse_grid_solver=False,
            pc_gamg_repartition=False,
        ),
    ]
    return parameter_search_space
