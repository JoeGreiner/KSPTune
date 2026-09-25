from __future__ import annotations

from copy import deepcopy
from typing import Any

from ConfigSpace import ConfigurationSpace

from ..search_space_helpers import initial_solver_configuration
from .petsc_gamg_extended import create_parameter_search_space as create_extended_space


def mono_initial_solver_configurations(base_space: ConfigurationSpace) -> list[dict[str, Any]]:
    initial_configurations = []
    for initial_configuration in deepcopy(
        getattr(base_space, "_ksptune_initial_solver_configurations", [])
    ):
        label = str(initial_configuration.get("label") or "")
        if "low_memory_filter" in label:
            continue
        solver_configuration = initial_configuration.get("solver_configuration") or {}
        if solver_configuration.get("mg_coarse_pc_type") == "ilu":
            solver_configuration["mg_coarse_pc_type"] = "bjacobi"
        solver_configuration["pc_gamg_low_memory_threshold_filter"] = False
        if solver_configuration.get("mg_levels_ksp_type") == "chebyshev":
            solver_configuration["mg_levels_ksp_chebyshev_esteig_noisy"] = True
        initial_configurations.append(initial_configuration)
    initial_configurations.extend(
        [
            initial_solver_configuration(
                "gamg_richardson_jacobi",
                ksp_reuse_preconditioner=True,
                ksp_gmres_cgs_refinement_type="refine_never",
                ksp_gmres_modifiedgramschmidt=False,
                ksp_gmres_preallocate=True,
                ksp_gmres_restart=50,
                ksp_max_it=1000,
                pc_mg_type="multiplicative",
                pc_mg_cycle_type="v",
                pc_mg_multiplicative_cycles=1,
                pc_gamg_agg_nsmooths=1,
                pc_gamg_aggressive_coarsening=3,
                pc_gamg_aggressive_square_graph=False,
                pc_gamg_aggressive_mis_k=3,
                pc_gamg_threshold_mode="set",
                pc_gamg_threshold=0.0485406826473,
                pc_gamg_threshold_scale=0.4437535377424,
                pc_gamg_low_memory_threshold_filter=False,
                pc_gamg_graph_symmetrize=False,
                pc_gamg_mis_k_minimum_degree_ordering=False,
                pc_gamg_mat_coarsen_type="misk",
                pc_gamg_mat_coarsen_misk_distance=3,
                mg_levels_ksp_type="richardson",
                mg_levels_ksp_max_it=1,
                mg_levels_pc_type="jacobi",
                mg_levels_pc_jacobi_type="diagonal",
                mg_levels_pc_jacobi_abs=False,
                mg_levels_pc_jacobi_fixdiagonal=True,
                mg_coarse_ksp_type="richardson",
                mg_coarse_ksp_max_it=1,
                mg_coarse_pc_type="bjacobi",
                mg_coarse_sub_pc_type="ilu",
                mg_coarse_sub_pc_factor_mat_ordering_type="rcm",
                pc_gamg_repartition=False,
                pc_gamg_use_sa_esteig=True,
                pc_gamg_recompute_esteig=True,
                pc_gamg_reuse_interpolation=True,
                pc_gamg_coarse_grid_layout_type="spread",
                pc_gamg_process_eq_limit=33,
                pc_gamg_parallel_coarse_grid_solver=False,
                pc_gamg_cpu_pin_coarse_grids=True,
                pc_gamg_esteig_ksp_type="cg",
                pc_gamg_esteig_ksp_max_it=10,
            ),
            initial_solver_configuration(
                "gamg_chebyshev_rowl1",
                ksp_reuse_preconditioner=True,
                ksp_gmres_restart=30,
                pc_mg_type="multiplicative",
                pc_mg_cycle_type="v",
                pc_mg_multiplicative_cycles=1,
                pc_gamg_agg_nsmooths=1,
                pc_gamg_aggressive_coarsening=2,
                pc_gamg_aggressive_square_graph=True,
                pc_gamg_threshold_mode="set",
                pc_gamg_threshold=0.01,
                pc_gamg_threshold_scale=1.0,
                pc_gamg_low_memory_threshold_filter=False,
                pc_gamg_graph_symmetrize=True,
                pc_gamg_mis_k_minimum_degree_ordering=False,
                pc_gamg_mat_coarsen_type="misk",
                pc_gamg_mat_coarsen_misk_distance=1,
                mg_levels_ksp_type="chebyshev",
                mg_levels_ksp_chebyshev_esteig_noisy=True,
                mg_levels_ksp_max_it=1,
                mg_levels_pc_type="jacobi",
                mg_levels_pc_jacobi_type="rowl1",
                mg_levels_pc_jacobi_abs=False,
                mg_levels_pc_jacobi_fixdiagonal=True,
                mg_levels_pc_jacobi_rowl1_scale=1.0,
                mg_coarse_ksp_type="preonly",
                mg_coarse_pc_type="bjacobi",
                mg_coarse_sub_pc_type="lu",
                mg_coarse_sub_pc_factor_mat_ordering_type="nd",
                pc_gamg_repartition=False,
                pc_gamg_use_sa_esteig=True,
                pc_gamg_recompute_esteig=True,
                pc_gamg_reuse_interpolation=True,
                pc_gamg_coarse_grid_layout_type="spread",
                pc_gamg_process_eq_limit=50,
                pc_gamg_parallel_coarse_grid_solver=False,
                pc_gamg_cpu_pin_coarse_grids=False,
            ),
        ]
    )
    return initial_configurations


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    space = create_extended_space(seed=seed, mono=True)
    space._ksptune_initial_solver_configurations = mono_initial_solver_configurations(space)
    return space
