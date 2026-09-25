from __future__ import annotations

from ..search_space_helpers import bool_choice, initial_solver_configuration, petsc_meta

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import GreaterThanCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.boomeramg_extended", seed=seed)

    ksp_type = Constant("ksp_type", "gmres")
    ksp_pc_side = Constant("ksp_pc_side", "left")
    ksp_norm_type = Constant(
        "ksp_norm_type",
        "preconditioned",
    )
    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-8)

    pc_type = Constant("pc_type", "hypre")
    pc_hypre_type = Constant(
        "pc_hypre_type",
        "boomeramg",
    )
    pc_hypre_boomeramg_cycle_type = Constant(
        "pc_hypre_boomeramg_cycle_type",
        "V",
    )
    pc_hypre_boomeramg_max_iter = Constant(
        "pc_hypre_boomeramg_max_iter",
        1,
    )
    pc_hypre_boomeramg_tol = Constant(
        "pc_hypre_boomeramg_tol",
        0.0,
    )

    pc_hypre_boomeramg_coarsen_type = Categorical(
        "pc_hypre_boomeramg_coarsen_type",
        # tested on 6x5x4 and 10x10x10
        ["PMIS", "HMIS"],
        default="PMIS",
    )
    pc_hypre_boomeramg_interp_type = Categorical(
        "pc_hypre_boomeramg_interp_type",
        # tested on 6x5x4 and 10x10x10
        [
            "ext+i",
            "ext+i-cc",
            "FF",
            "FF1",
            "ext",
            "ext-mm",
            "ext+i-mm",
            "ext+e-mm",
        ],
        default="ext+i-mm",
    )
    pc_hypre_boomeramg_relax_type_all = Categorical(
        "pc_hypre_boomeramg_relax_type_all",
        # tested on 6x5x4 and 10x10x10
        [
            "SOR/Jacobi",
            "symmetric-SOR/Jacobi",
            "l1scaled-SOR/Jacobi",
            "l1-Gauss-Seidel",
            "backward-l1-Gauss-Seidel",
            "Chebyshev",
            "l1scaled-Jacobi",
        ],
        default="l1-Gauss-Seidel",
    )
    pc_hypre_boomeramg_measure_type = Categorical(
        "pc_hypre_boomeramg_measure_type",
        ["local", "global"],
        default="local",
    )

    pc_hypre_boomeramg_strong_threshold = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_strong_threshold",
        lower=0.0,
        upper=0.9,
        default_value=0.1,
    )
    pc_hypre_boomeramg_truncfactor = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_truncfactor",
        lower=0.0,
        upper=0.5,
        default_value=0.0,
        meta=petsc_meta("-pc_hypre_boomeramg_truncfactor", skip_values=[0.0]),
    )
    pc_hypre_boomeramg_max_row_sum = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_max_row_sum",
        lower=0.0,
        upper=1.0,
        default_value=0.9,
    )

    pc_hypre_boomeramg_p_max = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_P_max",
        lower=0,
        upper=16,
        default_value=4,
    )
    pc_hypre_boomeramg_agg_nl = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_agg_nl",
        lower=0,
        upper=4,
        default_value=0,
    )
    pc_hypre_boomeramg_agg_num_paths = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_agg_num_paths",
        lower=1,
        upper=4,
        default_value=1,
    )
    pc_hypre_boomeramg_max_levels = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_max_levels",
        lower=10,
        upper=60,
        default_value=25,
    )
    pc_hypre_boomeramg_max_coarse_size = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_max_coarse_size",
        lower=8,
        upper=512,
        default_value=9,
    )
    pc_hypre_boomeramg_min_coarse_size = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_min_coarse_size",
        lower=1,
        upper=16,
        default_value=1,
    )
    # tested on 6x5x4 and 10x10x10
    pc_hypre_boomeramg_grid_sweeps_down = Constant(
        "pc_hypre_boomeramg_grid_sweeps_down",
        1,
    )
    # tested on 6x5x4 and 10x10x10
    pc_hypre_boomeramg_grid_sweeps_up = Constant(
        "pc_hypre_boomeramg_grid_sweeps_up",
        1,
    )
    pc_hypre_boomeramg_grid_sweeps_coarse = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_grid_sweeps_coarse",
        lower=1,
        upper=3,
        default_value=1,
    )

    pc_hypre_boomeramg_keeptranspose = bool_choice(
        "pc_hypre_boomeramg_keeptranspose",
        False,
        "-pc_hypre_boomeramg_keeptranspose",
        skip_false=True,
    )

    parameter_search_space.add(
        [
            ksp_type,
            ksp_pc_side,
            ksp_norm_type,
            ksp_atol,
            ksp_rtol,
            pc_type,
            pc_hypre_type,
            pc_hypre_boomeramg_cycle_type,
            pc_hypre_boomeramg_max_iter,
            pc_hypre_boomeramg_tol,
            pc_hypre_boomeramg_coarsen_type,
            pc_hypre_boomeramg_interp_type,
            pc_hypre_boomeramg_relax_type_all,
            pc_hypre_boomeramg_measure_type,
            pc_hypre_boomeramg_strong_threshold,
            pc_hypre_boomeramg_truncfactor,
            pc_hypre_boomeramg_max_row_sum,
            pc_hypre_boomeramg_p_max,
            pc_hypre_boomeramg_agg_nl,
            pc_hypre_boomeramg_agg_num_paths,
            pc_hypre_boomeramg_max_levels,
            pc_hypre_boomeramg_max_coarse_size,
            pc_hypre_boomeramg_min_coarse_size,
            pc_hypre_boomeramg_grid_sweeps_down,
            pc_hypre_boomeramg_grid_sweeps_up,
            pc_hypre_boomeramg_grid_sweeps_coarse,
            pc_hypre_boomeramg_keeptranspose,
        ]
    )
    parameter_search_space.add(
        [
            GreaterThanCondition(
                pc_hypre_boomeramg_agg_num_paths,
                pc_hypre_boomeramg_agg_nl,
                0,
            ),
        ]
    )
    parameter_search_space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "boomeramg_exti_threshold_0p05",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_strong_threshold=0.05,
        ),
        initial_solver_configuration(
            "boomeramg_ff_agg_1_paths_2",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.25,
        ),
        initial_solver_configuration(
            "boomeramg_hmis_exti_pmax_3",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_P_max=3,
            pc_hypre_boomeramg_strong_threshold=0.175,
            pc_hypre_boomeramg_truncfactor=0.225,
        ),
        initial_solver_configuration(
            "boomeramg_exti_pmax_8",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_P_max=8,
            pc_hypre_boomeramg_strong_threshold=0.25,
            pc_hypre_boomeramg_truncfactor=0.05,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_exticc_agg_0",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+i-cc",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_measure_type="local",
            pc_hypre_boomeramg_P_max=4,
            pc_hypre_boomeramg_agg_nl=0,
            pc_hypre_boomeramg_strong_threshold=0.075,
            pc_hypre_boomeramg_truncfactor=0.0,
        ),
        initial_solver_configuration(
            "boomeramg_hmis_exti_paths_2",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=7,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.0388801205714,
            pc_hypre_boomeramg_truncfactor=0.0021247466225,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_exti_paths_2",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=8,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.0388801205714,
            pc_hypre_boomeramg_truncfactor=0.0021247466225,
        ),
        initial_solver_configuration(
            "boomeramg_hmis_extimm_paths_2",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+i-mm",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=7,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.0388801205714,
            pc_hypre_boomeramg_truncfactor=0.0024114874969,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_exti_paths_3",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=8,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.0662860419053,
            pc_hypre_boomeramg_truncfactor=0.0044798272768,
        ),
        initial_solver_configuration(
            "boomeramg_hmis_ff_paths_1",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=8,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=1,
            pc_hypre_boomeramg_strong_threshold=0.0645004637445,
            pc_hypre_boomeramg_truncfactor=0.0044798272768,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_extimm_agg_0",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+i-mm",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=6,
            pc_hypre_boomeramg_agg_nl=0,
            pc_hypre_boomeramg_strong_threshold=0.025,
            pc_hypre_boomeramg_truncfactor=0.025,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_ff_agg_0",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=5,
            pc_hypre_boomeramg_agg_nl=0,
            pc_hypre_boomeramg_strong_threshold=0.05,
            pc_hypre_boomeramg_truncfactor=0.1,
        ),
    ]
    return parameter_search_space
