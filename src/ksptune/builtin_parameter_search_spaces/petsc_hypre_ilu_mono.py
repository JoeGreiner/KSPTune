from __future__ import annotations

from ..search_space_helpers import petsc_meta

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.hypre_ilu_mono", seed=seed)

    ksp_type = Constant("ksp_type", "gmres")
    ksp_gmres_restart = Constant(
        "ksp_gmres_restart",
        30,
        meta=petsc_meta("-ksp_gmres_restart", skip_values=[30]),
    )
    ksp_pc_side = Constant("ksp_pc_side", "left")
    ksp_norm_type = Constant(
        "ksp_norm_type",
        "preconditioned",
    )
    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-8)
    ksp_max_it = Constant("ksp_max_it", 1000)

    pc_type = Constant("pc_type", "hypre")
    pc_hypre_type = Constant("pc_hypre_type", "ilu")

    pc_hypre_ilu_type = Categorical(
        "pc_hypre_ilu_type",
        [
            "Block-Jacobi-ILUk",
            "Block-Jacobi-ILUT",
            "GMRES-ILUk",
            "GMRES-ILUT",
            "NSH-ILUk",
            "NSH-ILUT",
            "RAS-ILUk",
            "RAS-ILUT",
            "ddPQ-GMRES-ILUk",
            "ddPQ-GMRES-ILUT",
            "GMRES-ILU0",
        ],
        default="Block-Jacobi-ILUT",
    )
    pc_hypre_ilu_level = UniformIntegerHyperparameter(
        "pc_hypre_ilu_level",
        lower=0,
        upper=5,
        default_value=1,
    )
    pc_hypre_ilu_max_nnz_per_row = UniformIntegerHyperparameter(
        "pc_hypre_ilu_max_nnz_per_row",
        lower=10,
        upper=2000,
        default_value=657,
    )
    pc_hypre_ilu_tol = Constant(
        "pc_hypre_ilu_tol",
        0.0,
    )
    pc_hypre_ilu_maxiter = Constant(
        "pc_hypre_ilu_maxiter",
        1,
    )
    pc_hypre_ilu_drop_threshold = UniformFloatHyperparameter(
        "pc_hypre_ilu_drop_threshold",
        lower=1.0e-8,
        upper=1.0e-1,
        default_value=3.593594e-7,
        log=True,
    )
    pc_hypre_ilu_tri_solve = Categorical(
        "pc_hypre_ilu_tri_solve",
        [True, False],
        default=True,
    )
    pc_hypre_ilu_lower_jacobi_iters = UniformIntegerHyperparameter(
        "pc_hypre_ilu_lower_jacobi_iters",
        lower=1,
        upper=5,
        default_value=1,
    )
    pc_hypre_ilu_upper_jacobi_iters = UniformIntegerHyperparameter(
        "pc_hypre_ilu_upper_jacobi_iters",
        lower=1,
        upper=5,
        default_value=1,
    )
    pc_hypre_ilu_local_reordering = Categorical(
        "pc_hypre_ilu_local_reordering",
        [False, True],
        default=False,
    )

    parameter_search_space.add(
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
            pc_hypre_ilu_type,
            pc_hypre_ilu_level,
            pc_hypre_ilu_max_nnz_per_row,
            pc_hypre_ilu_tol,
            pc_hypre_ilu_maxiter,
            pc_hypre_ilu_drop_threshold,
            pc_hypre_ilu_tri_solve,
            pc_hypre_ilu_lower_jacobi_iters,
            pc_hypre_ilu_upper_jacobi_iters,
            pc_hypre_ilu_local_reordering,
        ]
    )

    parameter_search_space.add(
        [
            EqualsCondition(
                pc_hypre_ilu_lower_jacobi_iters,
                pc_hypre_ilu_tri_solve,
                False,
            ),
            EqualsCondition(
                pc_hypre_ilu_upper_jacobi_iters,
                pc_hypre_ilu_tri_solve,
                False,
            ),
        ]
    )

    return parameter_search_space
