from __future__ import annotations

from ..search_space_helpers import initial_solver_configuration, petsc_meta

from ConfigSpace import (
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.pilut_mono", seed=seed)

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
    pc_hypre_type = Constant(
        "pc_hypre_type",
        "pilut",
    )

    pc_hypre_pilut_factorrowsize = UniformIntegerHyperparameter(
        "pc_hypre_pilut_factorrowsize",
        lower=20,
        upper=8000,
        default_value=150,
    )
    pc_hypre_pilut_tol = UniformFloatHyperparameter(
        "pc_hypre_pilut_tol",
        lower=1.0e-8,
        upper=1.0,
        default_value=1.0e-4,
        log=True,
    )
    pc_hypre_pilut_maxiter = UniformIntegerHyperparameter(
        "pc_hypre_pilut_maxiter",
        lower=1,
        upper=50,
        default_value=1,
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
            pc_hypre_pilut_factorrowsize,
            pc_hypre_pilut_tol,
            pc_hypre_pilut_maxiter,
        ]
    )

    parameter_search_space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "pilut_factor_rows_5780",
            pc_hypre_pilut_factorrowsize=5780,
            pc_hypre_pilut_maxiter=2,
            pc_hypre_pilut_tol=0.0956704706854,
        ),
        initial_solver_configuration(
            "pilut_factor_rows_477",
            pc_hypre_pilut_factorrowsize=477,
            pc_hypre_pilut_maxiter=11,
            pc_hypre_pilut_tol=9.8313068e-06,
        ),
    ]

    return parameter_search_space
