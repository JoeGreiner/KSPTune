from __future__ import annotations

from ..search_space_helpers import initial_solver_configuration

from ConfigSpace import Categorical, ConfigurationSpace, Constant
from ConfigSpace.conditions import EqualsCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    space = ConfigurationSpace(name="petsc.direct_mono", seed=seed)

    ksp_type = Categorical(
        "ksp_type",
        ["preonly", "richardson"],
        default="preonly",
    )
    ksp_max_it = Constant("ksp_max_it", 1)
    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-8)

    pc_type = Constant("pc_type", "lu")
    pc_factor_mat_solver_type = Categorical(
        "pc_factor_mat_solver_type",
        ["superlu_dist", "strumpack", "pastix", "mumps"],
        default="superlu_dist",
    )
    pc_factor_mat_ordering_type = Categorical(
        "pc_factor_mat_ordering_type",
        ["natural", "nd", "rcm"],
        default="nd",
    )
    mat_mumps_icntl_20 = Constant(
        "mat_mumps_icntl_20",
        0,
    )

    space.add(
        [
            ksp_type,
            ksp_max_it,
            ksp_atol,
            ksp_rtol,
            pc_type,
            pc_factor_mat_solver_type,
            pc_factor_mat_ordering_type,
            mat_mumps_icntl_20,
        ]
    )
    space.add(
        [
            EqualsCondition(
                mat_mumps_icntl_20,
                pc_factor_mat_solver_type,
                "mumps",
            ),
        ]
    )

    initial_configurations = []
    for solver in ["superlu_dist", "strumpack", "pastix", "mumps"]:
        for ordering in ["nd", "natural", "rcm"]:
            overrides = {
                "ksp_type": "preonly",
                "pc_factor_mat_solver_type": solver,
                "pc_factor_mat_ordering_type": ordering,
            }
            if solver == "mumps":
                overrides["mat_mumps_icntl_20"] = 0
            initial_configurations.append(
                initial_solver_configuration(
                    f"lu_{solver}_preonly_{ordering}",
                    **overrides,
                )
            )
        overrides = {
            "ksp_type": "richardson",
            "pc_factor_mat_solver_type": solver,
            "pc_factor_mat_ordering_type": "nd",
        }
        if solver == "mumps":
            overrides["mat_mumps_icntl_20"] = 0
        initial_configurations.append(
            initial_solver_configuration(
                f"lu_{solver}_richardson_nd",
                **overrides,
            )
        )

    space._ksptune_initial_solver_configurations = initial_configurations

    return space
