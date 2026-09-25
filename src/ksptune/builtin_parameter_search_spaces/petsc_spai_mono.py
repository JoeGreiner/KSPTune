from __future__ import annotations

from ..search_space_helpers import initial_solver_configuration, petsc_meta

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.spai_mono", seed=seed)

    ksp_type = Categorical(
        "ksp_type",
        ["cg", "gmres"],
        default="cg",
    )
    ksp_gmres_restart = Categorical(
        "ksp_gmres_restart",
        [10, 20, 30, 50],
        default=30,
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

    pc_type = Constant("pc_type", "spai")

    pc_spai_epsilon = UniformFloatHyperparameter(
        "pc_spai_epsilon",
        lower=0.005,
        upper=0.8,
        default_value=0.0372778643732,
        log=True,
    )
    pc_spai_nbsteps = UniformIntegerHyperparameter(
        "pc_spai_nbsteps",
        lower=1,
        upper=100,
        default_value=10,
    )
    pc_spai_maxnew = UniformIntegerHyperparameter(
        "pc_spai_maxnew",
        lower=1,
        upper=64,
        default_value=2,
    )
    pc_spai_max = UniformIntegerHyperparameter(
        "pc_spai_max",
        lower=5000,
        upper=100000,
        default_value=10000,
    )

    pc_spai_block_size = Categorical(
        "pc_spai_block_size",
        [1, 0, 2, 4],
        default=1,
    )
    pc_spai_cache_size = Constant(
        "pc_spai_cache_size",
        5,
    )
    pc_spai_sp = Categorical(
        "pc_spai_sp",
        [1, 0],
        default=1,
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
            pc_spai_epsilon,
            pc_spai_nbsteps,
            pc_spai_maxnew,
            pc_spai_max,
            pc_spai_block_size,
            pc_spai_cache_size,
            pc_spai_sp,
        ]
    )
    parameter_search_space.add(EqualsCondition(ksp_gmres_restart, ksp_type, "gmres"))

    parameter_search_space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "spai_steps_10",
            pc_spai_epsilon=0.0372778643732,
            pc_spai_nbsteps=10,
            pc_spai_maxnew=2,
            pc_spai_max=10000,
            pc_spai_block_size=1,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_steps_3",
            pc_spai_epsilon=0.0348025385901,
            pc_spai_nbsteps=3,
            pc_spai_maxnew=10,
            pc_spai_max=5000,
            pc_spai_block_size=1,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_steps_5",
            pc_spai_epsilon=0.03,
            pc_spai_nbsteps=5,
            pc_spai_maxnew=1,
            pc_spai_max=10000,
            pc_spai_block_size=1,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_steps_20",
            pc_spai_epsilon=0.05,
            pc_spai_nbsteps=20,
            pc_spai_maxnew=4,
            pc_spai_max=20000,
            pc_spai_block_size=1,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_steps_30",
            pc_spai_epsilon=0.1,
            pc_spai_nbsteps=30,
            pc_spai_maxnew=8,
            pc_spai_max=50000,
            pc_spai_block_size=1,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_variable_blocks",
            pc_spai_epsilon=0.0372778643732,
            pc_spai_nbsteps=10,
            pc_spai_maxnew=2,
            pc_spai_max=20000,
            pc_spai_block_size=0,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_block_size_2",
            pc_spai_epsilon=0.0372778643732,
            pc_spai_nbsteps=10,
            pc_spai_maxnew=2,
            pc_spai_max=20000,
            pc_spai_block_size=2,
            pc_spai_sp=1,
        ),
        initial_solver_configuration(
            "spai_nonsymmetric_pattern",
            pc_spai_epsilon=0.0372778643732,
            pc_spai_nbsteps=10,
            pc_spai_maxnew=2,
            pc_spai_max=10000,
            pc_spai_block_size=1,
            pc_spai_sp=0,
        ),
    ]

    return parameter_search_space
