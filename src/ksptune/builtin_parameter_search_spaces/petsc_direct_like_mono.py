from __future__ import annotations

from ..search_space_helpers import initial_solver_configuration, petsc_meta

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    space = ConfigurationSpace(name="petsc.direct_like_mono", seed=seed)

    ksp_type = Categorical(
        "ksp_type",
        ["gmres", "cg"],
        default="gmres",
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

    pc_type = Categorical(
        "pc_type",
        ["hypre", "spai", "asm", "bjacobi"],
        default="hypre",
    )

    pc_hypre_type = Constant(
        "pc_hypre_type",
        "pilut",
    )
    pc_hypre_pilut_factorrowsize = Categorical(
        "pc_hypre_pilut_factorrowsize",
        [50, 100, 150, 200, 300, 400, 600, 800],
        default=150,
    )
    pc_hypre_pilut_maxiter = Categorical(
        "pc_hypre_pilut_maxiter",
        [-2, 1, 2, 5],
        default=-2,
        meta=petsc_meta("-pc_hypre_pilut_maxiter", skip_values=[-2]),
    )

    pc_spai_epsilon = UniformFloatHyperparameter(
        "pc_spai_epsilon",
        lower=0.02,
        upper=0.25,
        default_value=0.05,
        log=False,
    )
    pc_spai_nbsteps = Categorical(
        "pc_spai_nbsteps",
        [2, 3, 5, 8, 10],
        default=5,
        meta=petsc_meta("-pc_spai_nbsteps", skip_values=[5]),
    )
    pc_spai_maxnew = Categorical(
        "pc_spai_maxnew",
        [2, 5, 10, 20],
        default=5,
        meta=petsc_meta("-pc_spai_maxnew", skip_values=[5]),
    )
    pc_spai_max = Categorical(
        "pc_spai_max",
        [5000, 10000],
        default=5000,
        meta=petsc_meta("-pc_spai_max", skip_values=[5000]),
    )

    pc_asm_overlap = Constant(
        "pc_asm_overlap",
        1,
    )
    sub_pc_type = Constant(
        "sub_pc_type",
        "ilu",
    )

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
            pc_hypre_pilut_factorrowsize,
            pc_hypre_pilut_maxiter,
            pc_spai_epsilon,
            pc_spai_nbsteps,
            pc_spai_maxnew,
            pc_spai_max,
            pc_asm_overlap,
            sub_pc_type,
        ]
    )

    space.add(
        [
            EqualsCondition(ksp_gmres_restart, ksp_type, "gmres"),
            EqualsCondition(pc_hypre_type, pc_type, "hypre"),
            EqualsCondition(pc_hypre_pilut_factorrowsize, pc_type, "hypre"),
            EqualsCondition(pc_hypre_pilut_maxiter, pc_type, "hypre"),
            EqualsCondition(pc_spai_epsilon, pc_type, "spai"),
            EqualsCondition(pc_spai_nbsteps, pc_type, "spai"),
            EqualsCondition(pc_spai_maxnew, pc_type, "spai"),
            EqualsCondition(pc_spai_max, pc_type, "spai"),
            EqualsCondition(pc_asm_overlap, pc_type, "asm"),
            InCondition(sub_pc_type, pc_type, ["asm", "bjacobi"]),
        ]
    )

    # Solver/PC pairings from quick trials.
    space.add(
        [
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(ksp_type, "cg"),
                ForbiddenEqualsClause(pc_type, "hypre"),
            ),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(ksp_type, "gmres"),
                ForbiddenEqualsClause(pc_type, "spai"),
            ),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(ksp_type, "cg"),
                ForbiddenEqualsClause(pc_type, "asm"),
            ),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(ksp_type, "gmres"),
                ForbiddenEqualsClause(pc_type, "bjacobi"),
            ),
        ]
    )

    space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "pilut_factor_rows_150",
            ksp_type="gmres",
            pc_type="hypre",
            pc_hypre_pilut_factorrowsize=150,
        ),
        initial_solver_configuration(
            "pilut_factor_rows_200",
            ksp_type="gmres",
            pc_type="hypre",
            pc_hypre_pilut_factorrowsize=200,
        ),
        initial_solver_configuration(
            "pilut_factor_rows_400_max_iter_5",
            ksp_type="gmres",
            pc_type="hypre",
            pc_hypre_pilut_factorrowsize=400,
            pc_hypre_pilut_maxiter=5,
        ),
        initial_solver_configuration(
            "spai_epsilon_0p05",
            ksp_type="cg",
            pc_type="spai",
            pc_spai_epsilon=0.05,
        ),
        initial_solver_configuration(
            "spai_epsilon_0p1",
            ksp_type="cg",
            pc_type="spai",
            pc_spai_epsilon=0.1,
        ),
        initial_solver_configuration(
            "spai_epsilon_0p2_steps_8",
            ksp_type="cg",
            pc_type="spai",
            pc_spai_epsilon=0.2,
            pc_spai_nbsteps=8,
            pc_spai_maxnew=10,
        ),
        initial_solver_configuration(
            "asm_ilu_overlap_1",
            ksp_type="gmres",
            pc_type="asm",
        ),
        initial_solver_configuration(
            "bjacobi_ilu_cg_baseline",
            ksp_type="cg",
            pc_type="bjacobi",
        ),
    ]

    return space
