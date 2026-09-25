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
    space = ConfigurationSpace(name="petsc.parasails_mono", seed=seed)

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

    pc_type = Constant("pc_type", "hypre")
    pc_hypre_type = Constant(
        "pc_hypre_type",
        "parasails",
    )

    pc_hypre_parasails_nlevels = UniformIntegerHyperparameter(
        "pc_hypre_parasails_nlevels",
        lower=0,
        upper=5,
        default_value=1,
    )

    pc_hypre_parasails_thresh_mode = Categorical(
        "pc_hypre_parasails_thresh_mode",
        ["positive", "auto_drop"],
        default="positive",
        meta=petsc_meta(None, emit=False),
    )
    pc_hypre_parasails_thresh = UniformFloatHyperparameter(
        "pc_hypre_parasails_thresh",
        lower=0.0001,
        upper=0.8,
        default_value=0.4,
        log=True,
    )
    pc_hypre_parasails_thresh_auto = UniformFloatHyperparameter(
        "pc_hypre_parasails_thresh_auto",
        lower=-0.95,
        upper=-0.5,
        default_value=-0.9,
        meta=petsc_meta("-pc_hypre_parasails_thresh"),
    )

    pc_hypre_parasails_filter_mode = Categorical(
        "pc_hypre_parasails_filter_mode",
        ["positive", "zero", "auto_drop"],
        default="positive",
        meta=petsc_meta(None, emit=False),
    )
    pc_hypre_parasails_filter = UniformFloatHyperparameter(
        "pc_hypre_parasails_filter",
        lower=0.0005,
        upper=0.5,
        default_value=0.1,
        log=True,
    )
    pc_hypre_parasails_filter_zero = Constant(
        "pc_hypre_parasails_filter_zero",
        0.0,
        meta=petsc_meta("-pc_hypre_parasails_filter"),
    )
    pc_hypre_parasails_filter_auto = UniformFloatHyperparameter(
        "pc_hypre_parasails_filter_auto",
        lower=-0.95,
        upper=-0.5,
        default_value=-0.9,
        meta=petsc_meta("-pc_hypre_parasails_filter"),
    )

    pc_hypre_parasails_loadbal = UniformFloatHyperparameter(
        "pc_hypre_parasails_loadbal",
        lower=0.0,
        upper=1.0,
        default_value=0.0,
        meta=petsc_meta("-pc_hypre_parasails_loadbal", skip_values=[0.0]),
    )
    pc_hypre_parasails_reuse = Constant(
        "pc_hypre_parasails_reuse",
        False,
        meta=petsc_meta("-pc_hypre_parasails_reuse", skip_values=[False]),
    )
    pc_hypre_parasails_sym = Categorical(
        "pc_hypre_parasails_sym",
        ["nonsymmetric", "SPD", "nonsymmetric,SPD"],
        default="nonsymmetric",
        meta=petsc_meta("-pc_hypre_parasails_sym", skip_values=["nonsymmetric"]),
    )
    pc_hypre_parasails_logging = Constant(
        "pc_hypre_parasails_logging",
        False,
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
            pc_hypre_parasails_nlevels,
            pc_hypre_parasails_thresh_mode,
            pc_hypre_parasails_thresh,
            pc_hypre_parasails_thresh_auto,
            pc_hypre_parasails_filter_mode,
            pc_hypre_parasails_filter,
            pc_hypre_parasails_filter_zero,
            pc_hypre_parasails_filter_auto,
            pc_hypre_parasails_loadbal,
            pc_hypre_parasails_reuse,
            pc_hypre_parasails_sym,
            pc_hypre_parasails_logging,
        ]
    )

    space.add(
        [
            EqualsCondition(ksp_gmres_restart, ksp_type, "gmres"),
            EqualsCondition(
                pc_hypre_parasails_thresh,
                pc_hypre_parasails_thresh_mode,
                "positive",
            ),
            EqualsCondition(
                pc_hypre_parasails_thresh_auto,
                pc_hypre_parasails_thresh_mode,
                "auto_drop",
            ),
            EqualsCondition(
                pc_hypre_parasails_filter,
                pc_hypre_parasails_filter_mode,
                "positive",
            ),
            EqualsCondition(
                pc_hypre_parasails_filter_zero,
                pc_hypre_parasails_filter_mode,
                "zero",
            ),
            EqualsCondition(
                pc_hypre_parasails_filter_auto,
                pc_hypre_parasails_filter_mode,
                "auto_drop",
            ),
        ]
    )

    space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "parasails_no_filter",
            ksp_type="gmres",
            ksp_gmres_restart=20,
            pc_hypre_parasails_nlevels=0,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.0022403628029,
            pc_hypre_parasails_filter_mode="zero",
            pc_hypre_parasails_filter_zero=0.0,
            pc_hypre_parasails_loadbal=0.7172664481308,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_threshold_0p4",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.4,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.1,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_gmres_threshold_0p4",
            ksp_type="gmres",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.4,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.1,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_threshold_0p1",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.1,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.1,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_levels_0",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=0,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.4,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.1,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_levels_2_threshold_0p1",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=2,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.1,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.05,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_levels_2_threshold_0p01",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=2,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.01,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.001,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_auto_threshold",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="auto_drop",
            pc_hypre_parasails_thresh_auto=-0.9,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.05,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_auto_threshold_filter",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="auto_drop",
            pc_hypre_parasails_thresh_auto=-0.9,
            pc_hypre_parasails_filter_mode="auto_drop",
            pc_hypre_parasails_filter_auto=-0.9,
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_no_filter",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.1,
            pc_hypre_parasails_filter_mode="zero",
            pc_hypre_parasails_sym="nonsymmetric",
        ),
        initial_solver_configuration(
            "parasails_cg_spd",
            ksp_type="cg",
            pc_hypre_parasails_nlevels=1,
            pc_hypre_parasails_thresh_mode="positive",
            pc_hypre_parasails_thresh=0.4,
            pc_hypre_parasails_filter_mode="positive",
            pc_hypre_parasails_filter=0.1,
            pc_hypre_parasails_sym="SPD",
        ),
    ]

    return space
