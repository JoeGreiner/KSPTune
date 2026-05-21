from __future__ import annotations

from ConfigSpace import Categorical, ConfigurationSpace, Constant, OrdinalHyperparameter
from ConfigSpace.conditions import InCondition


def initial_solver_configuration(label: str, **overrides):
    return {"label": label, "solver_configuration": dict(overrides)}


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.hypre-basic", seed=seed)

    ksp_type = Constant("ksp_type", "fgmres", meta={"petsc_option": "-ksp_type"})
    ksp_pc_side = Constant(
        "ksp_pc_side",
        "right",
        meta={"petsc_option": "-ksp_pc_side"},
    )

    pc_type = Constant("pc_type", "hypre", meta={"petsc_option": "-pc_type"})
    pc_hypre_type = Constant(
        "pc_hypre_type",
        "boomeramg",
        meta={"petsc_option": "-pc_hypre_type"},
    )
    pc_hypre_boomeramg_coarsen_type = Categorical(
        "pc_hypre_boomeramg_coarsen_type",
        ["PMIS", "HMIS"],
        default="PMIS",
        meta={"petsc_option": "-pc_hypre_boomeramg_coarsen_type"},
    )
    pc_hypre_boomeramg_interp_type = Categorical(
        "pc_hypre_boomeramg_interp_type",
        ["ext+i", "ext+i-cc", "ext+i-mm", "FF"],
        default="ext+i-mm",
        meta={"petsc_option": "-pc_hypre_boomeramg_interp_type"},
    )
    pc_hypre_boomeramg_relax_type_all = Categorical(
        "pc_hypre_boomeramg_relax_type_all",
        [
            "l1-Gauss-Seidel",
            "l1scaled-Jacobi",
        ],
        default="l1-Gauss-Seidel",
        meta={"petsc_option": "-pc_hypre_boomeramg_relax_type_all"},
    )

    pc_hypre_boomeramg_strong_threshold = OrdinalHyperparameter(
        "pc_hypre_boomeramg_strong_threshold",
        sequence=[
            0.025,
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.175,
            0.2,
            0.225,
            0.25,
            0.275,
            0.3,
            0.325,
            0.35,
            0.375,
            0.4,
            0.425,
            0.45,
            0.475,
            0.5,
            0.525,
            0.55,
            0.575,
            0.6,
            0.625,
            0.65,
            0.675,
            0.7,
        ],
        default_value=0.1,
        meta={"petsc_option": "-pc_hypre_boomeramg_strong_threshold"},
    )
    pc_hypre_boomeramg_P_max = OrdinalHyperparameter(
        "pc_hypre_boomeramg_P_max",
        sequence=[2, 3, 4, 5, 6, 8],
        default_value=4,
        meta={"petsc_option": "-pc_hypre_boomeramg_P_max"},
    )
    pc_hypre_boomeramg_agg_nl = OrdinalHyperparameter(
        "pc_hypre_boomeramg_agg_nl",
        sequence=[0, 1, 2],
        default_value=0,
        meta={"petsc_option": "-pc_hypre_boomeramg_agg_nl"},
    )
    pc_hypre_boomeramg_agg_num_paths = OrdinalHyperparameter(
        "pc_hypre_boomeramg_agg_num_paths",
        sequence=[1, 2, 3],
        default_value=1,
        meta={"petsc_option": "-pc_hypre_boomeramg_agg_num_paths"},
    )
    pc_hypre_boomeramg_truncfactor = OrdinalHyperparameter(
        "pc_hypre_boomeramg_truncfactor",
        sequence=[
            0.0,
            0.025,
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.175,
            0.2,
            0.225,
            0.25,
            0.275,
            0.3,
            0.325,
            0.35,
            0.375,
            0.4,
        ],
        default_value=0.0,
        meta={
            "petsc_option": "-pc_hypre_boomeramg_truncfactor",
            "skip_values": [0.0],
        },
    )
    parameter_search_space.add(
        [
            ksp_type,
            ksp_pc_side,
            pc_type,
            pc_hypre_type,
            pc_hypre_boomeramg_coarsen_type,
            pc_hypre_boomeramg_interp_type,
            pc_hypre_boomeramg_relax_type_all,
            pc_hypre_boomeramg_strong_threshold,
            pc_hypre_boomeramg_P_max,
            pc_hypre_boomeramg_agg_nl,
            pc_hypre_boomeramg_agg_num_paths,
            pc_hypre_boomeramg_truncfactor,
        ]
    )
    parameter_search_space.add(
        InCondition(
            pc_hypre_boomeramg_agg_num_paths,
            pc_hypre_boomeramg_agg_nl,
            [1, 2],
        )
    )
    parameter_search_space._ksptuneinitial_solver_configurations = [
        initial_solver_configuration(
            "joe_hypre_opts8",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_strong_threshold=0.05,
        ),
        initial_solver_configuration(
            "joe_hypre_opts9_projected",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.25,
        ),
        initial_solver_configuration(
            "joe_hypre_opts4_projected",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_P_max=3,
            pc_hypre_boomeramg_strong_threshold=0.175,
            pc_hypre_boomeramg_truncfactor=0.225,
        ),
        initial_solver_configuration(
            "joe_hypre_opts3_projected",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_P_max=8,
            pc_hypre_boomeramg_strong_threshold=0.25,
            pc_hypre_boomeramg_truncfactor=0.05,
        ),
    ]
    return parameter_search_space
