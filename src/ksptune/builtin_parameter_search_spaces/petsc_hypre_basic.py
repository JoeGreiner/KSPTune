from __future__ import annotations

from ConfigSpace import Categorical, ConfigurationSpace, Constant, OrdinalHyperparameter


def initial_solver_configuration(label: str, **overrides):
    solver_configuration = {
        "ksp_type": "fgmres",
        "ksp_pc_side": "right",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_tol": 0.0,
        "pc_hypre_boomeramg_max_iter": 1,
        "pc_hypre_boomeramg_interp_refine": 0,
        "pc_hypre_boomeramg_cycle_type": "V",
        "pc_hypre_boomeramg_measure_type": "local",
        "pc_hypre_boomeramg_coarsen_type": "PMIS",
        "pc_hypre_boomeramg_interp_type": "ext+i-cc",
        "pc_hypre_boomeramg_relax_type_all": "l1-Gauss-Seidel",
        "pc_hypre_boomeramg_relax_type_coarse": "l1scaled-Jacobi",
        "pc_hypre_boomeramg_strong_threshold": 0.075,
        "pc_hypre_boomeramg_P_max": 4,
        "pc_hypre_boomeramg_agg_nl": 0,
        "pc_hypre_boomeramg_agg_num_paths": 0,
        "pc_hypre_boomeramg_nodal_coarsen": 3,
        "pc_hypre_boomeramg_truncfactor": 0.0,
        "pc_hypre_boomeramg_relax_weight_all": 1.0,
    }
    solver_configuration.update(overrides)
    return {"label": label, "solver_configuration": solver_configuration}


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
    pc_hypre_boomeramg_tol = Constant(
        "pc_hypre_boomeramg_tol",
        0.0,
        meta={"petsc_option": "-pc_hypre_boomeramg_tol"},
    )
    pc_hypre_boomeramg_max_iter = Constant(
        "pc_hypre_boomeramg_max_iter",
        1,
        meta={"petsc_option": "-pc_hypre_boomeramg_max_iter"},
    )
    pc_hypre_boomeramg_interp_refine = Constant(
        "pc_hypre_boomeramg_interp_refine",
        0,
        meta={"petsc_option": "-pc_hypre_boomeramg_interp_refine"},
    )

    pc_hypre_boomeramg_cycle_type = Categorical(
        "pc_hypre_boomeramg_cycle_type",
        ["V", "W"],
        default="V",
        meta={"petsc_option": "-pc_hypre_boomeramg_cycle_type"},
    )
    pc_hypre_boomeramg_measure_type = Categorical(
        "pc_hypre_boomeramg_measure_type",
        ["local", "global"],
        default="local",
        meta={"petsc_option": "-pc_hypre_boomeramg_measure_type"},
    )
    pc_hypre_boomeramg_coarsen_type = Categorical(
        "pc_hypre_boomeramg_coarsen_type",
        ["PMIS", "HMIS", "Falgout"],
        default="PMIS",
        meta={"petsc_option": "-pc_hypre_boomeramg_coarsen_type"},
    )
    pc_hypre_boomeramg_interp_type = Categorical(
        "pc_hypre_boomeramg_interp_type",
        ["ext+i", "ext+i-cc", "ext+i-mm", "FF"],
        default="ext+i-cc",
        meta={"petsc_option": "-pc_hypre_boomeramg_interp_type"},
    )
    pc_hypre_boomeramg_relax_type_all = Categorical(
        "pc_hypre_boomeramg_relax_type_all",
        [
            "l1-Gauss-Seidel",
            "l1scaled-Jacobi",
            "symmetric-SOR/Jacobi",
            "Jacobi",
            "FCF-Jacobi",
        ],
        default="l1-Gauss-Seidel",
        meta={"petsc_option": "-pc_hypre_boomeramg_relax_type_all"},
    )
    pc_hypre_boomeramg_relax_type_coarse = Categorical(
        "pc_hypre_boomeramg_relax_type_coarse",
        ["", "Gaussian-elimination", "Jacobi", "l1scaled-Jacobi", "Chebyshev"],
        default="l1scaled-Jacobi",
        meta={"petsc_option": "-pc_hypre_boomeramg_relax_type_coarse"},
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
        default_value=0.075,
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
        sequence=[0, 1, 2, 4],
        default_value=0,
        meta={"petsc_option": "-pc_hypre_boomeramg_agg_nl"},
    )
    pc_hypre_boomeramg_agg_num_paths = OrdinalHyperparameter(
        "pc_hypre_boomeramg_agg_num_paths",
        sequence=[0, 1, 2, 3],
        default_value=0,
        meta={
            "petsc_option": "-pc_hypre_boomeramg_agg_num_paths",
            "skip_values": [0],
        },
    )
    pc_hypre_boomeramg_nodal_coarsen = OrdinalHyperparameter(
        "pc_hypre_boomeramg_nodal_coarsen",
        sequence=[0, 1, 2, 3, 4, 6],
        default_value=3,
        meta={"petsc_option": "-pc_hypre_boomeramg_nodal_coarsen"},
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
    pc_hypre_boomeramg_relax_weight_all = OrdinalHyperparameter(
        "pc_hypre_boomeramg_relax_weight_all",
        sequence=[0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0],
        default_value=1.0,
        meta={
            "petsc_option": "-pc_hypre_boomeramg_relax_weight_all",
            "skip_values": [1.0],
        },
    )

    parameter_search_space.add(
        [
            ksp_type,
            ksp_pc_side,
            pc_type,
            pc_hypre_type,
            pc_hypre_boomeramg_tol,
            pc_hypre_boomeramg_max_iter,
            pc_hypre_boomeramg_interp_refine,
            pc_hypre_boomeramg_cycle_type,
            pc_hypre_boomeramg_measure_type,
            pc_hypre_boomeramg_coarsen_type,
            pc_hypre_boomeramg_interp_type,
            pc_hypre_boomeramg_relax_type_all,
            pc_hypre_boomeramg_relax_type_coarse,
            pc_hypre_boomeramg_strong_threshold,
            pc_hypre_boomeramg_P_max,
            pc_hypre_boomeramg_agg_nl,
            pc_hypre_boomeramg_agg_num_paths,
            pc_hypre_boomeramg_nodal_coarsen,
            pc_hypre_boomeramg_truncfactor,
            pc_hypre_boomeramg_relax_weight_all,
        ]
    )
    parameter_search_space._ksptuneinitial_solver_configurations = [
        initial_solver_configuration(
            "joe_hypre_opts8",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_coarse="Jacobi",
            pc_hypre_boomeramg_strong_threshold=0.05,
        ),
        initial_solver_configuration(
            "joe_hypre_opts9_projected",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_relax_type_coarse="Gaussian-elimination",
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.25,
        ),
        initial_solver_configuration(
            "joe_hypre_opts10",
            pc_hypre_boomeramg_relax_type_coarse="Chebyshev",
        ),
        initial_solver_configuration(
            "joe_hypre_opts6_projected",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_interp_type="ext+i-mm",
            pc_hypre_boomeramg_relax_type_coarse="",
            pc_hypre_boomeramg_agg_nl=4,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.1,
        ),
        initial_solver_configuration(
            "joe_hypre_opts4_projected",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_coarse="",
            pc_hypre_boomeramg_P_max=3,
            pc_hypre_boomeramg_strong_threshold=0.175,
            pc_hypre_boomeramg_truncfactor=0.225,
        ),
        initial_solver_configuration(
            "joe_hypre_opts5_projected",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_all="Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="",
            pc_hypre_boomeramg_P_max=3,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.175,
            pc_hypre_boomeramg_relax_weight_all=0.875,
        ),
        initial_solver_configuration(
            "joe_hypre_opts3_projected",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_coarse="",
            pc_hypre_boomeramg_P_max=8,
            pc_hypre_boomeramg_strong_threshold=0.25,
            pc_hypre_boomeramg_truncfactor=0.05,
        ),
        initial_solver_configuration(
            "joe_hypre_opts2_projected",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_all="symmetric-SOR/Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="",
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_strong_threshold=0.125,
        ),
    ]
    return parameter_search_space
