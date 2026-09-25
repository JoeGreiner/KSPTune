from __future__ import annotations

from ..search_space_helpers import bool_choice, initial_solver_configuration, petsc_meta

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition, GreaterThanCondition


def hidden_bool_choice(name: str, default: bool = False):
    return Categorical(
        name,
        [False, True],
        default=default,
        meta=petsc_meta(None, emit=False),
    )


# Normal PETSc/Hypre BoomerAMG relaxers, trimmed away from the most crash-prone
# coarse-grid/Krylov relaxers for this small monodomain replay case.
RELAX_TYPES = [
    "Jacobi",
    "sequential-Gauss-Seidel",
    "seqboundary-Gauss-Seidel",
    "SOR/Jacobi",
    "backward-SOR/Jacobi",
    "symmetric-SOR/Jacobi",
    "l1scaled-SOR/Jacobi",
    "l1scaled-Jacobi",
    "l1-Gauss-Seidel",
    "backward-l1-Gauss-Seidel",
]

RELAX_TYPES_ALL = [relax_type for relax_type in RELAX_TYPES if relax_type != "l1scaled-Jacobi"]


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.boomeramg_extended_mono", seed=seed)

    ksp_type = Constant("ksp_type", "gmres")
    ksp_pc_side = Constant("ksp_pc_side", "left")
    ksp_norm_type = Constant(
        "ksp_norm_type",
        "preconditioned",
    )
    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-8)
    ksp_gmres_restart = Categorical(
        "ksp_gmres_restart",
        [10, 20, 30, 50, 80, 100],
        default=30,
        meta=petsc_meta("-ksp_gmres_restart", skip_values=[30]),
    )

    pc_type = Constant("pc_type", "hypre")
    pc_hypre_type = Constant(
        "pc_hypre_type",
        "boomeramg",
    )
    pc_hypre_boomeramg_cycle_type = Categorical(
        "pc_hypre_boomeramg_cycle_type",
        ["V", "W"],
        default="V",
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
        [
            "CLJP",
            "Ruge-Stueben",
            "modifiedRuge-Stueben",
            "Falgout",
            "PMIS",
            "HMIS",
        ],
        default="HMIS",
    )
    pc_hypre_boomeramg_interp_type = Categorical(
        "pc_hypre_boomeramg_interp_type",
        [
            "classical",
            "direct",
            "multipass",
            "multipass-wts",
            "ext+i",
            "ext+i-cc",
            "standard",
            "standard-wts",
            "FF",
            "FF1",
            "ext",
            "ad-wts",
            "ext-mm",
            "ext+i-mm",
            "ext+e-mm",
        ],
        default="ext",
    )
    pc_hypre_boomeramg_relax_type_all = Categorical(
        "pc_hypre_boomeramg_relax_type_all",
        RELAX_TYPES_ALL,
        default="SOR/Jacobi",
    )
    pc_hypre_boomeramg_split_relax_enabled = hidden_bool_choice(
        "pc_hypre_boomeramg_split_relax_enabled",
    )
    pc_hypre_boomeramg_relax_type_down = Categorical(
        "pc_hypre_boomeramg_relax_type_down",
        RELAX_TYPES,
        default="SOR/Jacobi",
    )
    pc_hypre_boomeramg_relax_type_up = Categorical(
        "pc_hypre_boomeramg_relax_type_up",
        RELAX_TYPES,
        default="SOR/Jacobi",
    )
    pc_hypre_boomeramg_relax_type_coarse = Categorical(
        "pc_hypre_boomeramg_relax_type_coarse",
        RELAX_TYPES,
        default="Jacobi",
    )
    pc_hypre_boomeramg_measure_type = Categorical(
        "pc_hypre_boomeramg_measure_type",
        ["local", "global"],
        default="local",
    )

    pc_hypre_boomeramg_strong_threshold = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_strong_threshold",
        lower=0.0,
        upper=1.0,
        default_value=0.16,
    )
    pc_hypre_boomeramg_truncfactor = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_truncfactor",
        lower=0.0,
        upper=0.95,
        default_value=0.4,
        meta=petsc_meta("-pc_hypre_boomeramg_truncfactor", skip_values=[0.0]),
    )
    pc_hypre_boomeramg_max_row_sum = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_max_row_sum",
        lower=0.0,
        upper=0.85,
        default_value=0.6,
    )
    pc_hypre_boomeramg_p_max = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_P_max",
        lower=0,
        upper=64,
        default_value=5,
    )
    pc_hypre_boomeramg_agg_nl = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_agg_nl",
        lower=0,
        upper=8,
        default_value=2,
    )
    pc_hypre_boomeramg_agg_num_paths = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_agg_num_paths",
        lower=1,
        upper=8,
        default_value=2,
    )

    pc_hypre_boomeramg_max_levels = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_max_levels",
        lower=8,
        upper=80,
        default_value=25,
    )
    pc_hypre_boomeramg_max_coarse_size = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_max_coarse_size",
        lower=64,
        upper=4096,
        default_value=256,
    )
    pc_hypre_boomeramg_min_coarse_size = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_min_coarse_size",
        lower=1,
        upper=64,
        default_value=1,
    )
    pc_hypre_boomeramg_grid_sweeps_down = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_grid_sweeps_down",
        lower=1,
        upper=4,
        default_value=1,
    )
    pc_hypre_boomeramg_grid_sweeps_up = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_grid_sweeps_up",
        lower=1,
        upper=4,
        default_value=1,
    )
    pc_hypre_boomeramg_grid_sweeps_coarse = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_grid_sweeps_coarse",
        lower=1,
        upper=4,
        default_value=1,
    )

    pc_hypre_boomeramg_interp_refine = UniformIntegerHyperparameter(
        "pc_hypre_boomeramg_interp_refine",
        lower=0,
        upper=3,
        default_value=0,
        meta=petsc_meta("-pc_hypre_boomeramg_interp_refine", skip_values=[0]),
    )
    pc_hypre_boomeramg_no_CF = bool_choice(
        "pc_hypre_boomeramg_no_CF", False, "-pc_hypre_boomeramg_no_CF", skip_false=True
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
            ksp_gmres_restart,
            pc_type,
            pc_hypre_type,
            pc_hypre_boomeramg_cycle_type,
            pc_hypre_boomeramg_max_iter,
            pc_hypre_boomeramg_tol,
            pc_hypre_boomeramg_coarsen_type,
            pc_hypre_boomeramg_interp_type,
            pc_hypre_boomeramg_relax_type_all,
            pc_hypre_boomeramg_split_relax_enabled,
            pc_hypre_boomeramg_relax_type_down,
            pc_hypre_boomeramg_relax_type_up,
            pc_hypre_boomeramg_relax_type_coarse,
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
            pc_hypre_boomeramg_interp_refine,
            pc_hypre_boomeramg_no_CF,
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
            EqualsCondition(
                pc_hypre_boomeramg_relax_type_down,
                pc_hypre_boomeramg_split_relax_enabled,
                True,
            ),
            EqualsCondition(
                pc_hypre_boomeramg_relax_type_up,
                pc_hypre_boomeramg_split_relax_enabled,
                True,
            ),
            EqualsCondition(
                pc_hypre_boomeramg_relax_type_coarse,
                pc_hypre_boomeramg_split_relax_enabled,
                True,
            ),
        ]
    )

    parameter_search_space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "boomeramg_hmis_ext",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_P_max=5,
            pc_hypre_boomeramg_agg_nl=2,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.1629247287282,
            pc_hypre_boomeramg_truncfactor=0.4,
        ),
        initial_solver_configuration(
            "boomeramg_hmis_ext_high_truncation",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_P_max=5,
            pc_hypre_boomeramg_agg_nl=3,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.16,
            pc_hypre_boomeramg_truncfactor=0.65,
        ),
        initial_solver_configuration(
            "boomeramg_modified_rs_standard_wts",
            ksp_gmres_restart=10,
            pc_hypre_boomeramg_coarsen_type="modifiedRuge-Stueben",
            pc_hypre_boomeramg_interp_type="standard-wts",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_P_max=50,
            pc_hypre_boomeramg_agg_nl=0,
            pc_hypre_boomeramg_strong_threshold=0.4768892823096,
            pc_hypre_boomeramg_truncfactor=0.4264672411872,
            pc_hypre_boomeramg_max_coarse_size=242,
            pc_hypre_boomeramg_min_coarse_size=34,
            pc_hypre_boomeramg_grid_sweeps_up=3,
            pc_hypre_boomeramg_interp_refine=2,
            pc_hypre_boomeramg_max_levels=43,
            pc_hypre_boomeramg_max_row_sum=0.717936395637,
            pc_hypre_boomeramg_measure_type="global",
            pc_hypre_boomeramg_no_CF=True,
        ),
        initial_solver_configuration(
            "boomeramg_rs_multipass_wts",
            ksp_gmres_restart=100,
            pc_hypre_boomeramg_coarsen_type="Ruge-Stueben",
            pc_hypre_boomeramg_interp_type="multipass-wts",
            pc_hypre_boomeramg_relax_type_all="backward-l1-Gauss-Seidel",
            pc_hypre_boomeramg_split_relax_enabled=True,
            pc_hypre_boomeramg_relax_type_down="Jacobi",
            pc_hypre_boomeramg_relax_type_up="l1scaled-Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="SOR/Jacobi",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_P_max=46,
            pc_hypre_boomeramg_agg_nl=7,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.0564262284286,
            pc_hypre_boomeramg_truncfactor=0.0739439026114,
            pc_hypre_boomeramg_max_coarse_size=64,
            pc_hypre_boomeramg_min_coarse_size=4,
            pc_hypre_boomeramg_grid_sweeps_coarse=3,
            pc_hypre_boomeramg_grid_sweeps_down=4,
            pc_hypre_boomeramg_grid_sweeps_up=3,
            pc_hypre_boomeramg_interp_refine=1,
            pc_hypre_boomeramg_max_levels=72,
            pc_hypre_boomeramg_max_row_sum=0.4504900885569,
            pc_hypre_boomeramg_measure_type="local",
            pc_hypre_boomeramg_no_CF=True,
            pc_hypre_boomeramg_keeptranspose=True,
        ),
        initial_solver_configuration(
            "boomeramg_falgout_extemm",
            ksp_gmres_restart=50,
            pc_hypre_boomeramg_coarsen_type="Falgout",
            pc_hypre_boomeramg_interp_type="ext+e-mm",
            pc_hypre_boomeramg_relax_type_all="backward-l1-Gauss-Seidel",
            pc_hypre_boomeramg_cycle_type="V",
            pc_hypre_boomeramg_P_max=0,
            pc_hypre_boomeramg_agg_nl=8,
            pc_hypre_boomeramg_agg_num_paths=4,
            pc_hypre_boomeramg_strong_threshold=0.7402699424911,
            pc_hypre_boomeramg_truncfactor=0.1015163419005,
            pc_hypre_boomeramg_max_coarse_size=2251,
            pc_hypre_boomeramg_min_coarse_size=52,
            pc_hypre_boomeramg_grid_sweeps_coarse=4,
            pc_hypre_boomeramg_grid_sweeps_down=2,
            pc_hypre_boomeramg_grid_sweeps_up=2,
            pc_hypre_boomeramg_max_levels=50,
            pc_hypre_boomeramg_max_row_sum=0.6560041960572,
            pc_hypre_boomeramg_measure_type="local",
            pc_hypre_boomeramg_keeptranspose=True,
        ),
        initial_solver_configuration(
            "boomeramg_falgout_extimm_w_cycle",
            ksp_gmres_restart=10,
            pc_hypre_boomeramg_coarsen_type="Falgout",
            pc_hypre_boomeramg_interp_type="ext+i-mm",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_split_relax_enabled=True,
            pc_hypre_boomeramg_relax_type_down="Jacobi",
            pc_hypre_boomeramg_relax_type_up="SOR/Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="backward-SOR/Jacobi",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_P_max=56,
            pc_hypre_boomeramg_agg_nl=5,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.2632748437506,
            pc_hypre_boomeramg_truncfactor=0.3771198729468,
            pc_hypre_boomeramg_max_coarse_size=3913,
            pc_hypre_boomeramg_min_coarse_size=47,
            pc_hypre_boomeramg_grid_sweeps_coarse=4,
            pc_hypre_boomeramg_grid_sweeps_down=2,
            pc_hypre_boomeramg_grid_sweeps_up=1,
            pc_hypre_boomeramg_max_levels=32,
            pc_hypre_boomeramg_max_row_sum=0.7430038909048,
            pc_hypre_boomeramg_measure_type="local",
            pc_hypre_boomeramg_no_CF=True,
        ),
        initial_solver_configuration(
            "boomeramg_falgout_ext_l1scaled_jacobi",
            ksp_gmres_restart=10,
            pc_hypre_boomeramg_coarsen_type="Falgout",
            pc_hypre_boomeramg_interp_type="ext",
            pc_hypre_boomeramg_relax_type_all="backward-SOR/Jacobi",
            pc_hypre_boomeramg_split_relax_enabled=True,
            pc_hypre_boomeramg_relax_type_down="l1scaled-Jacobi",
            pc_hypre_boomeramg_relax_type_up="Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="l1scaled-Jacobi",
            pc_hypre_boomeramg_cycle_type="V",
            pc_hypre_boomeramg_P_max=39,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=4,
            pc_hypre_boomeramg_strong_threshold=0.3348891450252,
            pc_hypre_boomeramg_truncfactor=0.0778117187203,
            pc_hypre_boomeramg_max_coarse_size=2066,
            pc_hypre_boomeramg_min_coarse_size=60,
            pc_hypre_boomeramg_grid_sweeps_coarse=3,
            pc_hypre_boomeramg_grid_sweeps_down=1,
            pc_hypre_boomeramg_grid_sweeps_up=3,
            pc_hypre_boomeramg_interp_refine=3,
            pc_hypre_boomeramg_max_levels=73,
            pc_hypre_boomeramg_max_row_sum=0.1237572559207,
            pc_hypre_boomeramg_measure_type="global",
            pc_hypre_boomeramg_no_CF=True,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_extemm",
            ksp_gmres_restart=10,
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+e-mm",
            pc_hypre_boomeramg_relax_type_all="backward-SOR/Jacobi",
            pc_hypre_boomeramg_split_relax_enabled=True,
            pc_hypre_boomeramg_relax_type_down="backward-l1-Gauss-Seidel",
            pc_hypre_boomeramg_relax_type_up="symmetric-SOR/Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="SOR/Jacobi",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_P_max=31,
            pc_hypre_boomeramg_agg_nl=6,
            pc_hypre_boomeramg_agg_num_paths=5,
            pc_hypre_boomeramg_strong_threshold=0.1881073843669,
            pc_hypre_boomeramg_truncfactor=0.4317450185581,
            pc_hypre_boomeramg_max_coarse_size=64,
            pc_hypre_boomeramg_min_coarse_size=11,
            pc_hypre_boomeramg_grid_sweeps_coarse=4,
            pc_hypre_boomeramg_grid_sweeps_down=1,
            pc_hypre_boomeramg_grid_sweeps_up=2,
            pc_hypre_boomeramg_interp_refine=1,
            pc_hypre_boomeramg_max_levels=52,
            pc_hypre_boomeramg_max_row_sum=0.7556744010757,
            pc_hypre_boomeramg_measure_type="global",
            pc_hypre_boomeramg_no_CF=True,
        ),
        initial_solver_configuration(
            "boomeramg_falgout_extimm_v_cycle",
            ksp_gmres_restart=80,
            pc_hypre_boomeramg_coarsen_type="Falgout",
            pc_hypre_boomeramg_interp_type="ext+i-mm",
            pc_hypre_boomeramg_relax_type_all="backward-l1-Gauss-Seidel",
            pc_hypre_boomeramg_split_relax_enabled=True,
            pc_hypre_boomeramg_relax_type_down="backward-l1-Gauss-Seidel",
            pc_hypre_boomeramg_relax_type_up="l1scaled-Jacobi",
            pc_hypre_boomeramg_relax_type_coarse="sequential-Gauss-Seidel",
            pc_hypre_boomeramg_cycle_type="V",
            pc_hypre_boomeramg_P_max=51,
            pc_hypre_boomeramg_agg_nl=6,
            pc_hypre_boomeramg_agg_num_paths=6,
            pc_hypre_boomeramg_strong_threshold=0.4109735564926,
            pc_hypre_boomeramg_truncfactor=0.36365518536,
            pc_hypre_boomeramg_max_coarse_size=3598,
            pc_hypre_boomeramg_min_coarse_size=48,
            pc_hypre_boomeramg_grid_sweeps_coarse=3,
            pc_hypre_boomeramg_grid_sweeps_down=1,
            pc_hypre_boomeramg_grid_sweeps_up=2,
            pc_hypre_boomeramg_interp_refine=3,
            pc_hypre_boomeramg_max_levels=18,
            pc_hypre_boomeramg_max_row_sum=0.0409072782595,
            pc_hypre_boomeramg_measure_type="local",
        ),
        initial_solver_configuration(
            "boomeramg_cljp_ff",
            ksp_gmres_restart=20,
            pc_hypre_boomeramg_coarsen_type="CLJP",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_relax_type_all="backward-l1-Gauss-Seidel",
            pc_hypre_boomeramg_split_relax_enabled=True,
            pc_hypre_boomeramg_relax_type_down="l1scaled-SOR/Jacobi",
            pc_hypre_boomeramg_relax_type_up="sequential-Gauss-Seidel",
            pc_hypre_boomeramg_relax_type_coarse="Jacobi",
            pc_hypre_boomeramg_cycle_type="W",
            pc_hypre_boomeramg_P_max=16,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=7,
            pc_hypre_boomeramg_strong_threshold=0.2813738639694,
            pc_hypre_boomeramg_truncfactor=0.9219582490265,
            pc_hypre_boomeramg_max_coarse_size=635,
            pc_hypre_boomeramg_min_coarse_size=39,
            pc_hypre_boomeramg_grid_sweeps_coarse=1,
            pc_hypre_boomeramg_grid_sweeps_down=1,
            pc_hypre_boomeramg_grid_sweeps_up=3,
            pc_hypre_boomeramg_interp_refine=3,
            pc_hypre_boomeramg_max_levels=30,
            pc_hypre_boomeramg_max_row_sum=0.4091473324185,
            pc_hypre_boomeramg_measure_type="global",
            pc_hypre_boomeramg_no_CF=True,
        ),
        initial_solver_configuration(
            "boomeramg_hmis_exti",
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
            "boomeramg_pmis_ff_no_truncation",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=4,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=2,
            pc_hypre_boomeramg_strong_threshold=0.25,
            pc_hypre_boomeramg_truncfactor=0.0,
        ),
    ]
    return parameter_search_space
