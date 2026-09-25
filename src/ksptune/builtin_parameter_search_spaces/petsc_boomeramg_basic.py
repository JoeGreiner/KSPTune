from __future__ import annotations

from ..search_space_helpers import initial_solver_configuration

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
)
from ConfigSpace.conditions import InCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="petsc.boomeramg_basic", seed=seed)

    ksp_type = Constant("ksp_type", "gmres")
    ksp_pc_side = Constant(
        "ksp_pc_side",
        "left",
    )
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
    pc_hypre_boomeramg_coarsen_type = Categorical(
        "pc_hypre_boomeramg_coarsen_type",
        ["PMIS", "HMIS"],
        default="PMIS",
    )
    pc_hypre_boomeramg_interp_type = Categorical(
        "pc_hypre_boomeramg_interp_type",
        ["ext+i", "ext+i-cc", "ext+i-mm", "ext", "ext+e-mm", "FF"],
        default="ext+i-mm",
    )
    pc_hypre_boomeramg_relax_type_all = Categorical(
        "pc_hypre_boomeramg_relax_type_all",
        [
            "l1-Gauss-Seidel",
            "SOR/Jacobi",
            "l1scaled-Jacobi",
        ],
        default="l1-Gauss-Seidel",
    )

    pc_hypre_boomeramg_strong_threshold = UniformFloatHyperparameter(
        "pc_hypre_boomeramg_strong_threshold",
        lower=0.0001,
        upper=0.7,
        default_value=0.1,
        log=False,
    )
    pc_hypre_boomeramg_p_max = OrdinalHyperparameter(
        "pc_hypre_boomeramg_P_max",
        sequence=[0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16],
        default_value=4,
    )
    pc_hypre_boomeramg_agg_nl = OrdinalHyperparameter(
        "pc_hypre_boomeramg_agg_nl",
        sequence=[0, 1, 2],
        default_value=1,
    )
    pc_hypre_boomeramg_agg_num_paths = OrdinalHyperparameter(
        "pc_hypre_boomeramg_agg_num_paths",
        sequence=[1, 2, 3],
        default_value=1,
    )
    pc_hypre_boomeramg_truncfactor = OrdinalHyperparameter(
        "pc_hypre_boomeramg_truncfactor",
        sequence=[
            0.0,
            0.0021247466225,
            0.0024114874969,
            0.0044798272768,
            0.015826322068,
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
            ksp_norm_type,
            ksp_atol,
            ksp_rtol,
            pc_type,
            pc_hypre_type,
            pc_hypre_boomeramg_coarsen_type,
            pc_hypre_boomeramg_interp_type,
            pc_hypre_boomeramg_relax_type_all,
            pc_hypre_boomeramg_strong_threshold,
            pc_hypre_boomeramg_p_max,
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
            pc_hypre_boomeramg_P_max=4,
            pc_hypre_boomeramg_agg_nl=0,
            pc_hypre_boomeramg_strong_threshold=0.075,
            pc_hypre_boomeramg_truncfactor=0.0,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_ff_pmax_3_paths_1",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="FF",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=3,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=1,
            pc_hypre_boomeramg_strong_threshold=0.3165283876103,
            pc_hypre_boomeramg_truncfactor=0.0021247466225,
        ),
        initial_solver_configuration(
            "boomeramg_pmis_extimm_pmax_3_paths_1",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+i-mm",
            pc_hypre_boomeramg_relax_type_all="l1-Gauss-Seidel",
            pc_hypre_boomeramg_P_max=3,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=1,
            pc_hypre_boomeramg_strong_threshold=0.1828391454603,
            pc_hypre_boomeramg_truncfactor=0.025,
        ),
        initial_solver_configuration(
            "boomeramg_sor_pmis_extemm_pmax_15_paths_3",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+e-mm",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_P_max=15,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.0599966379034,
            pc_hypre_boomeramg_truncfactor=0.015826322068,
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
        initial_solver_configuration(
            "boomeramg_sor_pmis_exti_pmax_0_paths_3",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext+i",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_P_max=0,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.063,
            pc_hypre_boomeramg_truncfactor=0.125,
        ),
        initial_solver_configuration(
            "boomeramg_sor_pmis_ext_pmax_12_paths_3",
            pc_hypre_boomeramg_coarsen_type="PMIS",
            pc_hypre_boomeramg_interp_type="ext",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_P_max=12,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.06,
            pc_hypre_boomeramg_truncfactor=0.025,
        ),
        initial_solver_configuration(
            "boomeramg_sor_hmis_extemm_pmax_10_paths_3",
            pc_hypre_boomeramg_coarsen_type="HMIS",
            pc_hypre_boomeramg_interp_type="ext+e-mm",
            pc_hypre_boomeramg_relax_type_all="SOR/Jacobi",
            pc_hypre_boomeramg_P_max=10,
            pc_hypre_boomeramg_agg_nl=1,
            pc_hypre_boomeramg_agg_num_paths=3,
            pc_hypre_boomeramg_strong_threshold=0.06,
            pc_hypre_boomeramg_truncfactor=0.025,
        ),
    ]
    return parameter_search_space
