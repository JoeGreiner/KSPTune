from __future__ import annotations

import pytest

from ksptune.parameter_search_spaces import (
    build_configspace_from_parameter_search_space,
    list_builtin_parameter_search_spaces,
    load_parameter_search_space,
    parameter_search_space_to_yaml,
    validate_parameter_search_space,
    write_configspace_yaml,
)
from ksptune.petsc_options import render_petsc_options_from_solver_configuration
from ksptune.solver_configurations import default_solver_configuration_from_parameter_search_space


def test_builtin_parameter_search_spaces_are_loadable() -> None:
    names = list_builtin_parameter_search_spaces()
    assert names == ["petsc.hypre-basic"]

    parameter_search_space = load_parameter_search_space("petsc.hypre-basic")
    configuration_space = build_configspace_from_parameter_search_space(parameter_search_space)

    assert len(configuration_space) > 0
    assert "hyperparameters:" in parameter_search_space_to_yaml(parameter_search_space)


def test_petsc_hypre_basic_defaults_match_joe_hypre_opts12_baseline() -> None:
    parameter_search_space = load_parameter_search_space("petsc.hypre-basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )

    expected_values = {
        "ksp_type": "fgmres",
        "ksp_pc_side": "right",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_max_iter": 1,
        "pc_hypre_boomeramg_tol": 0.0,
        "pc_hypre_boomeramg_cycle_type": "V",
        "pc_hypre_boomeramg_measure_type": "local",
        "pc_hypre_boomeramg_interp_refine": 0,
        "pc_hypre_boomeramg_coarsen_type": "PMIS",
        "pc_hypre_boomeramg_interp_type": "ext+i-cc",
        "pc_hypre_boomeramg_relax_type_all": "l1-Gauss-Seidel",
        "pc_hypre_boomeramg_relax_type_coarse": "l1scaled-Jacobi",
        "pc_hypre_boomeramg_P_max": 4,
        "pc_hypre_boomeramg_agg_nl": 0,
        "pc_hypre_boomeramg_nodal_coarsen": 3,
        "pc_hypre_boomeramg_truncfactor": 0.0,
        "pc_hypre_boomeramg_strong_threshold": 0.075,
    }
    for name, value in expected_values.items():
        assert solver_configuration[name] == value

    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    assert "-pc_type" in options
    assert "hypre" in options
    assert "-pc_hypre_boomeramg_P_max" in options
    assert "4" in options
    assert "-pc_hypre_boomeramg_interp_type" in options
    assert "ext+i-cc" in options
    assert "-pc_hypre_boomeramg_nodal_coarsen" in options
    assert "3" in options
    assert "-pc_hypre_boomeramg_truncfactor" not in options
    assert not parameter_search_space["ksp_type"].legal_value("cg")


def test_petsc_hypre_basic_contains_joe_hypre_option_ranges() -> None:
    parameter_search_space = load_parameter_search_space("petsc.hypre-basic")

    categorical_joe_hypre_option_values = {
        "pc_hypre_boomeramg_coarsen_type": ["PMIS", "HMIS"],
        "pc_hypre_boomeramg_interp_type": ["ext+i", "ext+i-cc", "ext+i-mm", "FF"],
        "pc_hypre_boomeramg_relax_type_all": [
            "l1-Gauss-Seidel",
            "symmetric-SOR/Jacobi",
            "Jacobi",
            "FCF-Jacobi",
        ],
        "pc_hypre_boomeramg_relax_type_coarse": [
            "Gaussian-elimination",
            "Jacobi",
            "l1scaled-Jacobi",
            "Chebyshev",
        ],
        "pc_hypre_boomeramg_P_max": [3, 4, 8],
        "pc_hypre_boomeramg_agg_nl": [0, 1, 4],
        "pc_hypre_boomeramg_agg_num_paths": [2, 3],
        "pc_hypre_boomeramg_nodal_coarsen": [3],
    }
    for parameter_name, values in categorical_joe_hypre_option_values.items():
        for value in values:
            assert parameter_search_space[parameter_name].legal_value(value)

    strong_threshold_values = list(
        parameter_search_space["pc_hypre_boomeramg_strong_threshold"].sequence
    )
    truncfactor_values = list(parameter_search_space["pc_hypre_boomeramg_truncfactor"].sequence)
    relax_weight_values = list(
        parameter_search_space["pc_hypre_boomeramg_relax_weight_all"].sequence
    )

    assert strong_threshold_values == [round(index * 0.025, 3) for index in range(1, 29)]
    assert truncfactor_values == [round(index * 0.025, 3) for index in range(0, 17)]
    assert relax_weight_values == [round(0.8 + index * 0.025, 3) for index in range(0, 9)]


def test_petsc_options_render_readable_float_values() -> None:
    parameter_search_space = load_parameter_search_space("petsc.hypre-basic")
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    solver_configuration.update(
        {
            "pc_hypre_boomeramg_relax_weight_all": 0.825,
            "pc_hypre_boomeramg_strong_threshold": 0.1,
            "pc_hypre_boomeramg_truncfactor": 0.05,
        }
    )

    options_text = " ".join(
        render_petsc_options_from_solver_configuration(
            parameter_search_space,
            solver_configuration,
        )
    )

    assert "-pc_hypre_boomeramg_relax_weight_all 0.825" in options_text
    assert "-pc_hypre_boomeramg_strong_threshold 0.1" in options_text
    assert "-pc_hypre_boomeramg_truncfactor 0.05" in options_text
    assert "0.82499999999999996" not in options_text
    assert "0.10000000000000001" not in options_text
    assert "0.050000000000000003" not in options_text


def test_inactive_parameters_are_not_rendered() -> None:
    from ConfigSpace import Categorical, ConfigurationSpace, Constant
    from ConfigSpace.conditions import EqualsCondition

    parameter_search_space = ConfigurationSpace(name="conditional", seed=1)
    ksp_type = Categorical("ksp_type", ["cg", "fgmres"], default="cg")
    restart = Constant(
        "ksp_gmres_restart",
        80,
        meta={"petsc_option": "-ksp_gmres_restart"},
    )
    pc_type = Constant("pc_type", "hypre", meta={"petsc_option": "-pc_type"})
    parameter_search_space.add([ksp_type, restart, pc_type])
    parameter_search_space.add(EqualsCondition(restart, ksp_type, "fgmres"))

    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        {"ksp_type": "cg", "ksp_gmres_restart": 80, "pc_type": "hypre"},
    )

    assert "-pc_type" in options
    assert "hypre" in options
    assert "-ksp_gmres_restart" not in options


def test_invalid_cg_right_parameter_search_space_is_rejected() -> None:
    from ConfigSpace import Categorical, ConfigurationSpace, Constant

    parameter_search_space = ConfigurationSpace(name="invalid.cg-right", seed=1)
    ksp_type = Categorical(
        "ksp_type",
        ["cg", "fgmres"],
        default="cg",
        meta={"petsc_option": "-ksp_type"},
    )
    ksp_pc_side = Categorical(
        "ksp_pc_side",
        ["left", "right"],
        default="right",
        meta={"petsc_option": "-ksp_pc_side"},
    )
    pc_type = Constant("pc_type", "jacobi", meta={"petsc_option": "-pc_type"})
    parameter_search_space.add([ksp_type, ksp_pc_side, pc_type])

    errors = validate_parameter_search_space(parameter_search_space)

    assert errors == [
        "KSP CG does not support right preconditioning side. "
        "Make ksp_pc_side inactive for ksp_type=cg or restrict CG to left side."
    ]


def test_invalid_cg_right_python_parameter_search_space_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid_cg_right.py"
    path.write_text(
        """
from ConfigSpace import Categorical, ConfigurationSpace, Constant


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="invalid.cg-right", seed=seed)
    parameter_search_space.add(
        [
            Categorical(
                "ksp_type",
                ["cg", "fgmres"],
                default="cg",
                meta={"petsc_option": "-ksp_type"},
            ),
            Categorical(
                "ksp_pc_side",
                ["left", "right"],
                default="right",
                meta={"petsc_option": "-ksp_pc_side"},
            ),
            Constant("pc_type", "jacobi", meta={"petsc_option": "-pc_type"}),
        ]
    )
    return parameter_search_space
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="KSP CG does not support right"):
        load_parameter_search_space(path)


def test_native_configspace_yaml_can_be_loaded(tmp_path) -> None:
    source_parameter_search_space = load_parameter_search_space("petsc.hypre-basic")
    path = tmp_path / "configspace.yaml"
    write_configspace_yaml(source_parameter_search_space, path)

    loaded_parameter_search_space = load_parameter_search_space(path)
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        loaded_parameter_search_space
    )
    options = render_petsc_options_from_solver_configuration(
        loaded_parameter_search_space,
        solver_configuration,
    )

    assert "-pc_type" in options
    assert "hypre" in options


def test_python_parameter_search_space_file_can_be_loaded(tmp_path) -> None:
    path = tmp_path / "asm_ilu.py"
    path.write_text(
        """
from ConfigSpace import Categorical, ConfigurationSpace, Constant, Integer
from ConfigSpace.conditions import EqualsCondition


def create_parameter_search_space(seed: int = 1) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(name="asm_ilu", seed=seed)
    ksp_type = Categorical(
        "ksp_type",
        ["cg", "fgmres"],
        default="fgmres",
        meta={"petsc_option": "-ksp_type"},
    )
    pc_type = Constant("pc_type", "asm", meta={"petsc_option": "-pc_type"})
    pc_asm_overlap = Integer(
        "pc_asm_overlap",
        (0, 4),
        default=1,
        meta={"petsc_option": "-pc_asm_overlap"},
    )
    ksp_gmres_restart = Integer(
        "ksp_gmres_restart",
        (30, 200),
        default=100,
        meta={"petsc_option": "-ksp_gmres_restart"},
    )
    parameter_search_space.add([ksp_type, pc_type, pc_asm_overlap, ksp_gmres_restart])
    parameter_search_space.add(EqualsCondition(ksp_gmres_restart, ksp_type, "fgmres"))
    return parameter_search_space
""".lstrip(),
        encoding="utf-8",
    )

    parameter_search_space = load_parameter_search_space(path)
    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )

    assert "-pc_type" in options
    assert "asm" in options
