from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from typing import Any

from .file_io import atomic_text_file


def list_builtin_parameter_search_spaces() -> list[str]:
    from .builtin_parameter_search_spaces import BUILTIN_PARAMETER_SEARCH_SPACES

    return sorted(BUILTIN_PARAMETER_SEARCH_SPACES)


def get_parameter_search_space_name(parameter_search_space: Any) -> str:
    return str(getattr(parameter_search_space, "name", None) or "parameter_search_space")


def load_parameter_search_space(name_or_path: str | Path, seed: int | None = 1) -> Any:
    text = str(name_or_path)
    path = Path(text)
    if path.exists():
        parameter_search_space = load_parameter_search_space_from_path(path, seed=seed)
        return with_source(parameter_search_space, str(path.resolve()))

    from .builtin_parameter_search_spaces import BUILTIN_PARAMETER_SEARCH_SPACES

    if text in BUILTIN_PARAMETER_SEARCH_SPACES:
        parameter_search_space = BUILTIN_PARAMETER_SEARCH_SPACES[text](seed=seed or 1)
        return with_source(parameter_search_space, f"builtin:{text}")

    raise ValueError(f"Unknown Parameter Search Space: {name_or_path}")


def load_parameter_search_space_from_path(path: Path, seed: int | None) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".py":
        parameter_search_space = load_parameter_search_space_from_python_file(path, seed=seed)
    elif suffix in {".yaml", ".yml"}:
        from ConfigSpace import ConfigurationSpace

        parameter_search_space = ConfigurationSpace.from_yaml(path)
        seed_parameter_search_space(parameter_search_space, seed)
    elif suffix == ".json":
        from ConfigSpace import ConfigurationSpace

        parameter_search_space = ConfigurationSpace.from_json(path)
        seed_parameter_search_space(parameter_search_space, seed)
    else:
        raise ValueError(
            "Parameter Search Space files must be Python, ConfigSpace YAML, or ConfigSpace JSON."
        )

    errors = validate_parameter_search_space(parameter_search_space)
    if errors:
        raise ValueError("\n".join(errors))
    return parameter_search_space


def load_parameter_search_space_from_python_file(path: Path, seed: int | None) -> Any:
    normalized_stem = path.stem.replace("-", "_").replace(".", "_")
    module_name = f"_ksptune_parameter_search_space_{normalized_stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python Parameter Search Space: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    creator = getattr(module, "create_parameter_search_space", None)
    if not callable(creator):
        raise ValueError(f"{path} must define a callable create_parameter_search_space(seed: int).")
    return creator(seed=seed or 1)


def seed_parameter_search_space(parameter_search_space: Any, seed: int | None) -> None:
    if seed is not None and hasattr(parameter_search_space, "seed"):
        parameter_search_space.seed(seed)


def with_source(parameter_search_space: Any, source: str) -> Any:
    setattr(parameter_search_space, "_ksptune_source", source)
    errors = validate_parameter_search_space(parameter_search_space)
    if errors:
        raise ValueError("\n".join(errors))
    return parameter_search_space


def validate_parameter_search_space(parameter_search_space: Any) -> list[str]:
    from ConfigSpace import ConfigurationSpace

    errors: list[str] = []
    if not isinstance(parameter_search_space, ConfigurationSpace):
        return ["Parameter Search Space must be a ConfigSpace.ConfigurationSpace."]
    if len(parameter_search_space) == 0:
        errors.append("Parameter Search Space must define at least one hyperparameter.")

    for hyperparameter in parameter_search_space.values():
        meta = getattr(hyperparameter, "meta", None)
        if meta is not None and not isinstance(meta, dict):
            errors.append(f"{hyperparameter.name} meta must be a mapping when present.")
        if isinstance(meta, dict) and meta.get("flag", False) and meta.get("petsc_option") is None:
            errors.append(f"{hyperparameter.name} flag parameters need a PETSc option name.")
    errors.extend(validate_petsc_solver_constraints(parameter_search_space))
    return errors


def hyperparameter_allows_value(hyperparameter: Any, value: str) -> bool:
    if getattr(hyperparameter, "value", None) == value:
        return True
    choices = getattr(hyperparameter, "choices", None)
    return choices is not None and value in choices


def validate_petsc_solver_constraints(parameter_search_space: Any) -> list[str]:
    from .petsc_options import render_petsc_options_from_solver_configuration
    from .solver_configurations import default_solver_configuration_from_parameter_search_space

    hyperparameters = {hyperparameter.name: hyperparameter for hyperparameter in parameter_search_space.values()}
    ksp_type = hyperparameters.get("ksp_type")
    ksp_pc_side = hyperparameters.get("ksp_pc_side")
    if ksp_type is None or ksp_pc_side is None:
        return []
    if not hyperparameter_allows_value(ksp_type, "cg"):
        return []
    if not hyperparameter_allows_value(ksp_pc_side, "right"):
        return []

    solver_configuration = default_solver_configuration_from_parameter_search_space(
        parameter_search_space
    )
    solver_configuration["ksp_type"] = "cg"
    solver_configuration["ksp_pc_side"] = "right"
    petsc_options = render_petsc_options_from_solver_configuration(
        parameter_search_space,
        solver_configuration,
    )
    if "-ksp_pc_side" in petsc_options and "right" in petsc_options:
        return [
            "KSP CG does not support right preconditioning side. "
            "Make ksp_pc_side inactive for ksp_type=cg or restrict CG to left side."
        ]
    return []


def build_configspace_from_parameter_search_space(parameter_search_space: Any, seed: int = 1) -> Any:
    seed_parameter_search_space(parameter_search_space, seed)
    errors = validate_parameter_search_space(parameter_search_space)
    if errors:
        raise ValueError("\n".join(errors))
    return parameter_search_space


def write_configspace_json(configuration_space: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_file(path) as handle:
        configuration_space.to_json(handle, indent=2)


def write_configspace_yaml(configuration_space: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_file(path) as handle:
        configuration_space.to_yaml(handle, sort_keys=False)


def parameter_search_space_to_yaml(parameter_search_space: Any) -> str:
    buffer = io.StringIO()
    parameter_search_space.to_yaml(buffer, sort_keys=False)
    return buffer.getvalue()


def configspace_to_pretty_json(configuration_space: Any) -> str:
    return json.dumps(configuration_space.to_serialized_dict(), indent=2, sort_keys=True)
