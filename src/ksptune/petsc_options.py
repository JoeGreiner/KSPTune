from __future__ import annotations

from typing import Any

from .solver_configurations import parameter_is_active


def format_petsc_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(value)
    return str(value)


def render_petsc_options_from_solver_configuration(
    parameter_search_space: Any,
    solver_configuration: dict[str, Any],
) -> list[str]:
    options: list[str] = []
    for hyperparameter in parameter_search_space.values():
        meta = getattr(hyperparameter, "meta", None) or {}
        if meta.get("emit", True) is False:
            continue
        name = hyperparameter.name
        if name not in solver_configuration:
            continue
        if not parameter_is_active(parameter_search_space, name, solver_configuration):
            continue
        value = solver_configuration[name]
        if value is None or value == "":
            continue
        if value in (meta.get("skip_values") or []):
            continue

        petsc_option = meta.get("petsc_option", f"-{name}")
        if petsc_option is None:
            continue
        if meta.get("flag", False):
            if bool(value):
                options.append(str(petsc_option))
            continue
        options.extend([str(petsc_option), format_petsc_value(value)])
    return options


def petsc_options_to_text(petsc_options: list[str]) -> str:
    return " ".join(petsc_options)
