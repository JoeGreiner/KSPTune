from __future__ import annotations

from typing import Any


def plain_python_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            return value
    return value


def parameter_is_active(
    parameter_search_space: Any,
    parameter_name: str,
    solver_configuration: dict[str, Any],
) -> bool:
    for condition in getattr(parameter_search_space, "conditions", []):
        child = getattr(condition, "child", None)
        if getattr(child, "name", None) != parameter_name:
            continue
        try:
            condition_is_satisfied = condition.satisfied_by_value(solver_configuration)
        except KeyError:
            return False
        if not condition_is_satisfied:
            return False
    return True


def default_solver_configuration_from_parameter_search_space(
    parameter_search_space: Any,
) -> dict[str, Any]:
    return {
        name: plain_python_value(value)
        for name, value in dict(parameter_search_space.get_default_configuration()).items()
    }


def solver_configuration_from_configspace_configuration(configuration: Any) -> dict[str, Any]:
    return {name: plain_python_value(value) for name, value in dict(configuration).items()}


def validate_solver_configuration(
    parameter_search_space: Any,
    solver_configuration: dict[str, Any],
) -> list[str]:
    from ConfigSpace import Configuration

    try:
        Configuration(parameter_search_space, values=solver_configuration)
    except Exception as exc:  # noqa: BLE001
        return [str(exc)]
    return []
