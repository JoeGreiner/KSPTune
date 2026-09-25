from __future__ import annotations

from ConfigSpace import Categorical


def initial_solver_configuration(label: str, **overrides):
    return {"label": label, "solver_configuration": overrides}


def petsc_meta(option: str | None, *, emit: bool = True, flag: bool = False, skip_values=None):
    meta = {}
    if option is not None:
        meta["petsc_option"] = option
    if not emit:
        meta["emit"] = False
    if flag:
        meta["flag"] = True
    if skip_values is not None:
        meta["skip_values"] = list(skip_values)
    return meta


def bool_choice(name: str, default: bool, petsc_option: str, *, skip_false: bool = False):
    return Categorical(
        name,
        [False, True],
        default=default,
        meta=petsc_meta(petsc_option, skip_values=[False] if skip_false else None),
    )


def flag_choice(name: str, default: bool, petsc_option: str):
    return Categorical(
        name,
        [False, True],
        default=default,
        meta=petsc_meta(petsc_option, flag=True),
    )
