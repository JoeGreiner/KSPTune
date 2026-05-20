"""Bundled ConfigSpace-based Parameter Search Spaces."""

from __future__ import annotations

from .petsc_hypre_basic import create_parameter_search_space as create_petsc_hypre_basic

BUILTIN_PARAMETER_SEARCH_SPACES = {
    "petsc.hypre-basic": create_petsc_hypre_basic,
}
