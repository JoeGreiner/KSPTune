"""Bundled ConfigSpace-based parameter search spaces."""

from __future__ import annotations

from .petsc_boomeramg_basic import create_parameter_search_space as create_petsc_boomeramg_basic
from .petsc_boomeramg_extended import (
    create_parameter_search_space as create_petsc_boomeramg_extended,
)
from .petsc_boomeramg_extended_mono import (
    create_parameter_search_space as create_petsc_boomeramg_extended_mono,
)
from .petsc_direct_like_mono import create_parameter_search_space as create_petsc_direct_like_mono
from .petsc_direct_mono import create_parameter_search_space as create_petsc_direct_mono
from .petsc_gamg_basic import create_parameter_search_space as create_petsc_gamg_basic
from .petsc_gamg_extended import create_parameter_search_space as create_petsc_gamg_extended
from .petsc_gamg_extended_mono import (
    create_parameter_search_space as create_petsc_gamg_extended_mono,
)
from .petsc_hypre_ilu_mono import create_parameter_search_space as create_petsc_hypre_ilu_mono
from .petsc_parasails_mono import create_parameter_search_space as create_petsc_parasails_mono
from .petsc_pilut_mono import create_parameter_search_space as create_petsc_pilut_mono
from .petsc_solver_comparison_mono import (
    create_parameter_search_space as create_petsc_solver_comparison_mono,
)
from .petsc_spai_mono import create_parameter_search_space as create_petsc_spai_mono

BUILTIN_PARAMETER_SEARCH_SPACES = {
    "petsc.boomeramg_basic": create_petsc_boomeramg_basic,
    "petsc.boomeramg_extended": create_petsc_boomeramg_extended,
    "petsc.boomeramg_extended_mono": create_petsc_boomeramg_extended_mono,
    "petsc.direct_like_mono": create_petsc_direct_like_mono,
    "petsc.direct_mono": create_petsc_direct_mono,
    "petsc.gamg_basic": create_petsc_gamg_basic,
    "petsc.gamg_extended": create_petsc_gamg_extended,
    "petsc.gamg_extended_mono": create_petsc_gamg_extended_mono,
    "petsc.hypre_ilu_mono": create_petsc_hypre_ilu_mono,
    "petsc.parasails_mono": create_petsc_parasails_mono,
    "petsc.pilut_mono": create_petsc_pilut_mono,
    "petsc.solver_comparison_mono": create_petsc_solver_comparison_mono,
    "petsc.spai_mono": create_petsc_spai_mono,
}
