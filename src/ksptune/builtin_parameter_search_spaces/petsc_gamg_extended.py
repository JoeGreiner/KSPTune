from __future__ import annotations

from ..search_space_helpers import bool_choice, flag_choice, initial_solver_configuration

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import AndConjunction, EqualsCondition, InCondition


# PETSc options excluded or restricted in this search space:
#
# -pc_gamg_type geo/classical
#     This space uses agg. geo requires coordinates; classical has its own
#     interpolation options. Both need separate search spaces.
# -pc_mg_levels
#     The local PETSc GAMG source rejects this option. GAMG determines the number
#     of levels from its coarsening settings and coarse-grid size limits.
# -pc_mg_galerkin both/pmat/mat/none/external
#     GAMG builds its own coarse operators and sets this to external internally.
# -pc_gamg_classical_type, -pc_gamg_classical_interp_threshold,
# -pc_gamg_classical_nsmooths
#     Only active for -pc_gamg_type classical.
# -pc_gamg_threshold <array>, -pc_gamg_rank_reduction_factors <array>,
# -pc_gamg_eigenvalues <min,max>, -pc_gamg_injection_index <array>,
# -pc_gamg_mat_coarsen_strength_index <array>
#     These options take arrays, which need additional support in the parameter
#     renderer before they can be sampled.
# -pc_mg_adapt_interp_coarse_space, -pc_mg_adapt_interp_n,
# -pc_mg_adapt_cr, -pc_mg_mesp_monitor, gdsw_* options
#     These options change how the coarse space is built and need a separate
#     search space.
# -pc_mg_distinct_smoothup
#     Separate up/down smoothers would need two sets of smoother parameters.
# -mg_levels_ksp_chebyshev_eigenvalues <min,max>
#     Sets exact Chebyshev bounds as an array. This space tunes the esteig
#     transform instead.
# -ksp_gmres_haptol, -ksp_gmres_breakdown_tolerance
#     Control GMRES stability and early termination. Keep PETSc defaults unless
#     measurements show a need to tune these tolerances.
# -pc_gamg_prolongator_filter
#     Not found in the local PETSc 3.22 source inspected for this project.
#     Check that the active PETSc build supports it before adding it.
# -pc_gamg_esteig_ksp_rtol and other generic -pc_gamg_esteig_* KSP options
#     Control eigenvalue estimation for prolongation smoothing. This space
#     samples type and max_it; the other KSP options are not sampled.
# -mg_levels_esteig_ksp_rtol and other generic -mg_levels_esteig_* KSP options
#     Control eigenvalue estimation for Chebyshev smoothers. This space samples
#     type, max_it, and rtol; the other KSP options are not sampled.
# -mg_levels_pc_asm_overlap, -mg_levels_pc_asm_type,
# -mg_levels_pc_asm_local_type, -mg_levels_pc_asm_blocks,
# -mg_levels_pc_asm_local_blocks, -mg_levels_pc_asm_sub_mat_type,
# -mg_levels_sub_ksp_type, -mg_levels_sub_pc_type,
# -mg_levels_sub_pc_factor_*
#     Control ASM subdomains and their solvers when mg_levels_pc_type=asm.
#     Tuning them needs a separate set of conditional parameters.
# -mg_levels_pc_sor_symmetric/backward/forward/local_*
#     Select the SOR sweep direction through boolean flags. This space keeps
#     the default sweep and tunes omega and iteration counts.
# -mg_coarse_pc_bjacobi_blocks, -mg_coarse_pc_bjacobi_local_blocks
#     Set the coarse-grid BJacobi block layout. Keep PETSc defaults unless
#     diagnostics show that the block layout limits coarse-solve performance.
# -mg_coarse_pc_factor_fill, -mg_coarse_pc_factor_shift_*,
# -mg_coarse_sub_pc_factor_fill, -mg_coarse_sub_pc_factor_shift_*
#     Direct-solver settings for the coarse grid. Their effects depend on the
#     solver package; some choices can cause singular factors or high fill-in.
#     Consider tuning them when coarse solves dominate runtime.


def create_parameter_search_space(seed: int = 1, *, mono: bool = False) -> ConfigurationSpace:
    parameter_search_space = ConfigurationSpace(
        name="petsc.gamg_extended_mono" if mono else "petsc.gamg_extended", seed=seed
    )

    ksp_type = Constant("ksp_type", "gmres")
    ksp_pc_side = Constant("ksp_pc_side", "left")
    pc_type = Constant("pc_type", "gamg")
    pc_gamg_type = Constant("pc_gamg_type", "agg")
    ksp_norm_type = Constant(
        "ksp_norm_type",
        "preconditioned",
    )

    ksp_atol = Constant("ksp_atol", 1.0e-8)
    ksp_rtol = Constant("ksp_rtol", 1.0e-8 if mono else 1.0e-7)
    ksp_reuse_preconditioner = bool_choice(
        "ksp_reuse_preconditioner",
        True,
        "-ksp_reuse_preconditioner",
    )
    ksp_gmres_restart = Categorical(
        "ksp_gmres_restart",
        [30, 50, 100],
        default=30,
    )
    ksp_max_it = Categorical(
        "ksp_max_it",
        [1000, 3000, 10000],
        default=10000,
    )
    ksp_gmres_preallocate = flag_choice(
        "ksp_gmres_preallocate",
        False,
        "-ksp_gmres_preallocate",
    )
    ksp_gmres_modifiedgramschmidt = flag_choice(
        "ksp_gmres_modifiedgramschmidt",
        False,
        "-ksp_gmres_modifiedgramschmidt",
    )
    ksp_gmres_cgs_refinement_type = Categorical(
        "ksp_gmres_cgs_refinement_type",
        ["refine_never", "refine_ifneeded", "refine_always"],
        default="refine_never",
    )
    pc_gamg_esteig_ksp_type = Categorical(
        "pc_gamg_esteig_ksp_type",
        ["cg", "gmres"],
        default="cg",
    )
    pc_gamg_esteig_ksp_max_it = Categorical(
        "pc_gamg_esteig_ksp_max_it",
        [5, 10, 20],
        default=10,
    )

    pc_mg_type = Categorical(
        "pc_mg_type",
        ["multiplicative", "additive", "full", "kaskade"],
        default="multiplicative",
    )
    pc_mg_cycle_type = Categorical(
        "pc_mg_cycle_type",
        ["v", "w"],
        default="v",
    )
    pc_mg_multiplicative_cycles = UniformIntegerHyperparameter(
        "pc_mg_multiplicative_cycles",
        lower=1,
        upper=2,
        default_value=1,
    )

    pc_gamg_agg_nsmooths = UniformIntegerHyperparameter(
        "pc_gamg_agg_nsmooths",
        lower=1,
        upper=2,
        default_value=1,
    )
    pc_gamg_aggressive_coarsening = UniformIntegerHyperparameter(
        "pc_gamg_aggressive_coarsening",
        lower=0,
        upper=6,
        default_value=1,
    )
    pc_gamg_aggressive_square_graph = bool_choice(
        "pc_gamg_aggressive_square_graph",
        True,
        "-pc_gamg_aggressive_square_graph",
    )
    pc_gamg_aggressive_mis_k = UniformIntegerHyperparameter(
        "pc_gamg_aggressive_mis_k",
        lower=2,
        upper=4,
        default_value=2,
    )

    pc_gamg_threshold_mode = Categorical(
        "pc_gamg_threshold_mode",
        ["petsc-default", "set"],
        default="petsc-default",
        meta={"emit": False},
    )
    pc_gamg_threshold = UniformFloatHyperparameter(
        "pc_gamg_threshold",
        lower=0.0,
        upper=0.3 if mono else 0.06,
        default_value=0.03,
    )
    pc_gamg_threshold_scale = UniformFloatHyperparameter(
        "pc_gamg_threshold_scale",
        lower=0.05 if mono else 0.25,
        upper=2.0,
        default_value=1.0,
    )
    pc_gamg_low_memory_threshold_filter = (
        Constant("pc_gamg_low_memory_threshold_filter", False)
        if mono
        else bool_choice(
            "pc_gamg_low_memory_threshold_filter",
            False,
            "-pc_gamg_low_memory_threshold_filter",
        )
    )
    pc_gamg_prolongator_filter = Constant(
        "pc_gamg_prolongator_filter",
        False,
        meta={
            "emit": False,
            "petsc_option": "-pc_gamg_prolongator_filter",
            "note": "Inventory only: not present in the local PETSc 3.22 source.",
        },
    )
    pc_gamg_graph_symmetrize = bool_choice(
        "pc_gamg_graph_symmetrize",
        True,
        "-pc_gamg_graph_symmetrize",
    )
    pc_gamg_mis_k_minimum_degree_ordering = bool_choice(
        "pc_gamg_mis_k_minimum_degree_ordering",
        False,
        "-pc_gamg_mis_k_minimum_degree_ordering",
    )

    pc_gamg_mat_coarsen_type = Categorical(
        "pc_gamg_mat_coarsen_type",
        ["misk", "mis", "hem"],
        default="misk",
    )
    pc_gamg_mat_coarsen_misk_distance = UniformIntegerHyperparameter(
        "pc_gamg_mat_coarsen_misk_distance",
        lower=1,
        upper=4 if mono else 3,
        default_value=1,
    )
    pc_gamg_mat_coarsen_max_it = UniformIntegerHyperparameter(
        "pc_gamg_mat_coarsen_max_it",
        lower=1,
        upper=8,
        default_value=4,
    )
    pc_gamg_mat_coarsen_threshold_mode = Categorical(
        "pc_gamg_mat_coarsen_threshold_mode",
        ["petsc-default", "set"],
        default="petsc-default",
        meta={"emit": False},
    )
    pc_gamg_mat_coarsen_threshold = UniformIntegerHyperparameter(
        "pc_gamg_mat_coarsen_threshold",
        lower=0,
        upper=8,
        default_value=0,
    )

    mg_levels_ksp_type = Categorical(
        "mg_levels_ksp_type",
        ["chebyshev", "richardson"],
        default="chebyshev",
    )
    mg_levels_ksp_max_it = UniformIntegerHyperparameter(
        "mg_levels_ksp_max_it",
        lower=1,
        upper=2,
        default_value=1,
    )
    mg_levels_pc_type = Categorical(
        "mg_levels_pc_type",
        ["jacobi", "sor", "asm"],
        default="jacobi",
    )
    mg_levels_ksp_chebyshev_kind = Categorical(
        "mg_levels_ksp_chebyshev_kind",
        ["first", "fourth", "opt_fourth"],
        default="first",
    )
    mg_levels_ksp_chebyshev_esteig_steps = Categorical(
        "mg_levels_ksp_chebyshev_esteig_steps",
        [5, 10, 20],
        default=10,
    )
    mg_levels_ksp_chebyshev_esteig_mode = Categorical(
        "mg_levels_ksp_chebyshev_esteig_mode",
        ["petsc-default", "set"],
        default="petsc-default",
        meta={"emit": False},
    )
    mg_levels_ksp_chebyshev_esteig = Categorical(
        "mg_levels_ksp_chebyshev_esteig",
        ["0,0.25,0,1.2", "0,0.1,0,1.1", "0,0.3,0,1.3"],
        default="0,0.25,0,1.2",
    )
    mg_levels_ksp_chebyshev_esteig_noisy = (
        Constant("mg_levels_ksp_chebyshev_esteig_noisy", True)
        if mono
        else bool_choice(
            "mg_levels_ksp_chebyshev_esteig_noisy",
            False,
            "-mg_levels_ksp_chebyshev_esteig_noisy",
        )
    )
    mg_levels_esteig_ksp_type = Categorical(
        "mg_levels_esteig_ksp_type",
        ["cg", "gmres"],
        default="cg",
    )
    mg_levels_esteig_ksp_max_it = Categorical(
        "mg_levels_esteig_ksp_max_it",
        [5, 10, 20],
        default=10,
    )
    mg_levels_esteig_ksp_rtol = Categorical(
        "mg_levels_esteig_ksp_rtol",
        [1.0e-2, 1.0e-3, 1.0e-4],
        default=1.0e-2,
    )
    mg_levels_pc_jacobi_type = Categorical(
        "mg_levels_pc_jacobi_type",
        ["diagonal", "rowl1", "rowmax", "rowsum"],
        default="diagonal",
    )
    mg_levels_pc_jacobi_abs = bool_choice(
        "mg_levels_pc_jacobi_abs",
        False,
        "-mg_levels_pc_jacobi_abs",
    )
    mg_levels_pc_jacobi_fixdiagonal = bool_choice(
        "mg_levels_pc_jacobi_fixdiagonal",
        True,
        "-mg_levels_pc_jacobi_fixdiagonal",
    )
    mg_levels_pc_jacobi_rowl1_scale = UniformFloatHyperparameter(
        "mg_levels_pc_jacobi_rowl1_scale",
        lower=0.0,
        upper=1.0,
        default_value=1.0,
    )
    mg_levels_pc_sor_omega = UniformFloatHyperparameter(
        "mg_levels_pc_sor_omega",
        lower=0.5,
        upper=1.9,
        default_value=1.0,
    )
    mg_levels_pc_sor_diagonal_shift = UniformFloatHyperparameter(
        "mg_levels_pc_sor_diagonal_shift",
        lower=0.0,
        upper=1.0e-8,
        default_value=0.0,
    )
    mg_levels_pc_sor_its = UniformIntegerHyperparameter(
        "mg_levels_pc_sor_its",
        lower=1,
        upper=3,
        default_value=1,
    )
    mg_levels_pc_sor_lits = UniformIntegerHyperparameter(
        "mg_levels_pc_sor_lits",
        lower=1,
        upper=3,
        default_value=1,
    )
    mg_coarse_ksp_type = Categorical(
        "mg_coarse_ksp_type",
        ["preonly", "richardson", "gmres"],
        default="preonly",
    )
    mg_coarse_ksp_max_it = Categorical(
        "mg_coarse_ksp_max_it",
        [1, 10, 100],
        default=10,
    )
    mg_coarse_pc_type = Categorical(
        "mg_coarse_pc_type",
        ["bjacobi", "lu", "jacobi"] if mono else ["bjacobi", "lu", "ilu", "jacobi"],
        default="bjacobi",
    )
    mg_coarse_pc_factor_mat_ordering_type = Categorical(
        "mg_coarse_pc_factor_mat_ordering_type",
        ["nd", "rcm", "natural"],
        default="nd",
    )
    mg_coarse_sub_pc_type = Categorical(
        "mg_coarse_sub_pc_type",
        ["lu", "ilu"],
        default="lu",
    )
    mg_coarse_sub_pc_factor_mat_ordering_type = Categorical(
        "mg_coarse_sub_pc_factor_mat_ordering_type",
        ["nd", "rcm", "natural"],
        default="nd",
    )

    pc_gamg_repartition = bool_choice("pc_gamg_repartition", False, "-pc_gamg_repartition")
    pc_gamg_use_sa_esteig = bool_choice(
        "pc_gamg_use_sa_esteig",
        True,
        "-pc_gamg_use_sa_esteig",
    )
    pc_gamg_recompute_esteig = bool_choice(
        "pc_gamg_recompute_esteig",
        True,
        "-pc_gamg_recompute_esteig",
    )
    pc_gamg_reuse_interpolation = bool_choice(
        "pc_gamg_reuse_interpolation",
        True,
        "-pc_gamg_reuse_interpolation",
    )
    pc_gamg_asm_use_agg = bool_choice(
        "pc_gamg_asm_use_agg",
        False,
        "-pc_gamg_asm_use_agg",
    )
    pc_gamg_asm_hem_aggs = UniformIntegerHyperparameter(
        "pc_gamg_asm_hem_aggs",
        lower=0,
        upper=4,
        default_value=0,
    )
    pc_gamg_coarse_grid_layout_type = Categorical(
        "pc_gamg_coarse_grid_layout_type",
        ["spread", "compact"],
        default="spread",
    )
    pc_gamg_process_eq_limit = UniformIntegerHyperparameter(
        "pc_gamg_process_eq_limit",
        lower=8,
        upper=1000 if mono else 400,
        default_value=50,
    )
    pc_gamg_parallel_coarse_grid_solver = bool_choice(
        "pc_gamg_parallel_coarse_grid_solver",
        False,
        "-pc_gamg_parallel_coarse_grid_solver",
    )
    pc_gamg_cpu_pin_coarse_grids = bool_choice(
        "pc_gamg_cpu_pin_coarse_grids",
        False,
        "-pc_gamg_cpu_pin_coarse_grids",
    )
    pc_gamg_coarse_eq_limit = UniformIntegerHyperparameter(
        "pc_gamg_coarse_eq_limit",
        lower=25,
        upper=12000,
        default_value=50,
    )

    parameter_search_space.add(
        [
            ksp_type,
            ksp_pc_side,
            pc_type,
            pc_gamg_type,
            ksp_norm_type,
            ksp_atol,
            ksp_rtol,
            ksp_reuse_preconditioner,
            ksp_gmres_restart,
            ksp_max_it,
            ksp_gmres_preallocate,
            ksp_gmres_modifiedgramschmidt,
            ksp_gmres_cgs_refinement_type,
            pc_gamg_esteig_ksp_type,
            pc_gamg_esteig_ksp_max_it,
            pc_mg_type,
            pc_mg_cycle_type,
            pc_mg_multiplicative_cycles,
            pc_gamg_agg_nsmooths,
            pc_gamg_aggressive_coarsening,
            pc_gamg_aggressive_square_graph,
            pc_gamg_aggressive_mis_k,
            pc_gamg_threshold_mode,
            pc_gamg_threshold,
            pc_gamg_threshold_scale,
            pc_gamg_low_memory_threshold_filter,
            pc_gamg_prolongator_filter,
            pc_gamg_graph_symmetrize,
            pc_gamg_mis_k_minimum_degree_ordering,
            pc_gamg_mat_coarsen_type,
            pc_gamg_mat_coarsen_misk_distance,
            pc_gamg_mat_coarsen_max_it,
            pc_gamg_mat_coarsen_threshold_mode,
            pc_gamg_mat_coarsen_threshold,
            mg_levels_ksp_type,
            mg_levels_ksp_max_it,
            mg_levels_pc_type,
            mg_levels_ksp_chebyshev_kind,
            mg_levels_ksp_chebyshev_esteig_steps,
            mg_levels_ksp_chebyshev_esteig_mode,
            mg_levels_ksp_chebyshev_esteig,
            mg_levels_ksp_chebyshev_esteig_noisy,
            mg_levels_esteig_ksp_type,
            mg_levels_esteig_ksp_max_it,
            mg_levels_esteig_ksp_rtol,
            mg_levels_pc_jacobi_type,
            mg_levels_pc_jacobi_abs,
            mg_levels_pc_jacobi_fixdiagonal,
            mg_levels_pc_jacobi_rowl1_scale,
            mg_levels_pc_sor_omega,
            mg_levels_pc_sor_diagonal_shift,
            mg_levels_pc_sor_its,
            mg_levels_pc_sor_lits,
            mg_coarse_ksp_type,
            mg_coarse_ksp_max_it,
            mg_coarse_pc_type,
            mg_coarse_pc_factor_mat_ordering_type,
            mg_coarse_sub_pc_type,
            mg_coarse_sub_pc_factor_mat_ordering_type,
            pc_gamg_repartition,
            pc_gamg_use_sa_esteig,
            pc_gamg_recompute_esteig,
            pc_gamg_reuse_interpolation,
            pc_gamg_asm_use_agg,
            pc_gamg_asm_hem_aggs,
            pc_gamg_coarse_grid_layout_type,
            pc_gamg_process_eq_limit,
            pc_gamg_parallel_coarse_grid_solver,
            pc_gamg_cpu_pin_coarse_grids,
            pc_gamg_coarse_eq_limit,
        ]
    )

    parameter_search_space.add(
        [
            EqualsCondition(
                pc_mg_multiplicative_cycles,
                pc_mg_type,
                "multiplicative",
            ),
            InCondition(
                pc_gamg_aggressive_coarsening,
                pc_gamg_mat_coarsen_type,
                ["misk", "mis"],
            ),
            AndConjunction(
                InCondition(
                    pc_gamg_aggressive_square_graph,
                    pc_gamg_aggressive_coarsening,
                    [1, 2, 3, 4, 5, 6],
                ),
                EqualsCondition(
                    pc_gamg_aggressive_square_graph,
                    pc_gamg_mat_coarsen_type,
                    "misk",
                ),
            ),
            AndConjunction(
                InCondition(
                    pc_gamg_aggressive_mis_k,
                    pc_gamg_aggressive_coarsening,
                    [1, 2, 3, 4, 5, 6],
                ),
                EqualsCondition(
                    pc_gamg_aggressive_mis_k,
                    pc_gamg_aggressive_square_graph,
                    False,
                ),
                EqualsCondition(
                    pc_gamg_aggressive_mis_k,
                    pc_gamg_mat_coarsen_type,
                    "misk",
                ),
            ),
            EqualsCondition(
                pc_gamg_threshold,
                pc_gamg_threshold_mode,
                "set",
            ),
            EqualsCondition(
                pc_gamg_threshold_scale,
                pc_gamg_threshold_mode,
                "set",
            ),
            EqualsCondition(
                pc_gamg_mat_coarsen_misk_distance,
                pc_gamg_mat_coarsen_type,
                "misk",
            ),
            EqualsCondition(
                pc_gamg_mat_coarsen_max_it,
                pc_gamg_mat_coarsen_type,
                "hem",
            ),
            AndConjunction(
                EqualsCondition(
                    pc_gamg_mat_coarsen_threshold,
                    pc_gamg_mat_coarsen_type,
                    "hem",
                ),
                EqualsCondition(
                    pc_gamg_mat_coarsen_threshold,
                    pc_gamg_mat_coarsen_threshold_mode,
                    "set",
                ),
            ),
            InCondition(
                pc_gamg_mis_k_minimum_degree_ordering,
                pc_gamg_mat_coarsen_type,
                ["misk", "mis"],
            ),
            EqualsCondition(
                pc_gamg_esteig_ksp_type,
                pc_gamg_use_sa_esteig,
                True,
            ),
            EqualsCondition(
                pc_gamg_esteig_ksp_max_it,
                pc_gamg_use_sa_esteig,
                True,
            ),
            EqualsCondition(
                mg_levels_ksp_chebyshev_kind,
                mg_levels_ksp_type,
                "chebyshev",
            ),
            EqualsCondition(
                mg_levels_ksp_chebyshev_esteig_steps,
                mg_levels_ksp_type,
                "chebyshev",
            ),
            AndConjunction(
                EqualsCondition(
                    mg_levels_ksp_chebyshev_esteig,
                    mg_levels_ksp_type,
                    "chebyshev",
                ),
                EqualsCondition(
                    mg_levels_ksp_chebyshev_esteig,
                    mg_levels_ksp_chebyshev_esteig_mode,
                    "set",
                ),
            ),
            EqualsCondition(
                mg_levels_ksp_chebyshev_esteig_noisy,
                mg_levels_ksp_type,
                "chebyshev",
            ),
            EqualsCondition(
                mg_levels_esteig_ksp_type,
                mg_levels_ksp_type,
                "chebyshev",
            ),
            EqualsCondition(
                mg_levels_esteig_ksp_max_it,
                mg_levels_ksp_type,
                "chebyshev",
            ),
            EqualsCondition(
                mg_levels_esteig_ksp_rtol,
                mg_levels_ksp_type,
                "chebyshev",
            ),
            EqualsCondition(
                mg_levels_pc_jacobi_type,
                mg_levels_pc_type,
                "jacobi",
            ),
            EqualsCondition(
                mg_levels_pc_jacobi_abs,
                mg_levels_pc_type,
                "jacobi",
            ),
            EqualsCondition(
                mg_levels_pc_jacobi_fixdiagonal,
                mg_levels_pc_type,
                "jacobi",
            ),
            AndConjunction(
                EqualsCondition(
                    mg_levels_pc_jacobi_rowl1_scale,
                    mg_levels_pc_type,
                    "jacobi",
                ),
                EqualsCondition(
                    mg_levels_pc_jacobi_rowl1_scale,
                    mg_levels_pc_jacobi_type,
                    "rowl1",
                ),
            ),
            EqualsCondition(
                mg_levels_pc_sor_omega,
                mg_levels_pc_type,
                "sor",
            ),
            EqualsCondition(
                mg_levels_pc_sor_diagonal_shift,
                mg_levels_pc_type,
                "sor",
            ),
            EqualsCondition(
                mg_levels_pc_sor_its,
                mg_levels_pc_type,
                "sor",
            ),
            EqualsCondition(
                mg_levels_pc_sor_lits,
                mg_levels_pc_type,
                "sor",
            ),
            InCondition(
                mg_coarse_ksp_max_it,
                mg_coarse_ksp_type,
                ["richardson", "gmres"],
            ),
            InCondition(
                mg_coarse_pc_factor_mat_ordering_type,
                mg_coarse_pc_type,
                ["lu"] if mono else ["lu", "ilu"],
            ),
            EqualsCondition(
                mg_coarse_sub_pc_type,
                mg_coarse_pc_type,
                "bjacobi",
            ),
            AndConjunction(
                EqualsCondition(
                    mg_coarse_sub_pc_factor_mat_ordering_type,
                    mg_coarse_pc_type,
                    "bjacobi",
                ),
                InCondition(
                    mg_coarse_sub_pc_factor_mat_ordering_type,
                    mg_coarse_sub_pc_type,
                    ["lu", "ilu"],
                ),
            ),
            EqualsCondition(
                pc_gamg_asm_use_agg,
                mg_levels_pc_type,
                "asm",
            ),
            AndConjunction(
                EqualsCondition(
                    pc_gamg_asm_hem_aggs,
                    mg_levels_pc_type,
                    "asm",
                ),
                EqualsCondition(
                    pc_gamg_asm_hem_aggs,
                    pc_gamg_asm_use_agg,
                    True,
                ),
            ),
            EqualsCondition(
                pc_gamg_coarse_eq_limit,
                pc_gamg_parallel_coarse_grid_solver,
                True,
            ),
        ]
    )

    parameter_search_space._ksptune_initial_solver_configurations = [
        initial_solver_configuration(
            "gamg_default_baseline",
            ksp_reuse_preconditioner=False,
            ksp_gmres_restart=30,
            pc_mg_type="multiplicative",
            pc_mg_cycle_type="v",
            pc_mg_multiplicative_cycles=1,
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=1,
            pc_gamg_aggressive_square_graph=True,
            pc_gamg_threshold_mode="petsc-default",
            pc_gamg_low_memory_threshold_filter=False,
            pc_gamg_graph_symmetrize=True,
            pc_gamg_mis_k_minimum_degree_ordering=False,
            pc_gamg_mat_coarsen_type="misk",
            pc_gamg_mat_coarsen_misk_distance=1,
            mg_levels_ksp_type="chebyshev",
            mg_levels_ksp_max_it=1,
            mg_levels_pc_type="jacobi",
            pc_gamg_repartition=False,
            pc_gamg_use_sa_esteig=True,
            pc_gamg_recompute_esteig=True,
            pc_gamg_reuse_interpolation=True,
            pc_gamg_coarse_grid_layout_type="spread",
            pc_gamg_process_eq_limit=50,
            pc_gamg_parallel_coarse_grid_solver=False,
            pc_gamg_cpu_pin_coarse_grids=False,
        ),
        initial_solver_configuration(
            "gamg_mis_chebyshev_jacobi_compact",
            ksp_reuse_preconditioner=True,
            pc_mg_type="multiplicative",
            pc_mg_cycle_type="v",
            pc_mg_multiplicative_cycles=1,
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=4,
            pc_gamg_threshold_mode="set",
            pc_gamg_threshold=0.0085,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_mat_coarsen_type="mis",
            mg_levels_ksp_type="chebyshev",
            mg_levels_ksp_max_it=1,
            mg_levels_pc_type="jacobi",
            pc_gamg_repartition=False,
            pc_gamg_recompute_esteig=False,
            pc_gamg_reuse_interpolation=True,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_process_eq_limit=145,
            pc_gamg_parallel_coarse_grid_solver=False,
        ),
        initial_solver_configuration(
            "gamg_mis_parallel_coarse_compact",
            ksp_reuse_preconditioner=True,
            pc_mg_type="multiplicative",
            pc_mg_cycle_type="v",
            pc_mg_multiplicative_cycles=1,
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=5,
            pc_gamg_threshold_mode="set",
            pc_gamg_threshold=0.009,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_mat_coarsen_type="mis",
            mg_levels_ksp_type="chebyshev",
            mg_levels_ksp_max_it=1,
            mg_levels_pc_type="jacobi",
            pc_gamg_repartition=False,
            pc_gamg_recompute_esteig=False,
            pc_gamg_reuse_interpolation=True,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_process_eq_limit=100,
            pc_gamg_parallel_coarse_grid_solver=True,
            pc_gamg_coarse_eq_limit=6000,
        ),
        initial_solver_configuration(
            "gamg_mis_sor_low_mpi_probe",
            ksp_reuse_preconditioner=True,
            pc_mg_type="multiplicative",
            pc_mg_cycle_type="v",
            pc_mg_multiplicative_cycles=1,
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=3,
            pc_gamg_threshold_mode="set",
            pc_gamg_threshold=0.006,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_mat_coarsen_type="mis",
            mg_levels_ksp_type="chebyshev",
            mg_levels_ksp_max_it=1,
            mg_levels_pc_type="sor",
            pc_gamg_repartition=False,
            pc_gamg_recompute_esteig=False,
            pc_gamg_reuse_interpolation=True,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_process_eq_limit=200,
            pc_gamg_parallel_coarse_grid_solver=True,
            pc_gamg_coarse_eq_limit=2000,
        ),
        initial_solver_configuration(
            "gamg_mis_max_it_2_residual_probe",
            ksp_reuse_preconditioner=True,
            pc_mg_type="multiplicative",
            pc_mg_cycle_type="v",
            pc_mg_multiplicative_cycles=1,
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=3,
            pc_gamg_threshold_mode="set",
            pc_gamg_threshold=0.008,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_mat_coarsen_type="mis",
            mg_levels_ksp_type="chebyshev",
            mg_levels_ksp_max_it=2,
            mg_levels_pc_type="jacobi",
            pc_gamg_repartition=False,
            pc_gamg_recompute_esteig=False,
            pc_gamg_reuse_interpolation=True,
            pc_gamg_coarse_grid_layout_type="spread",
            pc_gamg_process_eq_limit=200,
            pc_gamg_parallel_coarse_grid_solver=True,
            pc_gamg_coarse_eq_limit=6000,
        ),
        initial_solver_configuration(
            "gamg_mis_low_memory_filter_probe",
            ksp_reuse_preconditioner=True,
            pc_mg_type="multiplicative",
            pc_mg_cycle_type="v",
            pc_mg_multiplicative_cycles=1,
            pc_gamg_agg_nsmooths=1,
            pc_gamg_aggressive_coarsening=4,
            pc_gamg_threshold_mode="set",
            pc_gamg_threshold=0.0085,
            pc_gamg_threshold_scale=1.0,
            pc_gamg_low_memory_threshold_filter=True,
            pc_gamg_mat_coarsen_type="mis",
            mg_levels_ksp_type="chebyshev",
            mg_levels_ksp_max_it=1,
            mg_levels_pc_type="jacobi",
            pc_gamg_repartition=False,
            pc_gamg_recompute_esteig=False,
            pc_gamg_reuse_interpolation=True,
            pc_gamg_coarse_grid_layout_type="compact",
            pc_gamg_process_eq_limit=145,
            pc_gamg_parallel_coarse_grid_solver=False,
        ),
    ]
    return parameter_search_space
