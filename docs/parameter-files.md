# Parameter files

Built-in files are located in `src/ksptune/builtin_parameter_search_spaces/`.
Please refer to them as examples.

Example exercept:
```python
ksp_type = Constant("ksp_type", "gmres")

pc_hypre_boomeramg_coarsen_type = Categorical(
    "pc_hypre_boomeramg_coarsen_type",
    ["PMIS", "HMIS"],
    default="PMIS",
)

pc_hypre_boomeramg_strong_threshold = UniformFloatHyperparameter(
    "pc_hypre_boomeramg_strong_threshold",
    lower=0.0001,
    upper=0.7,
    default_value=0.1,
    log=False,
)
```

These objects need to get added to a parameter search space:

```python
parameter_search_space = ConfigurationSpace(name="my_parameters")

(...)

parameter_search_space.add(
    [
        ksp_type,
        pc_hypre_boomeramg_coarsen_type,
        pc_hypre_boomeramg_strong_threshold
    ]
)
```


## Types

The quoted parameter name must match a PETSc option. For example,
`"pc_hypre_boomeramg_coarsen_type"` becomes `-pc_hypre_boomeramg_coarsen_type`.
The Python variable name can be chosen freely.


Available Types (see the [ConfigSpace documentation](https://automl.github.io/ConfigSpace/latest/reference/hyperparameters/)
for arguments and examples):

```python
Constant("ksp_type", "gmres")
CategoricalHyperparameter("ksp_type", choices=["gmres", "cg"])
OrdinalHyperparameter("ksp_gmres_restart", sequence=[20, 40, 80], default=40)
UniformIntegerHyperparameter("ksp_max_it", lower=20, upper=100, log=False)
UniformFloatHyperparameter("ksp_rtol", lower=1e-8, upper=1e-4)
NormalIntegerHyperparameter("ksp_max_it", lower=20, upper=100, mu=50, sigma=10)
NormalFloatHyperparameter("ksp_rtol", lower=1e-8, upper=1e-4, mu=1e-5, sigma=1e-6)
BetaIntegerHyperparameter("ksp_max_it", lower=20, upper=100, alpha=2, beta=5)
BetaFloatHyperparameter("ksp_rtol", lower=1e-8, upper=1e-4, alpha=2, beta=5)
```

`Integer` and `Float` accept `Uniform`, `Normal`, or `Beta` distributions;
`Categorical(..., ordered=True)` creates an ordinal (i.e., ordered) parameter.
`log=...` switches between linear and logarithmic scale.


## Conditions

One can add conditions between parameters, e.g., only add a paramter choice if another parameter is chosen to be active.

Examples:

```python
parameter_search_space.add(EqualsCondition(ksp_gmres_restart, ksp_type, "gmres"))
```

The GMRES restart parameter is active only when the solver is `gmres`.

```python
parameter_search_space.add(
    InCondition(
        pc_hypre_boomeramg_agg_num_paths,
        pc_hypre_boomeramg_agg_nl,
        [1, 2],
    )
)
```

The number of aggressive-coarsening paths is active only when the number of
aggressive-coarsening levels (`agg_nl`) is `1` or `2`.

```python
parameter_search_space.add(
    GreaterThanCondition(
        pc_hypre_boomeramg_agg_num_paths,
        pc_hypre_boomeramg_agg_nl,
        0,
    )
)
```

Here, the same parameter is active for any allowed `agg_nl` greater than zero.

```python
parameter_search_space.add(
    AndConjunction(
        EqualsCondition(mg_levels_pc_jacobi_rowl1_scale, mg_levels_pc_type, "jacobi"),
        EqualsCondition(mg_levels_pc_jacobi_rowl1_scale, mg_levels_pc_jacobi_type, "rowl1"),
    )
)
```

The row-L1 scaling parameter is active only when **both** the level preconditioner
is `jacobi` and its Jacobi type is `rowl1`.

Import these classes from `ConfigSpace.conditions`. For more conditions, see
[ConfigSpace documentation](https://automl.github.io/ConfigSpace/latest/reference/conditions/).

## Validation

```bash
ksptune parameter-search-spaces validate petsc.hypre-basic
```

It checks syntax, but not the options against the replay tool's PETSc build.


## Custom file

Save as `src/ksptune/builtin_parameter_search_spaces/my_parameters.py`:

```python
from ConfigSpace import ConfigurationSpace, Constant, Integer


def create_parameter_search_space(seed=1):
    parameter_search_space = ConfigurationSpace(name="my_parameters", seed=seed)
    parameter_search_space.add([
        Constant("ksp_type", "gmres"),
        Constant("pc_type", "jacobi"),
        Integer("ksp_gmres_restart", bounds=(20, 100), default=30),
    ])
    return parameter_search_space
```

This keeps GMRES and Jacobi fixed and tunes the restart length between 20 and 100.

Add your parameterspace in `src/ksptune/builtin_parameter_search_spaces/__init__.py`, add the import and
dictionary, using the same name as in
`ConfigurationSpace(name=...)`:

```python
from .my_parameters import create_parameter_search_space as create_my_parameters

BUILTIN_PARAMETER_SEARCH_SPACES = {
    # ... existing entries ...
    "my_parameters": create_my_parameters,
}
```

Validate it, then use `--parameter-search-space my_parameters` in `ksptune tune`:

```bash
ksptune parameter-search-spaces validate my_parameters
```
