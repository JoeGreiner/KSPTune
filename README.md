# KSPTune: Bayesian Optimization for PETSc KSP Solvers

KSPTune tunes PETSc KSP/PC options on exported linear-system snapshots using SMAC.
The simulation writes `A`, `b`, `x0`, and metadata; KSPTune replays and optimizes them.

## Install

```bash
conda env create -f environment.yml
conda activate ksptune
pip install -e .
```

## Build PETSc Tools

```bash
cmake -S . -B build
cmake --build build -j
```

If PETSc is not found:

```bash
export PETSC_DIR=/path/to/petsc
export PKG_CONFIG_PATH="$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
```

## Snapshot Files

KSPTune scans a snapshot directory directly. Expected files:

```text
<meshname>__solve_000000__A.bin
<meshname>__solve_000000__b.bin
<meshname>__solve_000000__x0.bin
<meshname>__solve_000000__metadata.txt
```

## Tune

```bash
ksptune tune \
  --snapshot-directory /path/to/snapshots \
  --parameter-search-space petsc.hypre-basic \
  --output-directory runs/hypre
```

By default this runs until `Ctrl+C`. For a fixed budget:

```bash
ksptune tune \
  --snapshot-directory /path/to/snapshots \
  --parameter-search-space petsc.hypre-basic \
  --output-directory runs/hypre \
  --trials 50
```

Parallel runs:

```bash
ksptune tune ... --workers 4 --np 16
```

This runs 4 SMAC workers, each launching PETSc with 16 MPI ranks.

## Analyze

```bash
ksptune analyze-snapshots /path/to/snapshots
```

This checks snapshot metadata and, when built, PETSc matrix/nullspace diagnostics.

## Parameter Search Spaces

List built-ins:

```bash
ksptune parameter-search-spaces list
```

Show one:

```bash
ksptune parameter-search-spaces show petsc.hypre-basic
```

Search spaces are ConfigSpace objects, usually authored as small Python files.
KSPTune also accepts ConfigSpace YAML/JSON.

## Outputs

Each tuning run writes:

```text
tuning_summary.yaml
solver_configuration_trials.jsonl
solver_configuration_trials.csv
best_solver_configuration.yaml
best_petsc_options.txt
configspace.yaml
configspace.json
```
