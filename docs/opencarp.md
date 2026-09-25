# Example: openCARP integration

## 1. Clone openCARP

```bash
git clone https://git.opencarp.org/openCARP/openCARP.git
cd openCARP
export ksptune_dir=/absolute/path/to/KSPTune
```

## 2. Add the exporter

Edit openCARP's root `CMakeLists.txt`:

```cmake
add_subdirectory(numerics) # <- existing openCARP code
if(ENABLE_PETSC) # <- addition start
    find_path(KSPTUNE_INCLUDE_DIR NAMES ksptune/petsc_snapshot.hpp)
    target_include_directories(numerics PUBLIC "${KSPTUNE_INCLUDE_DIR}")
endif() # <- addition end
```

In `numerics/petsc/SF_petsc_solver.h`, add this include inside `#ifdef WITH_PETSC`:

```cpp
#include <ksptune/petsc_snapshot.hpp>
```

Insert the export call immediately before the existing `KSPSolve`:

```cpp
PETSC_CHECK(KSPTuneExport(ksp, b.data, x.data)); // <-- add this
PETSC_CHECK(KSPSolve(ksp, b.data, x.data)); // <-- existing openCARP code
```

## 3. Build and install openCARP

Build openCARP with your preferred options, e.g. here with EMI in the release config.
Use the MPI C++ compiler matching your PETSc installation (`mpicxx` below).

```bash
conda activate ksptune
cmake -S . -B "build-ksptune/" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DENABLE_PETSC=ON -DENABLE_EMI_MODEL=ON \
  -DKSPTUNE_INCLUDE_DIR="$ksptune_dir/include"
cmake --build "build-ksptune/" -j
cmake --install "build-ksptune/"
```

## 4. Exporting snapshots from openCARP

If you want to follow the example on a sample dataset, you can find one here: [6x5x4 EMI example](https://www.joegreiner.de/assets/ksptune_example_data.zip).
Run from the extracted folder containing `parameters.par`, `solver_opts`, and the mesh files:

```bash
cd /path/to/extracted/example
export simulation_dir="$PWD/output"

KSP_EXPORT_ENABLE=1 \
KSPTUNE_SNAPSHOT_DIR="$simulation_dir/snapshots" \
KSPTUNE_SNAPSHOT_SOLVES=1,3,5 \
mpiexec -n 10 openCARP +F parameters.par
```

Export is disabled by default. Solve indices are numbered from 0.

The following environment variables can be used:


| Variable | Meaning                                            | Default |
| --- |----------------------------------------------------| --- |
| `KSP_EXPORT_ENABLE` | Enable: `1`, `true`, `on`, `yes`; disable with `0` | Disabled |
| `KSPTUNE_SNAPSHOT_DIR` | Output directory; required when enabled            | Unset |
| `KSPTUNE_SNAPSHOT_SOLVES` | Zero-based solve indices: `1,5,10`, or `all`       | `0` |
| `KSPTUNE_SNAPSHOT_PREFIX` | Optional filename prefix                           | `ksp` |

KSPTune exports the following snapshot files:

```text
A_<matrix_id>.bin
A_<matrix_id>.txt
b_<snapshot_id>.bin
x0_<snapshot_id>.bin
s_<snapshot_id>.txt
```

KSPTune will automatically detect a non-changing matrix and only save it once. x0 is the initial guess, b the rhs.

By the way: You can merge exports into one combined scenario by simply copying them into one folder, keeping their filenames:

```bash
scenario=/path/to/scenario
mkdir -p "$scenario"
cp -n /path/to/run1/snapshots/*.bin /path/to/run1/snapshots/*.txt "$scenario/"
cp -n /path/to/run2/snapshots/*.bin /path/to/run2/snapshots/*.txt "$scenario/"
```
Generated filenames distinguish runs and KSPs. Repeated solve indices are allowed.

## 5. Tune snapshots

Important: Use the same number of processes as in the openCARP run.
The parameter space petsc.hypre-basic is already supplied by the package, but you can also [add your own](parameter-files.md#custom-file).

```bash
cd "$ksptune_dir"
ksptune tune \
  --snapshot-directory "$simulation_dir/snapshots" \
  --parameter-search-space petsc.hypre-basic \
  --output-directory "$simulation_dir/ksptune" \
  --np 10 --workers 1 --trials 50 --warmup 0 --repeat 1
```

`--workers` specifies the number of parallel trials. For example, two workers
with `--np 10` use 20 MPI processes. `--warmup` selects unmeasured warmup runs;
`--repeat` selects measured repetitions. `--trials` sets the tuning budget.

## 6. Obtain the winning parameter configuration

After the run, get the parameters from stdout, or run:

```bash
ksptune export-petsc-options --tuning-run "$simulation_dir/ksptune"
```
## Further Analysis

```bash
ksptune analyze-parameter-importance --tuning-run "$simulation_dir/ksptune"
```
For all commands and options:

```bash
ksptune --help
ksptune tune --help
ksptune petsc-options -pc_type hypre -pc_hypre_type boomeramg
```
