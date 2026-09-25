# Installation

## Components

| Component | Purpose                                                                          |
| --- |----------------------------------------------------------------------------------|
| Python environment (`ksptune`) | run/resume/analyse tuning                                                        |
| `ksptune-petsc-replay` (C++) | minimal C++ PETSc application to run snapshots and get timings/diagnostics       |
| `ksptune-petsc-snapshot-analysis` (C++) | basic analysis, e.g. nullspace and rhs checks                                    |
| `include/ksptune/petsc_snapshot.hpp` (C++) | header for snapshot exports for integration in petsc applications, e.g. openCARP |

## Requirements

Requirements: Git, Conda, CMake >=3.18, Make (or Ninja), pkg-config, a C++17 compiler, and PETSc with MPI.

You may run into errors if the compiler used for PETSc differs from the one used for compiling KSPTune, ideally, please use the same.
Some of the supplied parameterspaces need external modules in PETSc, for example, the hypre preconditioners need hypre (duh).


## Install

### Clone the repository

```bash
git clone https://github.com/JoeGreiner/KSPTune.git
cd KSPTune
```

## Install the KSPTune Python Environment w/ conda

<details>
<summary>New to conda?</summary>

I'd strongly recommend installing KSPTune in a separated environment.

<p>Install <a href="https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html#installing-in-silent-mode">Miniconda</a>
on Linux/macOS (bash or zsh), if Conda is not installed:</p>

```bash
# please chose the right os
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
```

</details>

```bash
conda env create -f environment.yml
conda activate ksptune
python -m pip install -e .
```

## Build the C++ tools

Run from the repository directory. Use the MPI C++ compiler wrapper that belongs
to your PETSc installation (`mpicxx` below).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build -j
```

It builds:

```text
build/cpp/petsc_replay/ksptune-petsc-replay
build/cpp/petsc_snapshot_analysis/ksptune-petsc-snapshot-analysis
```

With the Python installation above `ksptune` should find these automatically.

If PETSc is not found, set its location, and rerun both CMake commands:

```bash
export PETSC_DIR=/path/to/petsc
export PKG_CONFIG_PATH="$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
```

For a PETSc source-tree build, use `$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig` instead for `$PKG_CONFIG_PATH`.
