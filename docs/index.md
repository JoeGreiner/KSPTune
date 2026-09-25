# KSPTune

KSPTune is a tool to tune [PETSc](https://petsc.org/release/) solver and preconditioner parameters using the bayesian optimisation package [SMAC3](https://jmlr.org/papers/v23/21-0888.html).
We provide a header-only C++ library that can be used to easily export snapshots (Matrix, initial guess, rhs vector, ownership ranges, etc.) of the solver state to be tuned.
With this, it can be easily integrated into existing PETSc applications, such as [openCARP](https://git.opencarp.org/openCARP/openCARP) or [CEPS](https://carmen.gitlabpages.inria.fr/ceps/).
After the snapshots are exported, a python interface runs an optimisation on the snapshots on user-configured parameterspaces. It is compatible with HPC systems/SLURM.

- [Why KSPTune?](why-ksptune.md)
- [Installation](installation.md)
- [openCARP example](opencarp.md)
- [Reading the tuning output](tuning-output.md)
- [Parameter files](parameter-files.md)
- [Citation](citation.md)
