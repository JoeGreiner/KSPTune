# PETSc Dump Fixtures

PETSc binary dump fixtures are intentionally not committed to this repository.
They are useful for optional replay checks, but even small real matrices are too
large for a simple source-only GitHub repository.

Future versions may provide a download script that populates:

```text
examples/petsc_dumps/downloaded/opencarp_two_steps
examples/petsc_dumps/downloaded/ceps_two_steps
```

Until then, run KSPTune against your own snapshot directory:

```bash
ksptune tune \
  --snapshot-directory /path/to/snapshots \
  --parameter-search-space petsc.hypre-basic \
  --output-directory runs/my-tuning-run
```
