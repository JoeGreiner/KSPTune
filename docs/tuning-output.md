# Reading the tuning output

<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>trial 049/050 <span class="terminal-cyan">[f4ec28]</span> <span class="terminal-green">ok</span>  objective: <span class="terminal-yellow terminal-bold">3.66359s</span>  (best: <span class="terminal-cyan terminal-bold">1.43689s</span>)
  solver: ksp=gmres pc=hypre side=left norm=preconditioned rtol=1e-8 atol=1e-8 hypre=boomeramg bamg_strong=0.651845 pc_hypre_boomeramg_P_max=3 pc_hypre_boomeramg_agg_nl=0 pc_hypre_boomeramg_coarsen_type=HMIS
          pc_hypre_boomeramg_interp_type=ext+i pc_hypre_boomeramg_relax_type_all=l1scaled-Jacobi pc_hypre_boomeramg_truncfactor=0.15
  runtime: solve=10.991s (samples=3, avg=3.664s, sem=0.217s rel_sem=5.9%, min-max=3.438s-4.097s)  setup=7.575s (ksp-cache=2/1)  wall=18.669s
</code></pre>

`[f4ec28]` is the short identifier derived from the solver configuration (to match with SMAC logs), with its parameter shown.
`ok/failed` refers to replay success, convergence, and residual checks; failures also include timeouts, PETSc/process errors, and invalid replay results. `best` is the lowest successful objective so far for all trials.

| Field | Meaning                                                          |
| --- |------------------------------------------------------------------|
| `solve=10.991s` | Total solve time (all solves).                                   |
| `samples=3`, `avg=3.664s` | spans individual snapshots and repetitions; warmups are excluded. |
| `sem=0.217s` | Standard error of the mean                                       |
| `rel_sem=5.9%` | `100 × sem / avg`                                                |
| `min-max=3.438s-4.097s` | Fastest and slowest values.                                      |
| `setup=7.575s` | setup of solver/preconditioner.                                  |
| `ksp-cache=2/1` | Setup-cache hits/misses.                                         |
| `wall=18.669s` | Total wall time inside replay, including data loading etc.       |

Each solve time is the maximum across MPI ranks.

## Replay

<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>
replay: request=18.672s data-load=0.011s matrix=0.000s vectors=0.006s nullspace=0.000s true-residuals=0.084s replay-other=0.009s outer-overhead=0.002s matrix-cache=3/0
</code></pre>


Replay loads the exported systems, runs the solver, and checks the solutions.

| Field | Meaning                                                                           |
| --- |-----------------------------------------------------------------------------------|
| `request` | Complete replay-request time.                                                     |
| `data-load` | load/cache of data.                                                               |
| `matrix`, `vectors` | Matrix and vector preparation within `data-load`                                  |
| `nullspace` | Preparing nullspace information.                                                  |
| `true-residuals` | Time spent computing true residuals before and after solves.                       |
| `replay-other` | Remaining time not measured elsewhere, with warmup. |
| `outer-overhead` | `request - wall`.                                     |
| `matrix-cache=3/0` | Matrix-cache hits/misses                            |

## Memory
<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>
memory: rss_sum sample_peak=9.6GB request_delta=107.3MB pcsetup_delta=0B solve_delta=436.9MB max_node=9.6GB max_rank=861.3MB
</code></pre>

RSS is resident RAM. `rss_sum` sums it across MPI ranks.

| Field | Meaning |
| --- | --- |
| `sample_peak` | Largest sampled total RSS during the request. |
| `request_delta` | Increase in total RSS from request start to end. |
| `pcsetup_delta`, `solve_delta` | Largest observed increase in total RSS across a setup or solve phase, respectively. |
| `max_node`, `max_rank` | Largest sampled RSS on one node (sum of its ranks) or one rank. |

`pcsetup_delta=0B` means no measured RSS increase, not zero preconditioner memory. Deltas are not additive.

## PC diagnostics

<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>
pc: hypre setup=1 levels=11 grid=1.55988 op=2.3434 interp=0.178763 max_op_nnz/row=66.8 max_interp_nnz/row=1.89 fine_nnz=6.815e+07 total_nnz=1.597e+08 coarse_rows=3 interp_nnz=1.218e+07 max_row_nnz=264 offdiag_nnz=3.974e+07 coarsest_nonzeros=9 finest_rows=4.293e+06 total_rows=6.696e+06
</code></pre>

The fields depend on the preconditioner. **This example shows diagnostics
specific to hypre AMG.** `nnz` denotes nonzero matrix
entries; operator matrices describe the systems on each level, and interpolation
matrices transfer corrections between levels.

| Field | Meaning                                                                                                                   |
| --- |---------------------------------------------------------------------------------------------------------------------------|
| `setup=1` | First recorded setup, not a duration.                                                                                     |
| `levels=11` | Number of multigrid levels.                                                                                               |
| `grid=1.55988` | Total operator rows / finest-level rows, complexity measure                                                               |
| `op=2.3434` | Total operator nonzeros / finest-level nonzeros, complexity measure                                                       |
| `interp=0.178763` | Total interpolation nonzeros / finest-level operator nonzeros, complexity measure                                         |
| `finest_rows`, `total_rows`, `coarse_rows` | Operator rows.                                                                                                            |
| `fine_nnz`, `total_nnz`, `coarsest_nonzeros` | Operator nonzeros.                                                                                                        |
| `interp_nnz` | Total interpolation nonzeros.                                                                                             |
| `max_op_nnz/row`, `max_interp_nnz/row` | Largest **level-average** nonzeros.                                                                                       |
| `max_row_nnz` | Largest individual row's nonzero.                                                                                         |
| `offdiag_nnz` | Total nonzeros in MPI off-process blocks, referring to columns owned by other ranks. |

Higher complexity generally means more storage and work per multigrid cycle,
but may reduce solver iterations.

## MPI diagnostics

<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>
mpi: solve msgs=8.448e+03 bytes=1.1GB reductions=198 avg=135.1KB/msg
</code></pre>

Communication logged by PETSc during measured solves, excluding setup and
explicit residual checks:

| Field | Meaning |
| --- | --- |
| `msgs` | Messages across all ranks, counting each send/receive pair once. |
| `bytes`, `avg` | Total message volume and average bytes per message. |
| `reductions` | Global reduction count, averaged across ranks, for operations such as norms and dot products. |

## Tolerances and convergence

<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>
tolerances: rtol=1e-8 atol=1e-8 dtol=10000 max_it=10000 true_abs_gate<=1e-4 true_rel_gate<=1e-4
diagnostics: iters=61 (samples=3, mean=20.3333) reason=KSP_CONVERGED_RTOL
</code></pre>
| Field | Meaning                                                                             |
| --- |-------------------------------------------------------------------------------------|
| `rtol`, `atol` | PETSc's relative and absolute convergence tolerances.                               |
| `dtol` | Divergence threshold.                                                               |
| `max_it` | Maximum iteration.                                                                  |
| `true_abs_gate`, `true_rel_gate` | Additional KSPTune's limits based on the final absolute and relative true residual. |
| `iters=61`, `samples=3`, `mean=20.3333` | iters=iterations over all samples, mean accordingly                                 |
| `reason=KSP_CONVERGED_RTOL` | PETSc's convergence criteria.                                                       |

## Residuals

<pre class="terminal-output" tabindex="0" aria-label="Example tuning output"><code>
true residual: abs_initial=0.00127649 rel_initial=0.379197 abs_final=3.430e-09 rel_final=5.924e-07
</code></pre>

| Line | Meaning                                                                                                                                                                                                                                 |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `true residual` | Residual of the original system, `r = b - A*x`. `abs_initial`/`abs_final` are its norms before/after solving. `rel_initial`/`rel_final` is divided by norm of `b`                                                                       |
| `ksp residual` | Residual norm monitored by PETSc (often preconditioned). `abs_initial`/`abs_final` as above. `rtol_ref` is PETSc's reference for relative convergence and can differ from `abs_initial`. Per solve, `rel_final = abs_final / rtol_ref`. |

Displayed values are averages. Accuracy gates use the largest
individual residuals.

The KSP and true residuals use different norms. Thus, `ksp rel_final=6.419e-09`
can meet `rtol=1e-8` while `true rel_final=5.924e-07` is larger. The latter is
checked separately against `true_rel_gate<=1e-4`.

## Saved runs

`tuning_run.yaml` stores settings and derived metadata separately (format version 2);
`configspace.json` stores the search space. Resume requires a version-2 run.
Use `--hard-timeout-sec` for the request timeout; the former `--timeout-sec` alias was removed from `tune` and `resume`.
