# KSPTune: Bayesian Optimization for PETSc KSP Solvers

<p align="center">
  <img src="docs/assets/KSPTune-logo.svg" alt="KSPTune logo" width="200" height="185">
</p>

KSPTune is a tool to tune [PETSc](https://petsc.org/release/) solver and preconditioner parameters using the bayesian optimisation package [SMAC3](https://jmlr.org/papers/v23/21-0888.html).
We provide a header-only C++ library that can be used to easily export snapshots (Matrix, initial guess, rhs vector, ownership ranges, etc.) of the solver state to be tuned.
With this, it can be easily integrated into existing PETSc applications, such as [openCARP](https://git.opencarp.org/openCARP/openCARP) or [CEPS](https://carmen.gitlabpages.inria.fr/ceps/).
After the snapshots are exported, a python interface runs an optimisation on the snapshots on user-configured parameterspaces. It is compatible with HPC systems/SLURM.

We observed substantial speedups over default parameters on cardiac simulation using the monodomain, bidomain, and EMI model, but the method is general and can be applied to any PETSc solver.

[Documentation](https://joegreiner.github.io/KSPTune/) /
[Installation](https://joegreiner.github.io/KSPTune/installation/) / [openCARP step-by-step example](https://joegreiner.github.io/KSPTune/opencarp/) /
[Parameter files](https://joegreiner.github.io/KSPTune/parameter-files/)

# Experiments: speedups over default parameters

These speedups depend, of course, on the problem size and the solver configuration.
Especially algebraic multigrid methods (gamg, hypre boomeramg) have many important parameters that can be tuned.

* Monodomain, atrial simulations: **1.7x speedup** (tuned: CG+SPAI, default: bjacobi+ilu+cg, openCARP)
* Bidomain, synthetic cube geometry with conductivity hetereogeneity: **11.2x speedup** (tuned: fgmres/fieldsplit hupre AMG, default: bjacobi+ilu+cg, CEPS)
* EMI, synthetic myocyte meshes: **4.0x speedup** (tuned: fgmres+hypre AMG, default: hypre AMG + cg, openCARP)

# Citation

If you find KSPTune useful in your research, please consider citing the following articles:

Paper that first used KSPTune:

Greiner, J., Sankarankutty, A. C., Seemann, G., Seidel, T., and Sachse, F. B. (2018). [Confocal microscopy-based estimation of parameters for computational modeling of electrical conduction in the normal and infarcted heart](https://doi.org/10.3389/fphys.2018.00239). *Frontiers in Physiology*, 9, Article 239.

<details>
<summary>BibTeX</summary>

```bibtex
@article{Greiner2018,
  author  = {Greiner, Joachim and Sankarankutty, Aparna C. and Seemann, Gunnar
             and Seidel, Thomas and Sachse, Frank B.},
  title   = {Confocal Microscopy-Based Estimation of Parameters for Computational
             Modeling of Electrical Conduction in the Normal and Infarcted Heart},
  journal = {Frontiers in Physiology},
  year    = {2018},
  volume  = {9},
  pages   = {239},
  doi     = {10.3389/fphys.2018.00239},
  url     = {https://doi.org/10.3389/fphys.2018.00239}
}
```

</details>

SMAC3, the optimisation package under the hood:

Lindauer, M., Eggensperger, K., Feurer, M., Biedenkapp, A., Deng, D., Benjamins, C., Ruhkopf, T., Sass, R., and Hutter, F. (2022). [SMAC3: A versatile Bayesian optimization package for hyperparameter optimization](https://jmlr.org/papers/v23/21-0888.html). *Journal of Machine Learning Research*, 23(54), 1–9.

<details>
<summary>BibTeX</summary>

```bibtex
@article{Lindauer2022SMAC3,
  author  = {Lindauer, Marius and Eggensperger, Katharina and Feurer, Matthias
             and Biedenkapp, Andr{\'e} and Deng, Difan and Benjamins, Carolin
             and Ruhkopf, Tim and Sass, Ren{\'e} and Hutter, Frank},
  title   = {{SMAC3}: A Versatile {Bayesian} Optimization Package for Hyperparameter
             Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {54},
  pages   = {1--9},
  url     = {https://jmlr.org/papers/v23/21-0888.html}
}
```

</details>
