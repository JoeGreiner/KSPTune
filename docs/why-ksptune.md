# Why KSPTune?

While the optimisation software (SMAC) can be run directly run with a wrapper around the simulation software, KSPTune makes the tuning process more efficient and informative by:

- Isolated replay of individual snapshots, without the overhead of running the simulation software (e.g. openCARP)
- Cache for KSP objects, vectors, matrices; vectors and matrices persist between trials, KSP objects are reused within a trial
- Preconfigured parameter spaces
- Diagnostics (memory, communication, preconditioner)
- Soft stops for solve timeouts; failure handling and replay restart for OOM/process kills and hard timeouts
