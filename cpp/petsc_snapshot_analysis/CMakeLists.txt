add_executable(ksptune-petsc-snapshot-analysis petsc_snapshot_analysis.cpp)
target_compile_features(ksptune-petsc-snapshot-analysis PRIVATE cxx_std_17)
ksptune_use_petsc(ksptune-petsc-snapshot-analysis PRIVATE)
target_link_libraries(ksptune-petsc-snapshot-analysis PRIVATE ksptune-json ksptune-snapshot-io)

install(TARGETS ksptune-petsc-snapshot-analysis RUNTIME DESTINATION bin)
