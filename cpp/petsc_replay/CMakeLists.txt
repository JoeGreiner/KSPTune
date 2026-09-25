add_executable(ksptune-petsc-replay petsc_replay.cpp replay_cache.cpp)
target_compile_features(ksptune-petsc-replay PRIVATE cxx_std_17)
ksptune_use_petsc(ksptune-petsc-replay PRIVATE)
target_link_libraries(ksptune-petsc-replay PRIVATE ksptune-json ksptune-snapshot-io)

install(TARGETS ksptune-petsc-replay RUNTIME DESTINATION bin)
