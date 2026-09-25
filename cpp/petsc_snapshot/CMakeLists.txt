add_library(ksptune-petsc-snapshot INTERFACE)

add_library(KSPTune::petsc_snapshot ALIAS ksptune-petsc-snapshot)

target_compile_features(ksptune-petsc-snapshot INTERFACE cxx_std_11)
target_include_directories(ksptune-petsc-snapshot INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
  $<INSTALL_INTERFACE:include>
)
if(BUILD_TESTING)
  add_executable(ksptune-petsc-snapshot-check petsc_snapshot_writer_check.cpp)
  target_link_libraries(ksptune-petsc-snapshot-check PRIVATE KSPTune::petsc_snapshot)
  ksptune_use_petsc(ksptune-petsc-snapshot-check PRIVATE)
  add_test(
    NAME ksptune-petsc-snapshot-check
    COMMAND ${CMAKE_COMMAND} -E env
      KSP_EXPORT_ENABLE=1
      KSPTUNE_SNAPSHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/snapshot_writer_check
      KSPTUNE_SNAPSHOT_SOLVES=0,2
      KSPTUNE_SNAPSHOT_PREFIX=check
      $<TARGET_FILE:ksptune-petsc-snapshot-check>
  )
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ksptune/petsc_snapshot.hpp
  DESTINATION include/ksptune
)
