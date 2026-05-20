cmake_minimum_required(VERSION 3.18)

project(KSPTune LANGUAGES CXX)

include(CTest)

option(KSPTUNE_BUILD_PETSC_SNAPSHOT "Build the PETSc KSP snapshot writer library" ON)
option(KSPTUNE_BUILD_PETSC_REPLAY "Build the PETSc KSP replay executable" ON)
option(KSPTUNE_BUILD_PETSC_SNAPSHOT_ANALYSIS "Build the PETSc snapshot analysis executable" ON)

if(KSPTUNE_BUILD_PETSC_SNAPSHOT)
  add_subdirectory(cpp/petsc_snapshot)
endif()

if(KSPTUNE_BUILD_PETSC_REPLAY)
  add_subdirectory(cpp/petsc_replay)
endif()

if(KSPTUNE_BUILD_PETSC_SNAPSHOT_ANALYSIS)
  add_subdirectory(cpp/petsc_snapshot_analysis)
endif()
