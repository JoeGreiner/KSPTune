cmake_minimum_required(VERSION 3.18)

project(KSPTune LANGUAGES CXX)

include(CTest)

option(KSPTUNE_BUILD_PETSC_REPLAY "Build the PETSc KSP replay executable" ON)
option(KSPTUNE_BUILD_PETSC_SNAPSHOT_ANALYSIS "Build the PETSc snapshot analysis executable" ON)

if(BUILD_TESTING OR KSPTUNE_BUILD_PETSC_REPLAY OR KSPTUNE_BUILD_PETSC_SNAPSHOT_ANALYSIS)
  include(cmake/PETSc.cmake)
endif()

if(KSPTUNE_BUILD_PETSC_REPLAY OR KSPTUNE_BUILD_PETSC_SNAPSHOT_ANALYSIS)
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  set(JSON_BuildTests OFF)
  set(JSON_Install OFF)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    TLS_VERIFY TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
  add_subdirectory(cpp/common)
endif()

add_subdirectory(cpp/petsc_snapshot)

if(KSPTUNE_BUILD_PETSC_REPLAY)
  add_subdirectory(cpp/petsc_replay)
endif()

if(KSPTUNE_BUILD_PETSC_SNAPSHOT_ANALYSIS)
  add_subdirectory(cpp/petsc_snapshot_analysis)
endif()
