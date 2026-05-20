find_package(PkgConfig REQUIRED)

if(DEFINED ENV{PETSC_DIR})
  set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(PETSC REQUIRED petsc)

add_library(ksptune-petsc-snapshot STATIC
  petsc_snapshot_writer.cpp
)

add_library(KSPTune::petsc_snapshot ALIAS ksptune-petsc-snapshot)

set_target_properties(ksptune-petsc-snapshot
  PROPERTIES
    BUILD_RPATH "${PETSC_LIBRARY_DIRS}"
    INSTALL_RPATH "${PETSC_LIBRARY_DIRS}"
)

target_compile_features(ksptune-petsc-snapshot
  PUBLIC
    cxx_std_11
)

target_include_directories(ksptune-petsc-snapshot
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PETSC_INCLUDE_DIRS}
)

target_compile_options(ksptune-petsc-snapshot
  PUBLIC
    ${PETSC_CFLAGS_OTHER}
)

target_link_directories(ksptune-petsc-snapshot
  PUBLIC
    ${PETSC_STATIC_LIBRARY_DIRS}
    ${PETSC_LIBRARY_DIRS}
)

target_link_options(ksptune-petsc-snapshot
  PUBLIC
    ${PETSC_LDFLAGS_OTHER}
    ${PETSC_STATIC_LDFLAGS_OTHER}
)

target_link_libraries(ksptune-petsc-snapshot
  PUBLIC
    ${PETSC_STATIC_LIBRARIES}
)

add_executable(ksptune-petsc-snapshot-check
  petsc_snapshot_writer_check.cpp
)

target_link_libraries(ksptune-petsc-snapshot-check
  PRIVATE
    KSPTune::petsc_snapshot
)

if(BUILD_TESTING)
  add_test(
    NAME ksptune-petsc-snapshot-check
    COMMAND ksptune-petsc-snapshot-check
      -snapshot_output_dir
      ${CMAKE_CURRENT_BINARY_DIR}/snapshot_writer_check
  )
endif()

install(TARGETS ksptune-petsc-snapshot
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include
)
