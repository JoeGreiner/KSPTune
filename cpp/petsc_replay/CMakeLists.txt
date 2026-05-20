find_package(PkgConfig REQUIRED)

if(DEFINED ENV{PETSC_DIR})
  set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(PETSC REQUIRED petsc)

add_executable(ksptune-petsc-replay petsc_replay.cpp)

set_target_properties(ksptune-petsc-replay
  PROPERTIES
    BUILD_RPATH "${PETSC_LIBRARY_DIRS}"
    INSTALL_RPATH "${PETSC_LIBRARY_DIRS}"
)

target_compile_features(ksptune-petsc-replay
  PRIVATE
    cxx_std_17
)

target_include_directories(ksptune-petsc-replay
  PRIVATE
    ${PETSC_INCLUDE_DIRS}
)

target_compile_options(ksptune-petsc-replay
  PRIVATE
    ${PETSC_CFLAGS_OTHER}
)

target_link_directories(ksptune-petsc-replay
  PRIVATE
    ${PETSC_STATIC_LIBRARY_DIRS}
    ${PETSC_LIBRARY_DIRS}
)

target_link_options(ksptune-petsc-replay
  PRIVATE
    ${PETSC_LDFLAGS_OTHER}
    ${PETSC_STATIC_LDFLAGS_OTHER}
)

target_link_libraries(ksptune-petsc-replay
  PRIVATE
    ${PETSC_STATIC_LIBRARIES}
)

install(TARGETS ksptune-petsc-replay
  RUNTIME DESTINATION bin
)
