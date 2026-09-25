find_package(PkgConfig REQUIRED)

if(DEFINED ENV{PETSC_DIR})
  set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(PETSC REQUIRED petsc)

add_library(ksptune-petsc INTERFACE)
target_include_directories(ksptune-petsc INTERFACE ${PETSC_INCLUDE_DIRS})
target_compile_options(ksptune-petsc INTERFACE ${PETSC_CFLAGS_OTHER})
target_link_directories(ksptune-petsc INTERFACE ${PETSC_STATIC_LIBRARY_DIRS} ${PETSC_LIBRARY_DIRS})
target_link_options(ksptune-petsc INTERFACE ${PETSC_LDFLAGS_OTHER} ${PETSC_STATIC_LDFLAGS_OTHER})
target_link_libraries(ksptune-petsc INTERFACE ${PETSC_STATIC_LIBRARIES})

function(ksptune_use_petsc target visibility)
  target_link_libraries(${target} ${visibility} ksptune-petsc)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${PETSC_LIBRARY_DIRS}"
    INSTALL_RPATH "${PETSC_LIBRARY_DIRS}"
  )
endfunction()
