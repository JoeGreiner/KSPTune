add_library(ksptune-json INTERFACE)
target_include_directories(ksptune-json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ksptune-json INTERFACE nlohmann_json::nlohmann_json)

add_library(ksptune-snapshot-io STATIC snapshot_io.cpp)
target_include_directories(ksptune-snapshot-io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ksptune-snapshot-io PUBLIC cxx_std_17)
ksptune_use_petsc(ksptune-snapshot-io PUBLIC)
