include_directories(arch PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR})

set(target_architecture
    shm
    CACHE STRING "Target architecture (shm)")
mark_as_advanced(target_architecture)
set_property(CACHE target_architecture PROPERTY STRINGS shm)

aux_source_directory(${target_architecture} arch_sources)
add_library(arch OBJECT ${arch_sources})

target_include_directories(arch PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_BINARY_DIR}/include/async
)
