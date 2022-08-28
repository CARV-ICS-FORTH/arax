include_directories(arch PRIVATE ${ARAX_SOURCE_DIR}/src ${ARAX_SOURCE_DIR})

set(target_architecture
    shm
    CACHE STRING "Target architecture (shm)")
mark_as_advanced(target_architecture)
set_property(CACHE target_architecture PROPERTY STRINGS shm)

aux_source_directory(${target_architecture} arch_sources)
add_library(arch OBJECT ${arch_sources})

target_include_directories(arch PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/async
)
