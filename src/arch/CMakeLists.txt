include_directories(arch PRIVATE ${VINE_SOURCE_DIR}/src ${VINE_SOURCE_DIR})

set(target_architecture shm CACHE STRING "Target architecture (shm or dummy)")
set_property(CACHE target_architecture PROPERTY STRINGS shm dummy)


aux_source_directory(${target_architecture} arch_sources)
add_library(arch OBJECT ${arch_sources})
