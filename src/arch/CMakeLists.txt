# TODO: Add option to choose architecture
include_directories(arch PRIVATE ${VINE_SOURCE_DIR}/src ${VINE_SOURCE_DIR})


set(target_architecture shm CACHE STRING "Target architecture (shm or dummy)")
aux_source_directory(${target_architecture} arch_sources)
add_library(arch OBJECT ${arch_sources})
