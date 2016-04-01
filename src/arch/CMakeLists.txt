# TODO: Add option to choose architecture
AUX_SOURCE_DIRECTORY(shm arch_sources)
add_library(arch OBJECT ${arch_sources})
target_include_directories(arch PRIVATE ${VINE_SOURCE_DIR}/src)
