# TODO: Add option to choose architecture
include_directories(async PRIVATE ${VINE_SOURCE_DIR}/src ${VINE_SOURCE_DIR}/src/core ${VINE_SOURCE_DIR}/include)

set(async_architecture spin CACHE STRING "Mechanism used for async API")
set_property(CACHE async_architecture PROPERTY STRINGS spin mutex ivshmem )

aux_source_directory(${async_architecture} async_sources)
add_library(async OBJECT ${async_sources})
