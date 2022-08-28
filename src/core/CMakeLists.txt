include_directories(${ARAX_SOURCE_DIR}/src)

aux_source_directory(./ core_sources)
add_library(core OBJECT ${core_sources})

target_include_directories(core PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/async
    ${ARAX_SOURCE_DIR}/src/async/${async_architecture}
)
