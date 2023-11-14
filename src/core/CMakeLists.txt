include_directories(${PROJECT_SOURCE_DIR}/src)

aux_source_directory(./ core_sources)
add_library(core OBJECT ${core_sources})

target_include_directories(core PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_BINARY_DIR}/include/async
    ${PROJECT_SOURCE_DIR}/src/async/${async_architecture}
)
