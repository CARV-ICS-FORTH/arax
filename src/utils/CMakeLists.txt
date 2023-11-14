aux_source_directory(./ utils_sources)
add_library(utils OBJECT ${utils_sources})

include_directories(utils PRIVATE
    ${PROJECT_BINARY_DIR}/include/core
    ${PROJECT_BINARY_DIR}/include
)
