include_directories(utils PRIVATE ${CMAKE_BINARY_DIR}/include/core ${CMAKE_BINARY_DIR}/include)
AUX_SOURCE_DIRECTORY(./ utils_sources)
add_library(utils OBJECT ${utils_sources})
