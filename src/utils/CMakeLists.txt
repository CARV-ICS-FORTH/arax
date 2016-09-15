include_directories(utils PRIVATE ${VINE_SOURCE_DIR}/src ${VINE_SOURCE_DIR}/src/core ${VINE_SOURCE_DIR}/include)
AUX_SOURCE_DIRECTORY(./ utils_sources)
add_library(utils OBJECT ${utils_sources})
