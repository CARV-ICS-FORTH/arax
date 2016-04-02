AUX_SOURCE_DIRECTORY(./ utils_sources)
add_library(utils OBJECT ${utils_sources})
target_include_directories(utils PRIVATE ${VINE_SOURCE_DIR})
