AUX_SOURCE_DIRECTORY(./ core_sources)
add_library(core OBJECT ${core_sources})
target_include_directories(core PRIVATE ${VINE_SOURCE_DIR}/src)
