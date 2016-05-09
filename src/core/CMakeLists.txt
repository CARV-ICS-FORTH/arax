include_directories(${VINE_SOURCE_DIR}/src)

aux_source_directory(./ core_sources)
add_library(core OBJECT ${core_sources})
