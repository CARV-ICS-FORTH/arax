include_directories(${VINE_SOURCE_DIR}/src)

AUX_SOURCE_DIRECTORY(./ core_sources)
add_library(core_dummy OBJECT ${core_sources})

LIST(REMOVE_ITEM core_sources vine_talk_dummy.c)
add_library(core OBJECT ${core_sources})
