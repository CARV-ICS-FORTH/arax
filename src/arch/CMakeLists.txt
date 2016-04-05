# TODO: Add option to choose architecture
include_directories(arch PRIVATE ${VINE_SOURCE_DIR}/src ${VINE_SOURCE_DIR})

AUX_SOURCE_DIRECTORY(shm arch_sources)
add_library(arch OBJECT ${arch_sources})

LIST(REMOVE_ITEM arch_sources shm/vine_talk.c)
add_library(arch_dummy OBJECT ${arch_sources})
