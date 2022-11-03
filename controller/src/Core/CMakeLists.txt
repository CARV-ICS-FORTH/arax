include(ELFIO)

aux_source_directory(./ core_sources)
option(MIGRATION "Enable migration accross heterogenous accelerators." OFF)
option(ERROR_CHECKING "Enable error checking." OFF)
add_library(cntrl_core OBJECT ${core_sources})
target_link_libraries(cntrl_core PUBLIC elfio)

