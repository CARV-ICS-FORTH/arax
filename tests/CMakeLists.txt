add_executable(queue queue.c)
target_include_directories(queue PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(queue vine rt check)

add_executable(list list.c)
target_include_directories(list PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(list vine rt check)

add_executable(config config.c)
target_include_directories(config PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(config vine rt check)
