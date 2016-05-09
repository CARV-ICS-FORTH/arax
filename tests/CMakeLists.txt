find_package(Check)

include_directories(${CHECK_INCLUDE_DIRS})

add_executable(queue queue.c)
target_include_directories(queue PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(queue vine rt ${CHECK_LIBRARIES})

add_executable(list list.c)
target_include_directories(list PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(list vine rt ${CHECK_LIBRARIES})

add_executable(config config.c)
target_include_directories(config PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(config vine rt ${CHECK_LIBRARIES})

add_executable(vine_talk vine_talk.c)
target_include_directories(vine_talk PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(vine_talk vine rt ${CHECK_LIBRARIES})
