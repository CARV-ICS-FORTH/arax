find_package(Check)
find_package(Threads)

include_directories(${CHECK_INCLUDE_DIRS})

add_executable(queue_unit queue.c)
target_include_directories(queue_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(queue_unit vine rt ${CHECK_LIBRARIES})

add_executable(list_unit list.c)
target_include_directories(list_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(list_unit vine rt ${CHECK_LIBRARIES})

add_executable(config_unit config.c)
target_include_directories(config_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(config_unit vine rt ${CHECK_LIBRARIES})

add_executable(vine_object_unit vine_object.c)
target_include_directories(vine_object_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(vine_object_unit vine rt ${CHECK_LIBRARIES})


add_executable(vine_talk_unit vine_talk.c)
target_include_directories(vine_talk_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(vine_talk_unit vine rt ${CHECK_LIBRARIES})

add_executable(alloc_perf_unit alloc_perf.c)
target_include_directories(alloc_perf_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(alloc_perf_unit vine rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(tracer_unit tracer.c)
target_include_directories(tracer_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(tracer_unit vine rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
