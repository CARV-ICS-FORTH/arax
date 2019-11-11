find_package(Check)
find_package(Threads)

include_directories(${CHECK_INCLUDE_DIRS})

add_executable(queue_unit queue.c)
target_include_directories(queue_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(queue_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(list_unit list.c)
target_include_directories(list_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(list_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(kv_unit kv.c)
target_include_directories(kv_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(kv_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(config_unit config.c)
target_include_directories(config_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(config_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(vine_object_unit vine_object.c)
target_include_directories(vine_object_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(vine_object_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(vine_talk_unit vine_talk.c)
target_include_directories(vine_talk_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(vine_talk_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(alloc_perf_unit alloc_perf.c)
target_include_directories(alloc_perf_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(alloc_perf_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(async_unit async.c)
target_include_directories(async_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(async_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(tracer_unit tracer.c)
target_include_directories(tracer_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(tracer_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(mmap_unit mmap.c)
target_include_directories(mmap_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(mmap_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(break_unit break.c)
target_include_directories(break_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(break_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(vine_throttle_unit vine_throttle.c)
target_include_directories(vine_throttle_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(vine_throttle_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(accel_throttle_unit accel_throttle.c)
target_include_directories(accel_throttle_unit PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(accel_throttle_unit vine_st rt ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})