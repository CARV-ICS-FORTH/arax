add_executable(producer producer.c)
target_include_directories(producer PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(producer vine rt)

add_executable(consumer consumer.c)
target_include_directories(consumer PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(consumer vine rt)

add_executable(queue_test queue_test.c)
target_include_directories(queue_test PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(queue_test vine rt pthread)
