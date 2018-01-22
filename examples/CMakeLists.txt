include_directories(producer PRIVATE ${VINE_SOURCE_DIR}/src)

add_executable(producer producer.c)
target_link_libraries(producer vine rt pthread)

add_executable(consumer consumer.c)
target_link_libraries(consumer vine_st rt pthread)

add_executable(queue_test queue_test.c)
target_link_libraries(queue_test vine_st rt pthread)
