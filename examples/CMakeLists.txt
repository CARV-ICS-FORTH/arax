add_executable(producer producer.c)
target_include_directories(producer PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(producer vine rt)

add_executable(consumer consumer.c)
target_include_directories(consumer PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(consumer vine rt)

add_executable(producer_dummy producer.c)
target_link_libraries(producer_dummy vine_dummy rt)

add_executable(consumer_dummy consumer.c)
target_link_libraries(consumer_dummy vine_dummy rt)

add_executable(ex3_dummy ex3.c)
target_link_libraries(ex3_dummy vine_dummy rt)

add_executable(queue_test queue_test.c)
target_link_libraries(queue_test vine rt pthread)
