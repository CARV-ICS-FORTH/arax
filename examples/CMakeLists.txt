include_directories(${VINE_SOURCE_DIR}/src)

add_executable(ex1 ex1.c)
target_link_libraries(ex1 vine rt)

add_executable(ex2 ex2.c)
target_link_libraries(ex2 vine rt)

add_executable(ex1_dummy ex1.c)
target_link_libraries(ex1_dummy vine_dummy rt)

add_executable(ex2_dummy ex2.c)
target_link_libraries(ex2_dummy vine_dummy rt)

add_executable(queue_test queue_test.c)
target_link_libraries(queue_test vine rt pthread)
