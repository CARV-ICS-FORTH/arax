add_executable(ex1 ex1.c)
target_include_directories(ex1 PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(ex1 vine rt)

add_executable(ex2 ex2.c)
target_include_directories(ex2 PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(ex2 vine rt)

add_executable(queue_test queue_test.c)
target_include_directories(queue_test PRIVATE ${VINE_SOURCE_DIR}/src)
target_link_libraries(queue_test vine rt pthread)
