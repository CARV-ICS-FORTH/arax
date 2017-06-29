include_directories(producer PRIVATE ${VINE_SOURCE_DIR}/src)

add_executable(producer producer.c)
target_link_libraries(producer vine rt pthread)

add_executable(consumer consumer.c)
target_link_libraries(consumer vine rt)

add_executable(queue_test queue_test.c)
target_link_libraries(queue_test vine rt pthread)

add_executable(vdf vdf.cpp)
set_target_properties(vdf PROPERTIES COMPILE_FLAGS -std=c++11)
target_link_libraries(vdf vine rt pthread -lPocoNet -lPocoFoundation -lPocoUtil)
