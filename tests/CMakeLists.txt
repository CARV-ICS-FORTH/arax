find_package(Check)
find_package(Threads)

add_library(testing testing.c)
target_link_libraries(testing vine_st ${CHECK_LIBRARIES})
target_include_directories(testing
	PUBLIC
	${CHECK_INCLUDE_DIRS}
	PRIVATE
	${CHECK_INCLUDE_DIRS}
)

function(define_test NAME)
	add_executable(${NAME}_unit ${NAME}.c)
	target_link_libraries(${NAME}_unit testing)
	add_test(${NAME} ${CMAKE_BINARY_DIR}/tests/${NAME}_unit)
endfunction(define_test)

define_test(queue)
define_test(list)
define_test(kv)
define_test(config)
define_test(vine_object)
define_test(vine_talk)
define_test(alloc_perf)
define_test(async)
define_test(mmap)
define_test(vine_throttle)
define_test(accel_throttle)
define_test(pipe_throttle)
define_test(system)
