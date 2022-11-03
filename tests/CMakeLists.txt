find_package(Catch2 QUIET)

option(BUILD_TESTS "Build unit tests" ON) # skip build_check

if(BUILD_TESTS)
	find_package(Threads)

	add_library(testing testing.cpp)
	target_link_libraries(testing arax_st Catch2::Catch2)

	function(define_test NAME)
		add_executable(${NAME}_unit ${NAME}.cpp)
		target_link_libraries(${NAME}_unit testing Catch2::Catch2)
		add_test(${NAME} ${CMAKE_BINARY_DIR}/tests/${NAME}_unit)
	endfunction(define_test)

	define_test(bitmap)
	define_test(queue)
	define_test(list)
	define_test(kv)
	define_test(config)
	define_test(arax_object)
	define_test(arax_data)
	define_test(arax_accel)
	define_test(arax_talk)
	define_test(arax_mmap)
	define_test(arax_plot)
	define_test(alloc_perf)
	define_test(async)
	define_test(mmap)
	define_test(handlers)
	if(ARAX_THROTTLE_ENABLE)
	define_test(arax_throttle)
	define_test(accel_throttle)
	define_test(pipe_throttle)
	endif()

	define_test(system)

	# struct_interop is a bit wierd as it consists of c and c++ objects
	define_test(struct_interop)
	add_library(si_c struct_interop.c)
	target_link_libraries(si_c arax_st)
	target_link_libraries(struct_interop_unit si_c)
endif()

