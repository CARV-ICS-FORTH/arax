include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(Catch2)

option(BUILD_TESTS "Build unit tests" ON) # skip build_check

if(BUILD_TESTS)
	find_package(Threads)

	add_library(testing testing.cpp)
	target_link_libraries(testing vine_st Catch2::Catch2)

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
	define_test(vine_object)
	define_test(vine_data)
	define_test(vine_accel)
	define_test(vine_talk)
	define_test(vine_mmap)
	define_test(vine_plot)
	define_test(alloc_perf)
	define_test(async)
	define_test(mmap)
	define_test(vine_throttle)
	define_test(accel_throttle)
	define_test(pipe_throttle)
	define_test(system)

   	# struct_interop is a bit wierd as it consists of c and c++ objects
	define_test(struct_interop)
	add_library(si_c struct_interop.c)
	target_link_libraries(si_c vine_st)
	target_link_libraries(struct_interop_unit si_c)
endif()

