find_package(OpenCL QUIET)

enable_backend(OpenCL_FOUND)

if(ENABLE_OpenCL)
	# If library is found but not header, check if CUDA has the headers
	if( NOT ${OpenCL_INCLUDE_DIR} )
		find_package(CUDA QUIET)
		if( CUDA_FOUND )
			FIND_PATH(OpenCL_INCLUDE_DIR CL/cl.h ${CUDA_TOOLKIT_INCLUDE} )
			find_library(OpenCL_LIBRARY libOpenCL.so)
			message("OpenCL lib: ${OpenCL_LIBRARY}")
		endif()
	endif()

	message("Ocl Include ${OpenCL_INCLUDE_DIR}")

	# Main OpenCL library
	add_library(ocl OBJECT OpenclaccelThread.cpp)
	target_include_directories(ocl SYSTEM PRIVATE ${OpenCL_INCLUDE_DIR})
	target_compile_definitions(ocl PRIVATE CL_TARGET_OPENCL_VERSION=120)

	# Builtins
	add_library(OpenclBuiltins SHARED OpenclBuiltins.cpp)
	target_include_directories(OpenclBuiltins SYSTEM PRIVATE ${OpenCL_INCLUDE_DIR})
	target_compile_definitions(OpenclBuiltins PRIVATE CL_TARGET_OPENCL_VERSION=120)

	target_link_libraries(arax_controller PRIVATE ocl)
	target_link_libraries(arax_controller PUBLIC ${OpenCL_LIBRARY})

	register_builtin(OpenclBuiltins)
endif(ENABLE_OpenCL)
