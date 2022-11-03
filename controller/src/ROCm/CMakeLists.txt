set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" ${CMAKE_MODULE_PATH})

# ROCm v4.0.0 requires both
find_package(HIP QUIET)
find_package(hip QUIET)

enable_backend(HIP_FOUND AND hip_FOUND)

if (ENABLE_ROCm)


	set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
	set( CMAKE_CXX_FLAGS "-w" )

	# For ROCm >=3.5, wipe hip-clang specific interface options which are propagated
	set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC")
	set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")

	# Thread
	set(hip_sources HIPaccelThread.cpp)

	hip_add_library(prepareHIPGPU prepareHIPGPU.cpp)

	vc_link_lib(prepareHIPGPU)

	# Main hip library
	hip_add_library(hip_lib OBJECT ${hip_sources})

	target_link_libraries(hip_lib prepareHIPGPU hip::device)
	# Builtins
	hip_add_library(HIPBuiltins SHARED HIPBuiltins.cpp)

	vc_link_target(hip_lib)

	register_builtin(HIPBuiltins)
endif(ENABLE_ROCm)
