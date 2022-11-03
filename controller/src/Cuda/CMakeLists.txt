find_package(CUDA QUIET)

enable_backend(CUDA_FOUND)

if(ENABLE_Cuda)
    option(FREE_THREAD "Enable an external thread that performs ref_dec." OFF)

    target_sources(arax_controller PRIVATE CUDAaccelThread.cpp)

	cuda_add_library(cuda_vc CublasService.cpp prepareGPU.cu)
	target_include_directories(cuda_vc PUBLIC ${CUDA_INCLUDE_DIRS})
	target_include_directories(cuda_vc PUBLIC ${CUDA_TOOLKIT_INCLUDE})

	cuda_add_cublas_to_target(cuda_vc)
	cuda_add_cufft_to_target(cuda_vc)

	find_package(CUDNN)
	if(CUDNN_FOUND)
		target_sources(cuda_vc PUBLIC CudnnService.cpp)
		target_link_libraries(arax_controller PUBLIC ${CUDNN_LIBRARY})
	endif(CUDNN_FOUND)

	target_sources(cuda_vc PUBLIC CurandService.cpp)
	target_link_libraries(arax_controller PUBLIC ${CUDA_curand_LIBRARY})

	# Builtins
	cuda_add_library(CUDABuiltins SHARED CUDABuiltins.cu OPTIONS -std=c++11)

	target_link_libraries(arax_controller PRIVATE cuda_vc)
	target_link_libraries(arax_controller PUBLIC ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

	register_builtin(CUDABuiltins)
endif(ENABLE_Cuda)
