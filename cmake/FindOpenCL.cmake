find_path(OpenCL_INCLUDE_DIR
	NAMES
	CL/cl.h OpenCL/cl.h hld/host/include/CL/cl.h
	PATHS
	ENV "PROGRAMFILES(X86)"
	ENV AMDAPPSDKROOT
	ENV INTELOCLSDKROOT
	ENV NVSDKCOMPUTE_ROOT
	ENV CUDA_PATH
	ENV ATISTREAMSDKROOT
	ENV OCL_ROOT
	ENV ALTERA_ROOT_DIR
	PATH_SUFFIXES
	include
	OpenCL/common/inc
	"AMD APP/include"
	
)

find_library(OpenCL_LIBRARY
	NAMES OpenCL
	PATHS
	ENV AMDAPPSDKROOT
	ENV ALTERA_ROOT_DIR
	ENV CUDA_PATH
	PATH_SUFFIXES 
	hld/linux64/lib/
	lib/x86_64
	lib/x64
	lib
	lib64
)

set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})

if(EXISTS ${OpenCL_LIBRARY} AND EXISTS ${OpenCL_INCLUDE_DIR})
	set(OpenCL_FOUND ${OpenCL_LIBRARY})
endif()

