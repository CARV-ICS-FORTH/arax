enable_backend(ON)

if(ENABLE_Cpu)
	add_library(cpu OBJECT CPUaccelThread.cpp)

	add_library(CpuBuiltins SHARED CpuBuiltins.cpp)

	target_link_libraries(arax_controller PRIVATE cpu)

	register_builtin(CpuBuiltins)
endif(ENABLE_Cpu)
