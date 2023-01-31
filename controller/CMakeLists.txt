cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0079 NEW)

set (CMAKE_CXX_STANDARD 14)

set(BUILTINS_PATH ${CMAKE_BINARY_DIR}/cntrl CACHE STRING "Set folder containing builtins")
mark_as_advanced(BUILTINS_PATH)

include(vc_utils)

include_directories(include)
include_directories(src)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(src/Core)
add_subdirectory(src/Schedulers)

add_executable(arax_controller $<TARGET_OBJECTS:cntrl_core> $<TARGET_OBJECTS:schedulers>)
target_link_libraries(arax_controller PUBLIC arax_st dl pthread rt)
set_target_properties(arax_controller PROPERTIES ENABLE_EXPORTS on)
target_include_directories(arax_controller PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/araxcontroller)

# Architecure backends
add_backend(Cpu)
add_backend(OpenCL)
add_backend(Sda)
add_backend(Cuda)
add_backend(ROCm)

if (ENABLE_ROCm)
	set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
	set( CMAKE_CXX_FLAGS "-w" )
endif()

if(NOT EXISTS ${CMAKE_BINARY_DIR}/conf.json)
file(COPY ${CMAKE_SOURCE_DIR}/controller/conf.json DESTINATION ${CMAKE_BINARY_DIR})
endif()

add_custom_target(run
	COMMAND cd .. && ${CMAKE_BINARY_DIR}/controller/arax_controller ${CMAKE_BINARY_DIR}/conf.json && cd -
	DEPENDS arax_controller
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(gdb
	COMMAND cd .. && gdb --args ${CMAKE_BINARY_DIR}/controller/arax_controller ${CMAKE_BINARY_DIR}/conf.json && cd -
	DEPENDS arax_controller
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

install(TARGETS arax_controller EXPORT arax DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/controller/include/picojson.h DESTINATION include/arax)
