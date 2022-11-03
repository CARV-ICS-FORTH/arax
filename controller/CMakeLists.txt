cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0079 NEW)

project(ARAXCONTROLLER)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Install prefix" FORCE)
endif()

include_directories(include)

# include modules from cmake dir
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Try the sytem method
find_package(arax)

if (NOT arax_DIR)
	get_filename_component(ARAX_DEF_BUILD_PATH "../arax/build" ABSOLUTE)
	set(ARAX_BUILD_PATH ${ARAX_DEF_BUILD_PATH} CACHE STRING "Arax build location")
	include("${ARAX_BUILD_PATH}/arax.cmake")
endif()

get_target_property(ARAX_INCLUDE_DIRECTORIES arax INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${ARAX_INCLUDE_DIRECTORIES})

set (CMAKE_CXX_STANDARD 11)

set(BUILTINS_PATH ${CMAKE_BINARY_DIR}/src CACHE STRING "Set folder containing builtins")
mark_as_advanced(BUILTINS_PATH)

include(vc_utils)

add_subdirectory(src/Core)
add_subdirectory(src/Schedulers)

add_executable(arax_controller $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:schedulers>)
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

add_custom_target(run
	COMMAND cd .. && ./build/arax_controller conf.json && cd -
	DEPENDS arax_controller
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(gdb
	COMMAND cd .. && gdb --args ./build/arax_controller conf.json && cd -
	DEPENDS arax_controller
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

include(Install)
