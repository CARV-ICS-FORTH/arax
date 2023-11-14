# Author: Foivos Zakkak
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

cmake_minimum_required(VERSION 3.13)

# project name
project(arax)

set (CMAKE_CXX_STANDARD 11)

# disable builds in source tree
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Install prefix" FORCE)
endif()

# export compile commands in compile_commands.json, for use with IDEs and
# editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# include modules from cmake dir
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cache_detect)
include(Options)
include(CTest)
include(Coverage)

# the default CFLAGS
set(CMAKE_C_FLAGS "-Wall -Werror -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3 ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE "-O2 -finline-functions -Werror ${CMAKE_C_FLAGS_RELEASE}")

add_subdirectory(src/async)
add_subdirectory(src/arch)
add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(src/alloc)
add_subdirectory(araxgrind)
add_subdirectory(vdf)
add_subdirectory(arax_plot)
add_subdirectory(noop)
add_subdirectory(vtop)
add_subdirectory(controller)
add_subdirectory(tests)

# link all sublibs to this lib ($<TARGET_OBJECTS:name> is to link OBJECT
# libraries)
set(DEPS $<TARGET_OBJECTS:arch> $<TARGET_OBJECTS:async> $<TARGET_OBJECTS:core>
         $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:shm_allocator>)
add_library(arax SHARED ${DEPS})
target_include_directories(arax PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/arax>
)

set_target_properties(arax  PROPERTIES INTERFACE_LINK_LIBRARIES "-lrt ${CMAKE_THREAD_LIBS_INIT}")

add_library(arax_st STATIC ${DEPS})
target_include_directories(arax_st PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/arax>
)
set_target_properties(arax_st PROPERTIES INTERFACE_LINK_LIBRARIES "-lrt ${CMAKE_THREAD_LIBS_INIT}")

# Print build path
add_custom_target(
  AraxBuildPath ALL
  ${CMAKE_COMMAND} -E cmake_echo_color --cyan
  "Arax build path: ${CMAKE_BINARY_DIR}"
  COMMENT "AraxBuildPath")
add_dependencies(AraxBuildPath arax)

find_package(Precommit)

include(CopyHeaders)
include(Documentation)
include(Package)
include(Install)
include(hide_advanced_options)
