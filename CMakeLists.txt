#                            Author: Foivos Zakkak

# we need at least 2.8.8 because we need the OBJECT library support
cmake_minimum_required(VERSION 2.8.8)
# project name
project(VINE)

# disable builds in source tree
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# export compile commands in compile_commands.json, for use with IDEs
# and editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# include modules from cmake dir
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(TRACE_ENABLE "Enable tracing of Vine calls" OFF)

if(TRACE_ENABLE)
  set(CMAKE_C_FLAGS "-DTRACE_ENABLE ${CMAKE_C_FLAGS}")
  message(STATUS "Tracer is enabled")
else()
  message(STATUS "Tracer is disabled")
endif()

option(BREAKS_ENABLE "Enable breakdowns of vine_talk procedures." OFF)

if(BREAKS_ENABLE)
  set(CMAKE_C_FLAGS "-DBREAKS_ENABLE ${CMAKE_C_FLAGS}")
  message(STATUS "Breakdowns enabled")
else()
  message(STATUS "Breakdowns disabled")
endif()

option(ALLOC_STATS "Enable allocator statistics" ON)

if(ALLOC_STATS)
  set(CMAKE_C_FLAGS "-DALLOC_STATS ${CMAKE_C_FLAGS}")
  message(STATUS "Alloc stats enabled")
else()
  message(STATUS "Alloc stats disabled")
endif()

#options
find_package(Check)
if(CHECK_FOUND)
  option(BUILD_TESTS "Build unit tests" ON)
else()
  option(BUILD_TESTS "Build unit tests" OFF)
endif()

option(COVERAGE "Enable coverage reports" OFF)

if(COVERAGE)
	set(CMAKE_C_FLAGS "-fprofile-arcs -ftest-coverage ${CMAKE_C_FLAGS}")
	SET(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
	set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 1)
	set(BUILD_TESTS 1)
	message(STATUS "Coverage is enabled")
else()
	message(STATUS "Coverage is disabled")
endif()

if(BUILD_TESTS)
  message(STATUS "Tests are enabled")
else()
  message(STATUS "Tests are disabled")
endif()

# version
set(VINE_VERSION_MAJOR 0)
set(VINE_VERSION_MINOR 1)
set(VINE_VERSION_PATCH 0)
set(VINE_VERSION_EXTRA "")
set(VINE_VERSION
  "\"${VINE_VERSION_MAJOR}.${VINE_VERSION_MINOR}.${VINE_VERSION_PATCH}-${VINE_VERSION_EXTRA}\"")

# if this is a debug build
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
  set(DEBUG 1)
  enable_testing()
  message(STATUS "Debug is enabled")
else()
  message(STATUS "Debug is disabled")
endif()

# print version when building
message(STATUS "version: ${VINE_VERSION}")

# configure_file(
#   ${VINE_SOURCE_DIR}/include/config.h.cmake
#   ${VINE_SOURCE_DIR}/include/config.h
# )

# the default CFLAGS
set(CMAKE_C_FLAGS "-Wall -Werror -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3")
set(CMAKE_C_FLAGS_RELEASE "-O2 -finline-functions -Werror")

# include the basics
include_directories(${VINE_SOURCE_DIR}/include ${VINE_SOURCE_DIR}/src/async)

add_subdirectory(src/async)

#include async specifics
message(STATUS ${VINE_SOURCE_DIR}/src/async/${async_architecture})
include_directories(${VINE_SOURCE_DIR}/include ${VINE_SOURCE_DIR}/src/async ${VINE_SOURCE_DIR}/src/async/${async_architecture})
add_subdirectory(src/arch)
add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(3rdparty/dlmalloc)
add_subdirectory(examples)

# if this is test build
if(BUILD_TESTS)
  enable_testing()
  file(GLOB test_srcs . "tests/*.c")
  file(GLOB test_srcs_excl . "tests/CMakeLists.txt")
  list(REMOVE_ITEM test_srcs ${test_srcs_excl})
  foreach(TEST ${test_srcs})
    get_filename_component(TEST_BIN ${TEST} NAME_WE)
    add_test(${TEST_BIN} tests/${TEST_BIN}_unit)
  endforeach()
  add_subdirectory(tests)
endif()

# link all sublibs to this lib ($<TARGET_OBJECTS:name> is to link OBJECT
# libraries)
set(DEPS $<TARGET_OBJECTS:arch> $<TARGET_OBJECTS:async> $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:dlmalloc>)
add_library(vine SHARED ${DEPS})

file(COPY ${VINE_SOURCE_DIR}/include DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${VINE_SOURCE_DIR}/src/arch/alloc.h DESTINATION ${CMAKE_BINARY_DIR}/include/arch)
file(GLOB UTILS_HEADERS "${VINE_SOURCE_DIR}/src/utils/*.h")
file(COPY ${UTILS_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/utils)
file(GLOB CORE_HEADERS "${VINE_SOURCE_DIR}/src/core/*.h")
file(COPY ${CORE_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/core)
file(COPY ${VINE_SOURCE_DIR}/src/async/${async_architecture}/async.h DESTINATION ${CMAKE_BINARY_DIR}/include/)
file(COPY ${VINE_SOURCE_DIR}/src/async/async_api.h DESTINATION ${CMAKE_BINARY_DIR}/include/)

# Generate conf.h
string(REGEX MATCHALL "-D.*" def_list ${CMAKE_C_FLAGS})
if(def_list)
  string(REPLACE "-D" "\n#define " def_list ${def_list})
  file(WRITE ${CMAKE_BINARY_DIR}/include/conf.h "#ifndef VINE_TALK_BUILD_CONFIG_HEADER\n#define VINE_TALK_BUILD_CONFIG_HEADER")
  file(APPEND ${CMAKE_BINARY_DIR}/include/conf.h ${def_list})
  file(APPEND ${CMAKE_BINARY_DIR}/include/conf.h "\n#endif\n")
endif()

# #Add the install target
# install(FILES ${VINE_SOURCE_DIR}/include/vine_talk.h ${VINE_SOURCE_DIR}/include/vine_pipe.h DESTINATION include)

# #ARCHIVE installs the static libraries
install(TARGETS vine LIBRARY DESTINATION /usr/lib)
