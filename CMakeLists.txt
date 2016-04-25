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

#options
find_package(Check)
if(CHECK_FOUND)
  option(BUILD_TESTS "Build unit tests" ON)
else()
  option(BUILD_TESTS "Build unit tests" OFF)
endif()

# version
set(VINE_VERSION_MAJOR 0)
set(VINE_VERSION_MINOR 1)
set(VINE_VERSION_PATCH 0)
set(VINE_VERSION_EXTRA "")
set(VINE_VERSION
  "\"${VINE_VERSION_MAJOR}.${VINE_VERSION_MINOR}.${VINE_VERSION_PATCH}-${VINE_VERSION_EXTRA}\"")
# print version when building
message(STATUS "version: ${VINE_VERSION}")

# if this is a debug build
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
  set(DEBUG 1)
  enable_testing()
endif()

# if this is test build
if(BUILD_TESTS)
  enable_testing()
  file(GLOB test_bins . "tests/*.c")
  file(GLOB test_bins_excl . "tests/CMakeLists.txt")
  list(REMOVE_ITEM test_bins ${test_bins_excl})
  foreach(TEST ${test_bins})
    get_filename_component(TEST_BIN ${TEST} NAME_WE)
    add_test(${TEST_BIN} tests/${TEST_BIN})
  endforeach()
  add_subdirectory(tests)
endif()

# configure_file(
#   ${VINE_SOURCE_DIR}/include/config.h.cmake
#   ${VINE_SOURCE_DIR}/include/config.h
# )

# the default CFLAGS
set(CMAKE_C_FLAGS
  "-O3 -Wall -finline-functions -fPIC"
)
set(CMAKE_C_FLAGS_DEBUG "-ggdb3")
set(CMAKE_C_FLAGS_RELEASE "-Werror")

# include the include directory
include_directories(${VINE_SOURCE_DIR}/include)

add_subdirectory(src/arch)
add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(3rdparty/dlmalloc)
add_subdirectory(examples)

# link all sublibs to this lib ($<TARGET_OBJECTS:name> is to link OBJECT
# libraries)
set(DEPS $<TARGET_OBJECTS:arch> $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:dlmalloc>)
add_library(vine ${DEPS})

# if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
#   add_subdirectory(tests)
# endif()

# #Add the install target
# install(FILES ${VINE_SOURCE_DIR}/include/vine_talk.h ${VINE_SOURCE_DIR}/include/vine_pipe.h DESTINATION include)

# #ARCHIVE installs the static libraries
# install(TARGETS libvine ARCHIVE DESTINATION lib)
