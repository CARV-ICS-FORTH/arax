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

set(VINE_CONFIG_FILE "~/.vinetalk" CACHE STRING "Vinetalk configuration file")

set(CONF_VINE_MMAP_BASE 0 CACHE STRING "Non zero values set shared segment mmap address")
exec_program("getconf LEVEL1_DCACHE_LINESIZE" OUTPUT_VARIABLE CONF_CACHE_LINE RETURN_VALUE CONF_CACHE_LINE_RET)
if(${CONF_CACHE_LINE_RET} MATCHES 0)
	message(STATUS "Cacheline autodetected: ${CONF_CACHE_LINE}")
else()
	set(CONF_CACHE_LINE 64)
	message(STATUS "Cacheline set to: ${CONF_CACHE_LINE}")
endif()

set(CONF_CACHE_LINE 64 CACHE STRING "CPU Cacheline size")

set(VINE_OBJECT_NAME_SIZE 32 CACHE STRING "Bytes reserved for VineObject names")

set(VINE_KV_CAP 32 CACHE STRING "Capacity of utils_kv_s instances")

option(MMAP_POPULATE "Populate mmap(good for many/larg tasks)" OFF)

if(MMAP_POPULATE)
  message(STATUS "MMAP_POPULATE is enabled")
else()
  message(STATUS "MMAP_POPULATE is disabled")
endif()

option(TRACE_ENABLE "Enable tracing of Vine calls" OFF) # skip build_check

if(TRACE_ENABLE)
  message(STATUS "Tracer is enabled")
else()
  message(STATUS "Tracer is disabled")
endif()

option(BREAKS_ENABLE "Enable breakdowns of vine_talk procedures." OFF)

if(BREAKS_ENABLE)
  message(STATUS "Breakdowns enabled")
else()
  message(STATUS "Breakdowns disabled")
endif()

INCLUDE(CMakeDependentOption)

CMAKE_DEPENDENT_OPTION(VINE_TELEMETRY "Enable task job telemetry" OFF "BREAKS_ENABLE" OFF)

if(VINE_TELEMETRY)
message(STATUS "Telemetry is enabled")
else()
message(STATUS "Telemetry is disabled")
endif()

option(ALLOC_STATS "Enable allocator statistics" ON)

set(ALLOC_PART_MB 2048 CACHE STRING "Allocator partition size(MB)")

if(ALLOC_STATS)
  message(STATUS "Alloc stats enabled")
else()
  message(STATUS "Alloc stats disabled")
endif()

option(VINE_REF_DEBUG "Enable reference inc/dec prints" OFF)

if(VINE_REF_DEBUG)
  message(STATUS "Referece debug prints enabled")
else()
  message(STATUS "Referece debug prints disabled")
endif()

option(VINE_DATA_ANNOTATE "Annotate vine_data for leak detection" OFF)

if(VINE_DATA_ANNOTATE)
  message(STATUS "VineData annotation enabled")
else()
  message(STATUS "VineData annotation disabled")
endif()

exec_program("git rev-parse HEAD" OUTPUT_VARIABLE VINE_TALK_GIT_REV RETURN_VALUE GIT_RET)
if(${GIT_RET} MATCHES 0)
	exec_program("git branch --no-color --show-current" OUTPUT_VARIABLE VINE_TALK_GIT_BRANCH RETURN_VALUE GIT_RET)
	if(NOT ${GIT_RET} MATCHES 0)
		set(VINE_TALK_GIT_BRANCH "oldgit")
	endif()
else()
	set(VINE_TALK_GIT_REV "gitless")
	set(VINE_TALK_GIT_BRANCH "branchless")
endif()
message(STATUS "Git revision is ${VINE_TALK_GIT_REV} - ${VINE_TALK_GIT_BRANCH}")

set(UTILS_QUEUE_CAPACITY 256U
  CACHE STRING "The maximum number of outstanding tasks per task queue (Up to 65536), MUST BE power of 2")

#options

find_package(Doxygen)

if(DOXYGEN_FOUND)
	option(BUILD_DOCS "Build documentation" ON) # skip build_check
else()
	option(BUILD_DOCS "Build documentation" OFF) # skip build_check
endif()

if(BUILD_DOCS)
	message(STATUS "Documentation generation enabled")
	if(DOXYGEN_DOT_FOUND)
		set(DOXYGEN_HAVE_DOT "YES")
	else()
		set(DOXYGEN_HAVE_DOT "NO")
	endif()
	configure_file(
		${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile
		${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	)
	add_custom_target(docs_pre
		COMMAND doxygen
	)
	add_custom_target(doc
		DEPENDS docs_pre
		COMMENT "Doc files at: ${CMAKE_CURRENT_BINARY_DIR}/docs/html/index.html"
	)
	add_custom_target(docs DEPENDS doc)
else()
	message(STATUS "Documentation generation disabled")
endif()

find_package(Check)
if(CHECK_FOUND)
  option(BUILD_TESTS "Build unit tests" ON)
else()
  option(BUILD_TESTS "Build unit tests" OFF)
endif()

find_package(Poco)

option(COVERAGE "Enable coverage reports" OFF)  # skip build_check

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

if(COVERAGE)
	add_custom_target(coverage_pre
		COMMAND make -j
		COMMAND -make -j test
		COMMAND -rm -Rf coverage
		COMMAND mkdir coverage
		COMMAND /usr/bin/gcovr -p --exclude-directories '3rdparty' --exclude-directories 'tests' -r ${CMAKE_CURRENT_SOURCE_DIR}/ --html-title 'VineTalk Coverage Report' --html --html-details -o coverage/coverage.html -s
		COMMAND sed -i.bak 's/GCC Code/VineTalk/g' coverage/*.html
	)
	add_custom_target(coverage
		DEPENDS coverage_pre
		COMMENT "Coverage results at: ${CMAKE_CURRENT_BINARY_DIR}/coverage/coverage.html"
	)

	add_custom_target(cov DEPENDS coverage)
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
set(CMAKE_C_FLAGS "-Wl,-rpath,/usr/lib ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3")
set(CMAKE_C_FLAGS_RELEASE "-O2 -finline-functions -Werror")

# include the basics
include_directories(${CMAKE_BINARY_DIR}/include ${CMAKE_BINARY_DIR}/include/async)

add_subdirectory(src/async)

#include async specifics
message(STATUS ${VINE_SOURCE_DIR}/src/async/${async_architecture})
include_directories(${CMAKE_BINARY_DIR}/include ${CMAKE_BINARY_DIR}/include/async ${VINE_SOURCE_DIR}/src/async/${async_architecture})
add_subdirectory(src/arch)
add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(3rdparty/dlmalloc)
add_subdirectory(vinegrind)
add_subdirectory(vdf)

# if this is test build
if(BUILD_TESTS)
  enable_testing()
  file(GLOB test_srcs . "tests/*.c")
  # FIXME: Make async_unit work with ivshmem as well?
  if(async_architecture STREQUAL "ivshmem")
    file(GLOB test_srcs_excl . "tests/async.c")
    list(REMOVE_ITEM test_srcs ${test_srcs_excl})
  endif()
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
add_library(vine_st STATIC ${DEPS})


# Java Integration
find_package(Java)

if(JAVA_FOUND)
  option(JVineTalk "Build java Vinetalk wrappers" ON)
else()
  option(JVineTalk "Build java Vinetalk wrappers" OFF)
endif()
if(JVineTalk)
  include(UseJava)
  file(GLOB JAR_FILES . "${VINE_SOURCE_DIR}/JVTalk/*.jar")
  set(CMAKE_JAVA_JAR_ENTRY_POINT hello)
  foreach(JAR_PATH ${JAR_FILES})
    get_filename_component(JAR_FILE ${JAR_PATH} NAME)
    set(CP_JARS "${JAR_FILE} ${CP_JARS}")
    set(JAR_PATHS "${JAR_PATH} ${JAR_PATHS}")
  endforeach()
  set(CMAKE_JAVA_INCLUDE_PATH "${JAR_FILES}")
  file(GLOB java_srcs . "${VINE_SOURCE_DIR}/JVTalk/*.java")
  add_jar(JVTalk ${java_srcs} MANIFEST "${VINE_SOURCE_DIR}/JVTalk/Manifest" )
  file(COPY ${JAR_FILES} DESTINATION ${CMAKE_BINARY_DIR})
  include_directories(${JAVA_HOME}/include)
  include_directories(${JAVA_HOME}/include/linux)
  set(CMAKE_C_FLAGS_DEBUG "-z noexecstack")
  target_link_libraries(vine -lrt)
endif()

file(GLOB INC_HEADERS "${VINE_SOURCE_DIR}/include/*.h")
file(COPY ${INC_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include)
file(COPY ${VINE_SOURCE_DIR}/src/arch/alloc.h DESTINATION ${CMAKE_BINARY_DIR}/include/arch)
file(GLOB UTILS_HEADERS "${VINE_SOURCE_DIR}/src/utils/*.h")
file(COPY ${UTILS_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/utils)
file(GLOB CORE_HEADERS "${VINE_SOURCE_DIR}/src/core/*.h")
file(COPY ${CORE_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/core)
file(COPY ${VINE_SOURCE_DIR}/src/async/${async_architecture}/async.h DESTINATION ${CMAKE_BINARY_DIR}/include/)
file(COPY ${VINE_SOURCE_DIR}/src/async/async_api.h DESTINATION ${CMAKE_BINARY_DIR}/include/)

# Generate conf.h
configure_file(${CMAKE_SOURCE_DIR}/include/conf.h.cmake ${CMAKE_BINARY_DIR}/include/conf.h)

#Print build path
add_custom_target( VineTalkBuildPath ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan "VineTalk build path: ${CMAKE_BINARY_DIR}"
    COMMENT "VineTalkBuildPath" )
add_dependencies( VineTalkBuildPath vine )

# #Add the install target
# install(FILES ${VINE_SOURCE_DIR}/include/vine_talk.h ${VINE_SOURCE_DIR}/include/vine_pipe.h DESTINATION include)

# #ARCHIVE installs the static libraries
install(TARGETS vine LIBRARY DESTINATION /usr/lib)
