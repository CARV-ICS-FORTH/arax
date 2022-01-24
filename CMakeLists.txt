# Author: Foivos Zakkak
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

cmake_minimum_required(VERSION 3.13)

# project name
project(VINE)

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

set(VINE_CONFIG_FILE
    "~/.vinetalk"
    CACHE STRING "Vinetalk configuration file")
mark_as_advanced(FORCE VINE_CONFIG_FILE)
set(VINE_CONTROLER_PATH
    "${CMAKE_SOURCE_DIR}/../vine_controller/"
    CACHE STRING "VineController path")

set(CONF_VINE_MMAP_BASE
    0
    CACHE STRING "Non zero values set shared segment mmap address")
mark_as_advanced(FORCE CONF_VINE_MMAP_BASE)
exec_program(
  "getconf LEVEL1_DCACHE_LINESIZE" OUTPUT_VARIABLE
  CONF_CACHE_LINE RETURN_VALUE
  CONF_CACHE_LINE_RET)

set(CONF_CACHE_LINE
    64
    CACHE STRING "CPU Cacheline size")
mark_as_advanced(FORCE CONF_CACHE_LINE)

set(VINE_OBJECT_NAME_SIZE
    32
    CACHE STRING "Bytes reserved for VineObject names")

set(VINE_KV_CAP
    32
    CACHE STRING "Capacity of utils_kv_s instances")
mark_as_advanced(FORCE VINE_KV_CAP)

set(VINE_PROC_MAP_SIZE
    1024
    CACHE STRING "Number of processes that can use Vinetalk")
mark_as_advanced(FORCE VINE_PROC_MAP_SIZE)

option(MMAP_POPULATE "Populate mmap(good for many/larg tasks)" OFF)
mark_as_advanced(FORCE MMAP_POPULATE)

include(CMakeDependentOption)
include(Coverage)

option(ALLOC_STATS "Enable allocator statistics" OFF)
mark_as_advanced(FORCE MMAP_POPULATE)

option(VINE_REF_DEBUG "Enable reference inc/dec prints" OFF) # skip build_check
mark_as_advanced(FORCE VINE_REF_DEBUG)

option(VINE_THROTTLE_DEBUG "Enable Throttle inc/dec prints" OFF) # skip build_check
mark_as_advanced(FORCE VINE_THROTTLE_DEBUG)

option(VINE_THROTTLE_ENABLE "Enable Throttle" ON) # skip build_check
mark_as_advanced(FORCE VINE_THROTTLE_ENABLE)

option(VINE_DATA_ANNOTATE "Annotate vine_data for leak detection" OFF)
mark_as_advanced(FORCE VINE_DATA_ANNOTATE)

option(VINE_DATA_TRACK "Track where vine_data objects are allocated" OFF)
mark_as_advanced(FORCE VINE_DATA_TRACK)

exec_program(
  "git rev-parse HEAD" OUTPUT_VARIABLE
  VINE_TALK_GIT_REV RETURN_VALUE
  GIT_RET)
if(${GIT_RET} MATCHES 0)
  exec_program(
    "git branch --no-color --show-current" OUTPUT_VARIABLE
    VINE_TALK_GIT_BRANCH RETURN_VALUE
    GIT_RET)
  if(NOT ${GIT_RET} MATCHES 0)
    set(VINE_TALK_GIT_BRANCH "oldgit")
  endif()
else()
  set(VINE_TALK_GIT_REV "gitless")
  set(VINE_TALK_GIT_BRANCH "branchless")
endif()

set(UTILS_QUEUE_CAPACITY
    256U
    CACHE
      STRING
      "Maximum number tasks in a task queue (Up to 65536), MUST BE power of 2")
mark_as_advanced(FORCE UTILS_QUEUE_CAPACITY)

option(UTILS_QUEUE_MPMC "Add lock to allow multimple producers" ON)
mark_as_advanced(FORCE UTILS_QUEUE_MPMC)

# version
set(VINE_VERSION_MAJOR 0)
set(VINE_VERSION_MINOR 1)
set(VINE_VERSION_PATCH 0)
set(VINE_VERSION_EXTRA "")
set(VINE_VERSION
    "\"${VINE_VERSION_MAJOR}.${VINE_VERSION_MINOR}.${VINE_VERSION_PATCH}-${VINE_VERSION_EXTRA}\""
)

# if this is a debug build
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
  set(DEBUG 1)
endif()

# the default CFLAGS
set(CMAKE_C_FLAGS "-Wall -Werror -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3 ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE "-O2 -finline-functions -Werror ${CMAKE_C_FLAGS_RELEASE}")

add_subdirectory(src/async)
add_subdirectory(src/arch)
add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(src/alloc)
add_subdirectory(vinegrind)
add_subdirectory(vdf)
add_subdirectory(vine_plot)
add_subdirectory(noop)
add_subdirectory(vtop)

enable_testing() # This has to be in the source root.
add_subdirectory(tests)

# link all sublibs to this lib ($<TARGET_OBJECTS:name> is to link OBJECT
# libraries)
set(DEPS $<TARGET_OBJECTS:arch> $<TARGET_OBJECTS:async> $<TARGET_OBJECTS:core>
         $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:shm_allocator>)
add_library(vine SHARED ${DEPS})
target_include_directories(vine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vinetalk>
)

set_target_properties(vine PROPERTIES INTERFACE_LINK_LIBRARIES "-lrt ${CMAKE_THREAD_LIBS_INIT}")

add_library(vine_st STATIC ${DEPS})
target_include_directories(vine_st PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vinetalk>
)
set_target_properties(vine_st PROPERTIES INTERFACE_LINK_LIBRARIES "-lrt ${CMAKE_THREAD_LIBS_INIT}")

# Java Integration
find_package(Java QUIET)

if(JAVA_FOUND)
  option(JVineTalk "Build java Vinetalk wrappers" ON) # skip build_check
else()
  option(JVineTalk "Build java Vinetalk wrappers" OFF) # skip build_check
endif()
if(JVineTalk)
  include(UseJava)
  file(GLOB JAR_FILES . "${CMAKE_SOURCE_DIR}/JVTalk/*.jar")
  set(CMAKE_JAVA_JAR_ENTRY_POINT hello)
  foreach(JAR_PATH ${JAR_FILES})
    get_filename_component(JAR_FILE ${JAR_PATH} NAME)
    set(CP_JARS "${JAR_FILE} ${CP_JARS}")
    set(JAR_PATHS "${JAR_PATH} ${JAR_PATHS}")
  endforeach()
  set(CMAKE_JAVA_INCLUDE_PATH "${JAR_FILES}")
  file(GLOB java_srcs . "${CMAKE_SOURCE_DIR}/JVTalk/*.java")
  add_jar(JVTalk ${java_srcs} MANIFEST "${CMAKE_SOURCE_DIR}/JVTalk/Manifest")
  file(COPY ${JAR_FILES} DESTINATION ${CMAKE_BINARY_DIR})
  include_directories(${JAVA_HOME}/include)
  include_directories(${JAVA_HOME}/include/linux)
  set(CMAKE_C_FLAGS_DEBUG "-z noexecstack ${CMAKE_C_FLAGS_DEBUG}")
endif()

file(GLOB INC_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
file(COPY ${INC_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include)
file(COPY ${CMAKE_SOURCE_DIR}/src/alloc/alloc.h
     DESTINATION ${CMAKE_BINARY_DIR}/include/arch)
file(GLOB UTILS_HEADERS "${CMAKE_SOURCE_DIR}/src/utils/*.h")
file(COPY ${UTILS_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/utils)
file(GLOB CORE_HEADERS "${CMAKE_SOURCE_DIR}/src/core/*.h")
file(COPY ${CORE_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/core)
file(COPY ${CMAKE_SOURCE_DIR}/src/async/${async_architecture}/async.h
     DESTINATION ${CMAKE_BINARY_DIR}/include/)
file(COPY ${CMAKE_SOURCE_DIR}/src/async/async_api.h
     DESTINATION ${CMAKE_BINARY_DIR}/include/)

# Generate conf.h
configure_file(${CMAKE_SOURCE_DIR}/include/conf.h.cmake
               ${CMAKE_BINARY_DIR}/include/conf.h)

# Print build path
add_custom_target(
  VineTalkBuildPath ALL
  ${CMAKE_COMMAND} -E cmake_echo_color --cyan
  "VineTalk build path: ${CMAKE_BINARY_DIR}"
  COMMENT "VineTalkBuildPath")
add_dependencies(VineTalkBuildPath vine)

find_package(Precommit)

include(Documentation)
include(Install)
