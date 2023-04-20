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

set(ARAX_CONFIG_FILE
    "~/.arax"
    CACHE STRING "Arax configuration file")
mark_as_advanced(FORCE ARAX_CONFIG_FILE)

set(CONF_ARAX_MMAP_BASE
    0
    CACHE STRING "Non zero values set shared segment mmap address")
mark_as_advanced(FORCE CONF_ARAX_MMAP_BASE)

include(cache_detect)

set(ARAX_KV_CAP
    32
    CACHE STRING "Capacity of utils_kv_s instances")
mark_as_advanced(FORCE ARAX_KV_CAP)

set(ARAX_PROC_MAP_SIZE
    1024
    CACHE STRING "Number of processes that can use Arax")
mark_as_advanced(FORCE ARAX_PROC_MAP_SIZE)

option(MMAP_POPULATE "Populate mmap(good for many/larg tasks)" OFF)
mark_as_advanced(FORCE MMAP_POPULATE)

include(CMakeDependentOption)
include(Coverage)

option(ALLOC_STATS "Enable allocator statistics" OFF)
mark_as_advanced(FORCE MMAP_POPULATE)

option(ARAX_REF_DEBUG "Enable reference inc/dec prints" OFF) # skip build_check
mark_as_advanced(FORCE ARAX_REF_DEBUG)

option(ARAX_THROTTLE_DEBUG "Enable Throttle inc/dec prints" OFF) # skip build_check
mark_as_advanced(FORCE ARAX_THROTTLE_DEBUG)

option(ARAX_THROTTLE_ENFORCE "Enforce Throttle Limits" OFF) # skip build_check
mark_as_advanced(FORCE ARAX_THROTTLE_ENFORCE)

option(ARAX_DATA_ANNOTATE "Annotate arax_data for leak detection" OFF)
mark_as_advanced(FORCE ARAX_DATA_ANNOTATE)

option(ARAX_DATA_TRACK "Track where arax_data objects are allocated" OFF)
mark_as_advanced(FORCE ARAX_DATA_TRACK)

exec_program(
  "git rev-parse HEAD" OUTPUT_VARIABLE
  ARAX_GIT_REV RETURN_VALUE
  GIT_RET)
if(${GIT_RET} MATCHES 0)
  exec_program(
    "git branch --no-color --show-current" OUTPUT_VARIABLE
    ARAX_GIT_BRANCH RETURN_VALUE
    GIT_RET)
  if(NOT ${GIT_RET} MATCHES 0)
    set(ARAX_GIT_BRANCH "oldgit")
  endif()
else()
  set(ARAX_GIT_REV "gitless")
  set(ARAX_GIT_BRANCH "branchless")
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
set(ARAX_VERSION_MAJOR 0)
set(ARAX_VERSION_MINOR 1)
set(ARAX_VERSION_PATCH 0)
set(ARAX_VERSION_EXTRA "")
set(ARAX_VERSION
    "\"${ARAX_VERSION_MAJOR}.${ARAX_VERSION_MINOR}.${ARAX_VERSION_PATCH}-${ARAX_VERSION_EXTRA}\""
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
add_subdirectory(araxgrind)
add_subdirectory(vdf)
add_subdirectory(arax_plot)
add_subdirectory(noop)
add_subdirectory(vtop)
add_subdirectory(controller)
enable_testing() # This has to be in the source root.
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

# Java Integration
find_package(Java QUIET)

if(JAVA_FOUND)
  option(JAVA_WRAPS "Build java Arax wrappers" ON) # skip build_check
else()
  option(JAVA_WRAPS "Build java Arax wrappers" OFF) # skip build_check
endif()
if(JAVA_WRAPS)
  include(UseJava)
  file(GLOB JAR_FILES . "${CMAKE_SOURCE_DIR}/Java/*.jar")
  set(CMAKE_JAVA_JAR_ENTRY_POINT hello)
  foreach(JAR_PATH ${JAR_FILES})
    get_filename_component(JAR_FILE ${JAR_PATH} NAME)
    set(CP_JARS "${JAR_FILE} ${CP_JARS}")
    set(JAR_PATHS "${JAR_PATH} ${JAR_PATHS}")
  endforeach()
  set(CMAKE_JAVA_INCLUDE_PATH "${JAR_FILES}")
  file(GLOB java_srcs . "${CMAKE_SOURCE_DIR}/Java/*.java")
  add_jar(JArax ${java_srcs} MANIFEST "${CMAKE_SOURCE_DIR}/Java/Manifest")
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
  AraxBuildPath ALL
  ${CMAKE_COMMAND} -E cmake_echo_color --cyan
  "Arax build path: ${CMAKE_BINARY_DIR}"
  COMMENT "AraxBuildPath")
add_dependencies(AraxBuildPath arax)

find_package(Precommit)

include(Documentation)
include(Package)
include(Install)
include(hide_advanced_options)
