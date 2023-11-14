include(CMakeDependentOption)

set(ARAX_CONFIG_FILE
    "~/.arax"
    CACHE STRING "Arax configuration file")
mark_as_advanced(FORCE ARAX_CONFIG_FILE)

set(CONF_ARAX_MMAP_BASE
    0
    CACHE STRING "Non zero values set shared segment mmap address")
mark_as_advanced(FORCE CONF_ARAX_MMAP_BASE)

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

set(UTILS_QUEUE_CAPACITY
    256U
    CACHE
      STRING
      "Maximum number tasks in a task queue (Up to 65536), MUST BE power of 2")
mark_as_advanced(FORCE UTILS_QUEUE_CAPACITY)

option(UTILS_QUEUE_MPMC "Add lock to allow multimple producers" ON)
mark_as_advanced(FORCE UTILS_QUEUE_MPMC)

# if this is a debug build
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
  set(DEBUG 1)
endif()

# Java Integration
find_package(Java QUIET)

if(JAVA_FOUND)
  option(JAVA_WRAPS "Build java Arax wrappers" ON) # skip build_check
else()
  option(JAVA_WRAPS "Build java Arax wrappers" OFF) # skip build_check
endif()

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
