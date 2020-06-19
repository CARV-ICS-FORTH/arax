# TODO: Add option to choose architecture
include_directories(async PRIVATE ${VINE_SOURCE_DIR}/src
                    ${VINE_SOURCE_DIR}/src/core ${VINE_SOURCE_DIR}/include)

exec_program("dmesg | grep kvm-clock |wc -l" OUTPUT_VARIABLE INVM)

if(${INVM} MATCHES "0")
  set(async_architecture
      mutex
      CACHE STRING "Mechanism used for async API")
else()
  message(STATUS "KVM detected")
  set(async_architecture
      ivshmem
      CACHE STRING "Mechanism used for async API")
endif()

set_property(CACHE async_architecture PROPERTY STRINGS spin mutex ivshmem)

aux_source_directory(${async_architecture} async_sources)
add_library(async OBJECT ${async_sources})
