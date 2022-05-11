set(dlmalloc_sources malloc.c malloc.h alloc.c)
add_library(shm_allocator OBJECT ${dlmalloc_sources})
target_include_directories(shm_allocator PRIVATE
    ${CMAKE_BINARY_DIR}/include)
set_property(
  SOURCE ${dlmalloc_sources}
  APPEND_STRING
  PROPERTY
    COMPILE_FLAGS
    " -DONLY_MSPACES -DUSE_SPIN_LOCKS -DMSPACES -DHAVE_USR_INCLUDE_MALLOC_H -DMALLOC_INSPECT_ALL -DHAVE_MMAP=1 -DHAVE_MORECORE=0 -Wno-expansion-to-defined"
)
