include(CheckCCompilerFlag)

set(dlmalloc_sources malloc.c malloc.h alloc.c)

add_library(shm_allocator OBJECT ${dlmalloc_sources})

check_c_compiler_flag(-Wno-unused-but-set-variable SILENCE_UNUSED_BUT_SET)
if(SILENCE_UNUSED_BUT_SET)
  set(DLMALLOC_CFLAGS "-Wno-unused-but-set-variable")
endif()

target_include_directories(shm_allocator PRIVATE
    ${PROJECT_BINARY_DIR}/include)

set_property(
  SOURCE ${dlmalloc_sources}
  APPEND_STRING
  PROPERTY
    COMPILE_FLAGS
    " -DONLY_MSPACES -DUSE_SPIN_LOCKS -DMSPACES -DHAVE_USR_INCLUDE_MALLOC_H -DMALLOC_INSPECT_ALL -DHAVE_MMAP=1 -DHAVE_MORECORE=0 -Wno-expansion-to-defined ${DLMALLOC_CFLAGS}"
)
