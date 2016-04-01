SET(dlmalloc_sources malloc.c malloc.h)
add_library(dlmalloc OBJECT ${dlmalloc_sources})
set_property(SOURCE ${dlmalloc_sources} APPEND_STRING PROPERTY COMPILE_FLAGS " -DONLY_MSPACES -DUSE_SPIN_LOCKS -DMSPACES ")
