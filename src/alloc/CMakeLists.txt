set(SHM_ALLOCATOR
    dlmalloc
    CACHE STRING "Shared Memory allocator")
set_property(CACHE SHM_ALLOCATOR PROPERTY STRINGS dlmalloc jemalloc tlsf)

add_subdirectory(${SHM_ALLOCATOR})
