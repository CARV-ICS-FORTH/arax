include(FetchContent)

FetchContent_Declare(tlsf
    GIT_REPOSITORY https://github.com/mattconte/tlsf
    GIT_SHALLOW true
    PREFIX .
    DOWNLOAD_DIR .
)
FetchContent_MakeAvailable(tlsf)

add_library(shm_allocator OBJECT alloc.c ${tlsf_SOURCE_DIR}/tlsf.c)
target_include_directories(shm_allocator PRIVATE
    ${CMAKE_BINARY_DIR}/include ${tlsf_SOURCE_DIR})
