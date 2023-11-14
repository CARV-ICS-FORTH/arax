file(GLOB INC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(COPY ${INC_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/alloc/alloc.h
     DESTINATION ${CMAKE_BINARY_DIR}/include/arch)
file(GLOB UTILS_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.h")
file(COPY ${UTILS_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/utils)
file(GLOB CORE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.h")
file(COPY ${CORE_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/core)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/async/${async_architecture}/async.h
     DESTINATION ${CMAKE_BINARY_DIR}/include/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/async/async_api.h
     DESTINATION ${CMAKE_BINARY_DIR}/include/)

# Generate conf.h
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/conf.h.cmake
               ${CMAKE_BINARY_DIR}/include/conf.h)
