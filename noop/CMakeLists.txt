add_library(nooplib EXCLUDE_FROM_ALL SHARED noop.cpp)
target_compile_definitions(nooplib PRIVATE BUILD_SO)
target_link_libraries(nooplib arax_st)

add_executable(noop EXCLUDE_FROM_ALL noop.cpp)
target_compile_definitions(noop PRIVATE BUILD_MAIN)
target_link_libraries(noop arax_st)
add_dependencies(noop nooplib)

add_executable(noop_stress EXCLUDE_FROM_ALL noop_stress.cpp)
target_compile_definitions(noop_stress PRIVATE BUILD_MAIN)
target_link_libraries(noop_stress arax_st)
add_dependencies(noop nooplib)
