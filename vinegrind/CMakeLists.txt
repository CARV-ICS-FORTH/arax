add_executable(vinegrind main.cpp Args.cpp)
set_target_properties(vinegrind PROPERTIES COMPILE_FLAGS -std=c++11)
target_link_libraries(vinegrind vine_st)
install(TARGETS vinegrind DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
