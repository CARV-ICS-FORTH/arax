add_executable(vinegrind main.cpp)
set_target_properties(vinegrind PROPERTIES COMPILE_FLAGS -std=c++11)
target_link_libraries(vinegrind vine_st)
