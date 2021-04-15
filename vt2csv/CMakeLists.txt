add_executable(vt2csv main.cpp)
set_target_properties(vt2csv PROPERTIES COMPILE_FLAGS -std=c++11)
target_link_libraries(vt2csv vine_st)