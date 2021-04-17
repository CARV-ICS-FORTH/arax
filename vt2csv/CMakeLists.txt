aux_source_directory(. vt2csv_src)
add_executable(vt2csv ${vt2csv_src})
set_target_properties(vt2csv PROPERTIES COMPILE_FLAGS -std=c++11)
target_link_libraries(vt2csv vine_st)