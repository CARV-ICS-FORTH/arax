add_executable(vinegrind main.cpp Args.cpp)
target_link_libraries(vinegrind vine_st)
install(TARGETS vinegrind DESTINATION bin)
