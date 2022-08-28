add_executable(araxgrind main.cpp Args.cpp)
target_link_libraries(araxgrind arax_st)
install(TARGETS araxgrind DESTINATION bin)
