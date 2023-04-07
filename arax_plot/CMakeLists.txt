aux_source_directory(. arax_plot_src)
add_executable(arax_plot ${arax_plot_src})
target_link_libraries(arax_plot arax_st)
install(TARGETS arax_plot DESTINATION bin COMPONENT utils)
