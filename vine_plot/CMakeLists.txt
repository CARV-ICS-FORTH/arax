aux_source_directory(. vine_plot_src)
add_executable(vine_plot ${vine_plot_src})
target_link_libraries(vine_plot vine_st)
install(TARGETS vine_plot DESTINATION bin)
