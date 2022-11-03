include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/  DESTINATION include/araxcontroller)

install(TARGETS arax_controller DESTINATION bin EXPORT araxcontroller)

add_custom_target( uninstall
  COMMAND xargs rm -v < ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
)

export(EXPORT araxcontroller
    FILE "${CMAKE_CURRENT_BINARY_DIR}/araxcontroller.cmake"
)

install(EXPORT araxcontroller DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_PROJECT_NAME})

install(FILES ${CMAKE_SOURCE_DIR}/cmake/araxcontroller-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_PROJECT_NAME})
