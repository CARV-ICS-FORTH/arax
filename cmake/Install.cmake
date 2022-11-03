include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/  DESTINATION include/arax)

install(TARGETS arax
  EXPORT arax
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include)

install(TARGETS arax_st
  EXPORT arax
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include)

install(FILES ${CMAKE_SOURCE_DIR}/cmake/arax-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_PROJECT_NAME})
install(EXPORT arax DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_PROJECT_NAME})

export(EXPORT arax
    FILE "${CMAKE_CURRENT_BINARY_DIR}/arax.cmake"
)

add_custom_target(uninstall
  COMMAND xargs rm -v < ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
)
