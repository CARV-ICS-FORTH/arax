include(GNUInstallDirs)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include/arax
  COMPONENT headers)

install(TARGETS arax
  EXPORT arax
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include)

install(TARGETS arax_st
  EXPORT arax
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include)

install(FILES ${PROJECT_SOURCE_DIR}/cmake/arax-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_PROJECT_NAME} COMPONENT libs)
install(EXPORT arax DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_PROJECT_NAME} COMPONENT libs)

export(EXPORT arax
    FILE "${PROJECT_BINARY_DIR}/arax.cmake"
)

add_custom_target(uninstall
  COMMAND xargs rm -v < ${PROJECT_BINARY_DIR}/install_manifest.txt
)
