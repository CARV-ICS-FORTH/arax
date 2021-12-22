include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/  DESTINATION include/vinetalk)

install(TARGETS vine
  EXPORT vinetalk
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include)

install(TARGETS vine_st 
  EXPORT vinetalk 
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include)

install(FILES ${CMAKE_SOURCE_DIR}/cmake/vinetalk-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(EXPORT vinetalk DESTINATION ${CMAKE_INSTALL_PREFIX})

export(EXPORT vinetalk
    FILE "${CMAKE_CURRENT_BINARY_DIR}/vinetalk.cmake"
)

add_custom_target( uninstall
  COMMAND xargs rm -v < ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
)
