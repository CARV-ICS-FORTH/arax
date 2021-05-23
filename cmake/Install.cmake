include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/vinetalk)

install(TARGETS vine EXPORT vinetalk LIBRARY DESTINATION lib INCLUDES DESTINATION include)
install(TARGETS vine_st EXPORT vinetalk LIBRARY DESTINATION lib INCLUDES DESTINATION include)

if(TARGET vdf)
install(TARGETS vdf DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
install(TARGETS vinegrind DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

install(FILES ${CMAKE_SOURCE_DIR}/cmake/vinetalk-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(EXPORT vinetalk DESTINATION ${CMAKE_INSTALL_PREFIX})

export(EXPORT vinetalk
    FILE "${CMAKE_CURRENT_BINARY_DIR}/vinetalk.cmake"
)
