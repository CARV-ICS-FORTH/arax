find_package(Poco QUIET)

if(Poco_FOUND AND NOT SKIP_VDF)
  find_package(Poco REQUIRED Net)

  aux_source_directory(Views views)

  add_executable(vdf vdf.cpp Misc.cpp WebUI.cpp Pallete.cpp ${views})
  target_link_libraries(vdf arax_st ${Poco_Util_LIBRARY}
                        ${Poco_Foundation_LIBRARY} ${Poco_Net_LIBRARY})
  target_include_directories(vdf PRIVATE ${CMAKE_BINARY_DIR}/include)
  install(TARGETS vdf DESTINATION bin COMPONENT utils)
endif()
