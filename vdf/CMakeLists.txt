if(Poco_FOUND)
  find_package(Poco REQUIRED Net)

  aux_source_directory(Views views)

  add_executable(vdf vdf.cpp Misc.cpp WebUI.cpp Pallete.cpp ${views})
  set_target_properties(vdf PROPERTIES COMPILE_FLAGS -std=c++11)
  target_link_libraries(vdf vine_st ${Poco_Util_LIBRARY}
                        ${Poco_Foundation_LIBRARY} ${Poco_Net_LIBRARY})
  target_include_directories(vdf PRIVATE ${CMAKE_BINARY_DIR}/include)
  install(TARGETS vdf DESTINATION bin)
endif()
