if(Poco_FOUND)
    find_package(Poco REQUIRED Net)
    add_executable(vdf vdf.cpp Misc.cpp WebUI.cpp Collector.cpp)
    set_target_properties(vdf PROPERTIES COMPILE_FLAGS -std=c++11)
    target_link_libraries(vdf vine rt pthread ${Poco_Util_LIBRARY} ${Poco_Foundation_LIBRARY} ${Poco_Net_LIBRARY})
endif()

