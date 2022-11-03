list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add backend folder to cmake build
function (add_backend backend)
	add_subdirectory(src/${backend})
endfunction(add_backend)

# If backend should by default be enabled
function (enable_backend enable)
	if(${enable})
		option(ENABLE_${backend} "Enable ${backend} backend." ON)
	else()
		option(ENABLE_${backend} "Enable ${backend} backend." OFF)
	endif(${enable})
endfunction(enable_backend)

function (register_builtin builtin)
	install(TARGETS ${builtin} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/araxcontroller)
endfunction(register_builtin)

function (vc_link_target lib)
	target_sources(arax_controller PUBLIC $<TARGET_OBJECTS:${lib}>)
endfunction(vc_link_target)

function (vc_link_lib lib)
	target_sources(arax_controller PUBLIC ${lib})
endfunction(vc_link_lib)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/definesEnable.h.cmake ${CMAKE_CURRENT_SOURCE_DIR}/include/definesEnable.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pre-commit ${CMAKE_CURRENT_SOURCE_DIR}/.git/hooks/ @ONLY)
