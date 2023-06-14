include(ExternalProject)

ExternalProject_Add(
	ELFIO
	GIT_REPOSITORY https://github.com/serge1/ELFIO.git
	GIT_TAG main
	GIT_SHALLOW true
	GIT_PROGRESS true
	BUILD_COMMAND ""
	CMAKE_ARGS ""
	INSTALL_COMMAND ""
	UPDATE_COMMAND ""
)

ExternalProject_Get_property(ELFIO SOURCE_DIR)

add_library(elfio INTERFACE)
target_include_directories(elfio INTERFACE ${SOURCE_DIR})
add_dependencies(elfio ELFIO)
target_compile_features(elfio INTERFACE cxx_std_17)
# file(MAKE_DIRECTORY ${SOURCE_DIR})
