configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pre-commit
       ${CMAKE_CURRENT_SOURCE_DIR}/.git/hooks/ @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare-commit-msg
       ${CMAKE_CURRENT_SOURCE_DIR}/.git/hooks/ @ONLY)

add_custom_target(
	precommit-all
	COMMAND ${CMAKE_BINARY_DIR}/.miniconda/bin/conda run pre-commit run -a
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS ${CMAKE_BINARY_DIR}/.miniconda
)

add_custom_target(
	precommit
	COMMAND ${CMAKE_BINARY_DIR}/.miniconda/bin/conda run pre-commit run
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS ${CMAKE_BINARY_DIR}/.miniconda
)

add_custom_command(
	OUTPUT ${CMAKE_BINARY_DIR}/.miniconda
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/.git/hooks/pre-commit
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# pre-commit alias for precommit
add_custom_target(pre-commit-all DEPENDS precommit-all)
add_custom_target(pre-commit DEPENDS precommit)

