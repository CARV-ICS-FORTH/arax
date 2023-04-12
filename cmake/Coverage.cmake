option(COVERAGE "Enable coverage reports" OFF)

if(COVERAGE)
  set(CMAKE_C_FLAGS "-fprofile-arcs -ftest-coverage ${CMAKE_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage ${CMAKE_EXE_LINKER_FLAGS}")
  set(CMAKE_C_OUTPUT_EXTENSION_REPLACE ON)
  set(BUILD_TESTS 1)
endif()

if(COVERAGE)
  find_program(GCOVR gcovr REQUIRED)
  mark_as_advanced(FORCE GCOVR)
  add_custom_target(
    coverage_pre
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND make -j
    COMMAND -make -j test
    COMMAND -rm -Rf coverage
    COMMAND mkdir coverage
    COMMAND
      ${GCOVR} -p -d --gcov-ignore-parse-errors --exclude-directories
      'tests' --exclude-directories 'src/alloc' -r ${CMAKE_CURRENT_SOURCE_DIR}/ --html-title
      'Arax Coverage Report' --html --html-details --html-self-contained -o coverage/coverage.html
      -s
    COMMAND sed -i 's/GCC Code/Arax/g' coverage/*.html)
  add_custom_target(
    coverage
    DEPENDS coverage_pre
    COMMENT
      "Coverage results at: ${CMAKE_CURRENT_BINARY_DIR}/coverage/coverage.html")

  add_custom_target(cov DEPENDS coverage)
endif()
