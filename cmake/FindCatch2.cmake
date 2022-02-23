include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.8
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(Catch2)

mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_CATCH2)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_CATCH2)
mark_as_advanced(Catch2_DIR)
mark_as_advanced(CATCH_BUILD_EXAMPLES)
mark_as_advanced(CATCH_BUILD_EXTRA_TESTS)
mark_as_advanced(CATCH_BUILD_STATIC_LIBRARY)
mark_as_advanced(CATCH_BUILD_TESTING)
mark_as_advanced(CATCH_ENABLE_COVERAGE)
mark_as_advanced(CATCH_ENABLE_WERROR)
mark_as_advanced(CATCH_INSTALL_DOCS)
mark_as_advanced(CATCH_INSTALL_HELPERS)
mark_as_advanced(CATCH_USE_VALGRIND)
