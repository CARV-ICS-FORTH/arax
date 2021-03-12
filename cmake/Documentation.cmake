find_package(Doxygen)

if(DOXYGEN_FOUND)
  option(BUILD_DOCS "Build documentation" ON) # skip build_check
else()
  option(BUILD_DOCS "Build documentation" OFF) # skip build_check
endif()

if(BUILD_DOCS)
  message(STATUS "Documentation generation enabled")
  if(DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
  else()
    set(DOXYGEN_HAVE_DOT "NO")
  endif()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  add_custom_target(docs_pre COMMAND doxygen)
  add_custom_target(
    doc
    DEPENDS docs_pre
    COMMENT "Doc files at: ${CMAKE_CURRENT_BINARY_DIR}/docs/html/index.html")
  add_custom_target(docs DEPENDS doc)
else()
  message(STATUS "Documentation generation disabled")
endif()
