find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
  option(BUILD_DOCS "Build documentation" ON) # skip build_check
else()
  option(BUILD_DOCS "Build documentation" OFF) # skip build_check
endif()

if(BUILD_DOCS)
  if(DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
  else()
    set(DOXYGEN_HAVE_DOT "NO")
  endif()
  file(DOWNLOAD https://github.com/jothepro/doxygen-awesome-css/raw/main/doxygen-awesome.css ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen-awesome.css)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen-awesome.css "#main-menu {padding-right:5em;}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  add_custom_target(docs_pre 
    COMMAND doxygen 
    COMMAND sed -i 's/<html/<html class="dark-mode"/g' ${CMAKE_CURRENT_BINARY_DIR}/docs/html/*.html)
  add_custom_target(
    doc
    DEPENDS docs_pre
    COMMENT "Doc files at: ${CMAKE_CURRENT_BINARY_DIR}/docs/html/index.html")
  add_custom_target(docs DEPENDS doc)
endif()
