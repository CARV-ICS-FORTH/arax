set(CPACK_PACKAGE_VENDOR "CARV - ICS - FORTH")
set(CPACK_SOURCE_IGNORE_FILES "build;.git;ci_scripts")
set(CPACK_SET_DESTDIR ON)
set(CPACK_SOURCE_GENERATOR "STGZ")
set(CPACK_GENERATOR "STGZ;DEB")
set(CPACK_PACKAGE_CONTACT "mavridis@ics.forth.gr")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/tools/lic")
set(CPACK_MONOLITHIC_INSTALL ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_COMPONENTS_ALL libs controller headers utils)

include(CPack)

cpack_add_component(libs REQUIRED)
cpack_add_component(controller DEPENDS libs)
cpack_add_component(headers DEPENDS libs)
cpack_add_component(utils DEPENDS libs DISABLED)
