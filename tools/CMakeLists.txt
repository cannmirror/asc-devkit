# if(NOT ENABLE_TEST)
#   add_subdirectory(opbuild)
#   add_subdirectory(ascc)
#   add_subdirectory(build)
#   add_subdirectory(aclrtc)
# endif()

install(FILES scripts/ascendc_parse_dumpinfo.py
  DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)
