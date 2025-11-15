if(NOT ENABLE_TEST)
  add_subdirectory(ascc)
  add_subdirectory(build)
  add_subdirectory(aclrtc)
endif()

install(FILES scripts/ascendc_parse_dumpinfo.py
  DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  COMMAND rm -rf ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  COMMAND cp -rf ${ASCENDC_DIR}/cmake ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  COMMAND cp -rf ${ASCENDC_DIR}/tools/ascc/cmake/. ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  COMMAND cp -rf ${ASCENDC_DIR}/tools/build/scripts/AICPU_CMake ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
)
add_custom_target(ascendc_kernel_cmake ALL DEPENDS ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake)

install(DIRECTORY ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
  PATTERN "test_*" EXCLUDE
)