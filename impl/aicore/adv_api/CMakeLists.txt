# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

add_subdirectory(tiling)

install(
  FILES ${KERNEL_TILING_HAED}
  DESTINATION
    ${INSTALL_LIBRARY_DIR}/ascendc/include/highlevel_api/kernel_tiling/
  OPTIONAL)

# copy detail to highlevel_api impl
install(
  DIRECTORY detail/
  DESTINATION ${INSTALL_LIBRARY_DIR}/ascendc/include/highlevel_api/impl
  OPTIONAL FILES_MATCHING
  PATTERN "*.h"
  PATTERN "CMakeLists.txt" EXCLUDE)

# cp include to highlevel_api lib
install(
  DIRECTORY ${ASCENDC_INCLUDE_DIR}/aicore/adv_api/
  DESTINATION ${INSTALL_LIBRARY_DIR}/ascendc/include/highlevel_api/lib
  OPTIONAL
  PATTERN "OWNERS" EXCLUDE)

configure_file(${ASCENDC_ADV_API_CMAKE_DIR}/kernel_headers.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/kernel_headers.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kernel_headers.cmake)

if(NOT BUILD_OPEN_PROJECT)
  install_package(
    PACKAGE
    kernel_tiling
    TARGETS
    kernel_tiling_headers
    FILES
    ${KERNEL_TILING_HAED}
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/ascendc/include/highlevel_api/kernel_tiling)

  install(SCRIPT ${ASCENDC_DIR}/cmake/adv_api/kernel_directory.cmake)
endif()
