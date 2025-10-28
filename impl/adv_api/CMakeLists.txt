# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_subdirectory(tiling)

install(
  FILES ${KERNEL_TILING_HAED}
  DESTINATION
    ${INSTALL_LIBRARY_DIR}/asc/include/adv_api/
  OPTIONAL)

# copy detail to asc impl
install(
  DIRECTORY detail/
  DESTINATION ${INSTALL_LIBRARY_DIR}/asc/impl/adv_api/detail
  OPTIONAL FILES_MATCHING
  PATTERN "*.h"
  PATTERN "CMakeLists.txt" EXCLUDE)

# copy tiling to asc impl
install(
  DIRECTORY tiling/
  DESTINATION ${INSTALL_LIBRARY_DIR}/asc/impl/adv_api/tiling
  OPTIONAL FILES_MATCHING
  PATTERN "*.h"
  PATTERN "CMakeLists.txt" EXCLUDE)

# cp include to asc include
install(
  DIRECTORY ${ASCENDC_INCLUDE_DIR}/adv_api/
  DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/adv_api/
  OPTIONAL
  PATTERN "OWNERS" EXCLUDE)

configure_file(
  ${ASCENDC_ADV_API_CMAKE_DIR}/kernel_headers.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/kernel_headers.cmake
  @ONLY
)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kernel_headers.cmake)

configure_file(${ASCENDC_ADV_API_CMAKE_DIR}/kernel_impl.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl.cmake)

if(NOT BUILD_OPEN_PROJECT)
  install_package(
    PACKAGE kernel_tiling
    TARGETS kernel_tiling_headers
    FILES ${KERNEL_TILING_HAED}
    DESTINATION ${INSTALL_INCLUDE_DIR}/asc/include/adv_api/)

  install(SCRIPT ${ASCENDC_DIR}/impl/adv_api/cmake/kernel_directory.cmake)
endif()

configure_file(
    ${ASCENDC_ADV_API_CMAKE_DIR}/utils_headers.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/utils_headers.cmake
    @ONLY
)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/utils_headers.cmake)
