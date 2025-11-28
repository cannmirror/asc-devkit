# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
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
  COMMAND cp -rf ${ASCENDC_DIR}/cmake/asc ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  COMMAND cp -rf ${ASCENDC_DIR}/tools/ascc/cmake/. ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  COMMAND cp -rf ${ASCENDC_DIR}/tools/build/scripts/AICPU_CMake ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
)
add_custom_target(ascendc_kernel_cmake ALL DEPENDS ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake)

install(DIRECTORY ${CMAKE_BINARY_DIR}/ascendc_kernel_cmake
  DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
  PATTERN "test_*" EXCLUDE
)