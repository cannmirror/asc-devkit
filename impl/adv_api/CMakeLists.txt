# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_subdirectory(tiling)

if (BUILD_OPEN_PROJECT)
  include(ExternalProject)
  ExternalProject_Add(device_tiling
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tiling
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
    -G ${CMAKE_GENERATOR}
    -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}
    -DKERNEL_TILING_DIR=${KERNEL_TILING_DIR}
    -DASCENDC_DIR=${ASCENDC_DIR}
    -DINSTALL_LIBRARY_DIR=${INSTALL_LIBRARY_DIR}/lib64/device
    -DASCENDC_INCLUDE_DIR=${ASCENDC_INCLUDE_DIR}
    -DASCENDC_ADV_API_CMAKE_DIR=${ASCENDC_ADV_API_CMAKE_DIR}
    -DASCENDC_API_ADV_MULT_INSTALL_DIR=${ASCENDC_API_ADV_MULT_INSTALL_DIR}
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DSYS_VERSION=${SYS_VERSION}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_C_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
    -DPRODUCT_SIDE=device
    <SOURCE_DIR>
    BUILD_ALWAYS 1)
endif()


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

configure_file(${ASCENDC_ADV_API_CMAKE_DIR}/kernel_headers.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/kernel_headers.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kernel_headers.cmake)

configure_file(${ASCENDC_ADV_API_CMAKE_DIR}/kernel_impl.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl.cmake)

configure_file(${ASCENDC_ADV_API_CMAKE_DIR}/kernel_directory.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/kernel_directory.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kernel_directory.cmake)

if(NOT BUILD_OPEN_PROJECT)
  install_package(
    PACKAGE kernel_tiling
    TARGETS kernel_tiling_headers
    FILES ${KERNEL_TILING_HAED}
    DESTINATION ${INSTALL_INCLUDE_DIR}/asc/include/adv_api/)

endif()

configure_file(
    ${ASCENDC_ADV_API_CMAKE_DIR}/utils_headers.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/utils_headers.cmake
    @ONLY
)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/utils_headers.cmake)
