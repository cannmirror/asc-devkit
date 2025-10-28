# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16.0)
project(ascend-c-dev)

option(BUILD_OPEN_PROJECT "Build open ascendc lib project" ON)

set(ASCENDC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASCENDC_INCLUDE_DIR ${ASCENDC_DIR}/include)
set(ASCENDC_IMPL_DIR ${ASCENDC_DIR}/impl)

set(ASCENDC_ADV_API_IMPL_DIR ${ASCENDC_IMPL_DIR}/adv_api)
set(ASCENDC_ADV_API_CMAKE_DIR ${ASCENDC_ADV_API_IMPL_DIR}/cmake)

if(BUILD_OPEN_PROJECT)
  include(${ASCENDC_ADV_API_CMAKE_DIR}/config.cmake)
  set(INSTALL_LIBRARY_DIR packages/${CMAKE_SYSTEM_PROCESSOR}-linux)
else()
  set(INSTALL_LIBRARY_DIR lib)
endif()

set(GEN_KERENL_TILING_DATA_SCRIPT
    ${ASCENDC_ADV_API_CMAKE_DIR}/scripts/gen_kernel_tiling_data_def.py)
set(GENERATE_INSTALL_SCRIPT
    ${ASCENDC_ADV_API_CMAKE_DIR}/scripts/generate_install_script.sh)
set(TILING_DATA_DEF_DIR ${ASCENDC_INCLUDE_DIR}/adv_api)
set(KERNEL_TILING_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(KERNEL_TILING_HAED ${KERNEL_TILING_DIR}/kernel_tiling/kernel_tiling.h)

if(NOT BUILD_OPEN_PROJECT)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tiling/tiling_api.h
    COMMAND mkdir ${CMAKE_CURRENT_BINARY_DIR}/tiling
    COMMAND ln -s ${ASCENDC_INCLUDE_DIR}/adv_api/tiling_api.h
            ${CMAKE_CURRENT_BINARY_DIR}/tiling/tiling_api.h
    DEPENDS ${ASCENDC_INCLUDE_DIR}/adv_api/tiling_api.h)
  add_custom_target(gen_kernel_api ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tiling/tiling_api.h)
endif()

# generate kernel_tiling/kernel_tiling.h
add_custom_command(
  OUTPUT ${KERNEL_TILING_HAED}
  COMMAND ${HI_PYTHON} ${GEN_KERENL_TILING_DATA_SCRIPT} ${TILING_DATA_DEF_DIR}
          ${KERNEL_TILING_HAED}
  DEPENDS ${GEN_KERENL_TILING_DATA_SCRIPT})

add_custom_target(gen_kernel_tiling ALL DEPENDS ${KERNEL_TILING_HAED})

add_library(kernel_tiling_headers INTERFACE)
add_dependencies(kernel_tiling_headers gen_kernel_tiling)
if(NOT BUILD_OPEN_PROJECT)
  add_dependencies(kernel_tiling_headers gen_kernel_api)
endif()
target_include_directories(
  kernel_tiling_headers
  INTERFACE $<INSTALL_INTERFACE:include>
            $<INSTALL_INTERFACE:include/tikcpp>
            $<INSTALL_INTERFACE:include/tikcpp/tikcfw>
            $<INSTALL_INTERFACE:include/tikcpp/tikcfw/kernel_tiling>
            $<BUILD_INTERFACE:${KERNEL_TILING_DIR}>)

add_subdirectory(impl)
add_subdirectory(ops_templates)
if(ENABLE_TEST)
  add_subdirectory(test)
endif()

# if(PACKAGE_OPEN_PROJECT)
#   cmake_minimum_required(VERSION 3.16)
#   project(ascendc VERSION 1.0.0)
#   include(CMakePrintHelpers)
#   set(CMAKE_CXX_STANDARD 17)
#   include(tools/package.cmake)
# endif()
