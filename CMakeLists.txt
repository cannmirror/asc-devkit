# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(asc-devkit)

option(BUILD_OPEN_PROJECT "Build open asc-devkit lib project" ON)

set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project asc-devkit with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build asc-devkit with cann pkg" ON)
else()
  message(STATUS "compile project asc-devkit with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build asc-devkit with cann source" OFF)
endif()

set(ASCENDC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASCENDC_INCLUDE_DIR ${ASCENDC_DIR}/include)
set(ASCENDC_IMPL_DIR ${ASCENDC_DIR}/impl)


if(BUILD_OPEN_PROJECT)
  include(cmake/config.cmake)
  set(INSTALL_LIBRARY_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux)
else()
  set(INSTALL_LIBRARY_DIR lib)
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

add_subdirectory(impl)

if(BUILD_OPEN_PROJECT)
  include(cmake/intf_pub_linux.cmake)
  include(cmake/dependencies.cmake)
endif()

add_subdirectory(tools)

set(ASC_DEVKIT_VERSION_FILE ${CMAKE_CURRENT_BINARY_DIR}/_CPack_Packages/makeself_staging/share/info/asc-devkit/version.info)
add_custom_command(
    OUTPUT ${ASC_DEVKIT_VERSION_FILE}
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/update_version_info/update_version_info.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/version.info
            ${ASC_DEVKIT_VERSION_FILE}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/version.info
    VERBATIM
)

add_custom_target(generate_ASC_DEVKIT_VERSION_FILE ALL
    DEPENDS ${ASC_DEVKIT_VERSION_FILE}
)

if(ENABLE_TEST)
  add_subdirectory(tests)
endif()

if(PACKAGE_OPEN_PROJECT)
  cmake_minimum_required(VERSION 3.16)
  project(asc-devkit VERSION 1.0.0)
  include(CMakePrintHelpers)
  set(CMAKE_CXX_STANDARD 17)
  include(cmake/package.cmake)
  include(cmake/version_info.cmake)
endif()
