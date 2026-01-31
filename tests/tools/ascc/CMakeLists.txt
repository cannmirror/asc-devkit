# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
project(ascendc_tools_asc_plugin_tests)
cmake_minimum_required(VERSION 3.16.0)

########################### asc_plugin ut ###########################
set(ASC_PLUGIN_TEST_HEADERS_FILES
    ${ASCENDC_DIR}/tools/ascc/asc_plugin/include/internal
    ${ASCENDC_DIR}/tools/ascc/asc_plugin/include/external
    ${CMAKE_CURRENT_LIST_DIR}
)

file(GLOB ASC_PLUGIN_SRC_FILES
    ${ASCENDC_DIR}/tools/ascc/asc_plugin/src/*.cpp
)

file(GLOB TEST_ASC_PLUGIN_SRC_FILES
    ${ASCENDC_DIR}/tests/main.cpp
    llvm/Support/*.cpp
    asc_plugin/*.cpp
)

add_executable(ascendc_ut_asc_plugin
    ${ASC_PLUGIN_SRC_FILES}
    ${TEST_ASC_PLUGIN_SRC_FILES}
)

target_compile_definitions(ascendc_ut_asc_plugin PRIVATE
    UT_TEST
    ASCENDC_CPU_DEBUG=1
)

target_include_directories(ascendc_ut_asc_plugin PRIVATE
    ${ASC_PLUGIN_TEST_HEADERS_FILES}
    ${JSON_INCLUDE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_compile_options(ascendc_ut_asc_plugin PRIVATE
    -Werror
)

target_link_libraries(ascendc_ut_asc_plugin PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_basic>
    json
    -Wl,--no-as-needed
    c_sec
    dl
    -Wl,--as-needed
)

run_llt_test(
    TARGET ascendc_ut_asc_plugin
    TASK_NUM 1
)
