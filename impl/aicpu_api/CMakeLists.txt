# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
add_library(aicpu_api STATIC aicpu_dump.cpp)

target_compile_options(aicpu_api PRIVATE
    -O2
    -std=c++17
    -D_FORTIFY_SOURCE=2
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fPIC>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility=hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fstack-protector-strong>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-ftrapv>
)

if(BUILD_OPEN_PROJECT)
    target_include_directories(aicpu_api PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
else()
    target_include_directories(aicpu_api PRIVATE
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/asc/asc-devkit/include
    )
endif()

target_compile_definitions(aicpu_api PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},device>:__AICPU_DEVICE__>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:_FORTIFY_SOURCE=2>
    _GLIBCXX_USE_CXX11_ABI=0
)

install(TARGETS aicpu_api
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)

install(FILES
    ${ASCENDC_DIR}/include/aicpu_api/aicpu_api.h
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/aicpu_api OPTIONAL
)