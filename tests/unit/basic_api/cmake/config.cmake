# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
find_package(GTest CONFIG)
if (NOT ${GTest_FOUND})
    if (EXISTS "${ASCEND_HOME_PATH}/opensdk/opensdk/gtest")
        list(APPEND CMAKE_PREFIX_PATH ${ASCEND_HOME_PATH}/opensdk/opensdk/gtest)
        find_package(GTest CONFIG)
    endif ()
endif ()
if (NOT ${GTest_FOUND})
    message(FATAL_ERROR "Can't find any googletest.")
endif ()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(third_party_TEM_DIR ${ASCENDC_TOOLS_ROOT_DIR}/build/tmp)

include(${ASCENDC_TOOLS_ROOT_DIR}/third_party/json.cmake)
include(${ASCENDC_TOOLS_ROOT_DIR}/third_party/boost.cmake)
include(${ASCENDC_TOOLS_ROOT_DIR}/third_party/mockcpp.cmake)
