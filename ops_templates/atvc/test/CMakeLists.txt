# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

list(APPEND PRODUCT_TYPE_LIST ascend910B1_ATVC)
set(PRODUCT_TYPE_LIST ${PRODUCT_TYPE_LIST}
    CACHE STRING "Supported product types" FORCE)

set(ASCENDC_TEST_ascend910B1_ATVC_HEADER_FILES
    ${ASCENDC_DIR}/ops_templates/atvc/include
    ${ASCENDC_DIR}/ops_templates/atvc/examples
    PARENT_SCOPE
)

set(ASCENDC_TEST_ascend910B1_ATVC_CASE_SRC_FILES
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_broadcast_to.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_sum.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_add_with_broadcast.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_addcmul.cpp
    PARENT_SCOPE
)

set(ASCENDC_TILING_TEST_ascend910B1_ATVC_CASE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling/test_atvc_tiling.cpp
    PARENT_SCOPE
)
