# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

list(APPEND PRODUCT_TYPE_LIST ascend910B1_ACT)
set(PRODUCT_TYPE_LIST ${PRODUCT_TYPE_LIST}
    CACHE STRING "Supported product types" FORCE)

set(ASCENDC_TEST_ascend910B1_ACT_HEADER_FILES
    ${ASCENDC_DIR}/ops_templates/act
    PARENT_SCOPE
)

set(ASCENDC_TEST_ascend910B1_ACT_CASE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/block/test_block_mmad_multi_block_with_layout.cpp
    PARENT_SCOPE
)
