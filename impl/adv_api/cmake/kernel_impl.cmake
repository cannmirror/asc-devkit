# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

set(ASCENDC_API_PATH @INSTALL_LIBRARY_DIR@)
set(ASCENDC_INSTALL_BASE_PATH ${CMAKE_INSTALL_PREFIX}/${ASCENDC_API_PATH})
file(
    CREATE_LINK ../../utils/std
    ${ASCENDC_INSTALL_BASE_PATH}/asc/impl/adv_api/detail/std
    SYMBOLIC)

file(MAKE_DIRECTORY ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/basic_api/impl/utils/)
file(
    CREATE_LINK ../../../../../asc/impl/utils/std
    ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/basic_api/impl/utils/std
    SYMBOLIC)
