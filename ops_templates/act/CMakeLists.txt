# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

install(
  DIRECTORY ${ASCENDC_DIR}/ops_templates/act
  DESTINATION ${INSTALL_LIBRARY_DIR}/ascendc/ops_templates/
  OPTIONAL
  PATTERN "OWNERS" EXCLUDE)

file(
  CREATE_LINK
  "ops_templates/act"
  ${CMAKE_BINARY_DIR}/act
  SYMBOLIC)

install(
  DIRECTORY ${CMAKE_BINARY_DIR}/act
  DESTINATION ${INSTALL_LIBRARY_DIR}/ascendc/
  OPTIONAL)