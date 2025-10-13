# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

add_subdirectory(tiling)

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
