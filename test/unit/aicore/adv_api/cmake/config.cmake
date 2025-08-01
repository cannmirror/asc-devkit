# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH})
  get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.."
                         ABSOLUTE)
else()
  set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
  message(
    FATAL_ERROR
      "${ASCEND_CANN_PACKAGE_PATH} does not exist, please install the cann package and set environment variables."
  )
endif()

set(GENERATE_CPP_COV
    ${ASCENDC_ADV_API_TESTS_DIR}/cmake/scripts/generate_cpp_cov.sh)

find_package(GTest CONFIG)
if(NOT ${GTest_FOUND})
  if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/gtest")
    list(APPEND CMAKE_PREFIX_PATH
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/gtest)
    find_package(GTest CONFIG)
  endif()
endif()
if(NOT ${GTest_FOUND})
  message(FATAL_ERROR "Can't find any googletest.")
endif()
