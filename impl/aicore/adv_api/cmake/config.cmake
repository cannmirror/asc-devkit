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

execute_process(
  COMMAND bash ${ASCENDC_ADV_API_CMAKE_DIR}/scripts/check_version_compatiable.sh
          ${ASCEND_CANN_PACKAGE_PATH} toolkit ${ASCENDC_DIR}/version.info
  RESULT_VARIABLE result
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CANN_VERSION)

if(result)
  message(FATAL_ERROR "${CANN_VERSION}")
else()
  string(TOLOWER ${CANN_VERSION} CANN_VERSION)
endif()

if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_CURRENT_SOURCE_DIR}/output"
      CACHE STRING "path for install()" FORCE)
endif()

set(HI_PYTHON
    "python3"
    CACHE STRING "python executor")
set(PRODUCT_SIDE host)

set(TILING_API_LIB ${ASCEND_CANN_PACKAGE_PATH}/lib64/libtiling_api.a)
if(NOT EXISTS "${TILING_API_LIB}")
  message(
    FATAL_ERROR
      "${TILING_API_LIB} does not exist, please check whether the toolkit package is installed."
  )
endif()

if(ENABLE_TEST)
  set(CMAKE_SKIP_RPATH FALSE)
else()
  set(CMAKE_SKIP_RPATH TRUE)
endif()

set(ASCENDC_ADV_API_OBJ ASCENDC_adv_api_obj)
set(ASCENDC_ADV_API_OBJ_PATH ${CMAKE_CURRENT_BINARY_DIR}/ASCENDC_adv_api_objs)

file(REMOVE_RECURSE ${ASCENDC_ADV_API_OBJ_PATH})
file(MAKE_DIRECTORY ${ASCENDC_ADV_API_OBJ_PATH})

execute_process(
  COMMAND ${CMAKE_AR} -x ${TILING_API_LIB}
  WORKING_DIRECTORY ${ASCENDC_ADV_API_OBJ_PATH}
  RESULT_VARIABLE result
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(${ASCENDC_ADV_API_OBJ} OBJECT IMPORTED)
set_target_properties(
  ${ASCENDC_ADV_API_OBJ}
  PROPERTIES
    IMPORTED_OBJECTS
    "${ASCENDC_ADV_API_OBJ_PATH}/platform_ascendc.cpp.o;${ASCENDC_ADV_API_OBJ_PATH}/context_builder.cpp.o;${ASCENDC_ADV_API_OBJ_PATH}/context_builder_impl.cpp.o;${ASCENDC_ADV_API_OBJ_PATH}/template_argument.cpp.o"
)

get_filename_component(ASCENDC_API_ADV_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}"
                       ABSOLUTE)
include(${ASCENDC_API_ADV_CMAKE_DIR}/intf_pub_linux.cmake)
