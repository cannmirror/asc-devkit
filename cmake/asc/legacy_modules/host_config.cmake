# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set(ascend910b_list ascend910b1 ascend910b2 ascend910b2c ascend910b3 ascend910b4 ascend910b4-1 ascend910_9391 ascend910_9381 ascend910_9372 ascend910_9392 ascend910_9382 ascend910_9362)
set(ascend910_list  ascend910a ascend910proa ascend910b ascend910prob ascend910premiuma)
set(ascend310p_list ascend310p1 ascend310p3 ascend310p5 ascend310p7 ascend310p3vir01 ascend310p3vir02 ascend310p3vir04 ascend310p3vir08)
set(ascend310b_list ascend310b1 ascend310b2 ascend310b3 ascend310b4)
set(ascend910_95_list ascend910_9599 ascend910_9589 ascend910_9579 ascend910_958b ascend910_957b ascend910_957d ascend910_950z ascend910_958a ascend910_957c
    ascend910_95a1 ascend910_95a2 ascend910_9591 ascend910_9592 ascend910_9595 ascend910_9596 ascend910_9581 ascend910_9582 ascend910_9583 ascend910_9584
    ascend910_9585 ascend910_9586 ascend910_9587 ascend910_9588 ascend910_9571 ascend910_9572 ascend910_9573 ascend910_9574 ascend910_9575 ascend910_9576
    ascend910_9577 ascend910_9578 ascend910_950x ascend910_950y)
set(ascend910_55_list ascend910_5591)
set(kirinx90_list kirinx90)
set(kirin9030_list kirin9030)
set(all_product ${ascend910b_list} ${ascend910_list} ${ascend310p_list} ${ascend910_95_list} ${ascend910_55_list} ${kirinx90_list} ${kirin9030_list})

if(NOT DEFINED SOC_VERSION)
    message(FATAL_ERROR "SOC_VERSION value not set.")
endif()

string(TOLOWER "${SOC_VERSION}" _LOWER_SOC_VERSION)

if(_LOWER_SOC_VERSION IN_LIST ascend910_95_list)
    set(DYNAMIC_MODE ON)
    set(BUILD_MODE   c310)
elseif(_LOWER_SOC_VERSION IN_LIST ascend910b_list)
    set(DYNAMIC_MODE ON)
    set(BUILD_MODE   c220)
elseif(_LOWER_SOC_VERSION IN_LIST ascend910_55_list)
    set(DYNAMIC_MODE ON)
    set(BUILD_MODE   310r6)
elseif(_LOWER_SOC_VERSION IN_LIST ascend910_list)
    set(BUILD_MODE   c100)
elseif(_LOWER_SOC_VERSION IN_LIST ascend310p_list)
    set(BUILD_MODE   m200)
elseif(_LOWER_SOC_VERSION IN_LIST ascend310b_list)
    set(BUILD_MODE   m300)
elseif(_LOWER_SOC_VERSION IN_LIST kirinx90_list)
    set(BUILD_MODE   l300)
elseif(_LOWER_SOC_VERSION IN_LIST kirin9030_list)
    set(BUILD_MODE   l311)
else()
    message(FATAL_ERROR "SOC_VERSION ${SOC_VERSION} does not support, the support list is ${all_product}")
endif()

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE "npu")
endif()

if(NOT DEFINED ASCEND_KERNEL_LAUNCH_ONLY)
    set(ASCEND_KERNEL_LAUNCH_ONLY OFF)
endif()

if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR "${ASCEND_CANN_PACKAGE_PATH} does not exist, please check the setting of ASCEND_CANN_PACKAGE_PATH.")
endif()

set(ASCEND_PYTHON_EXECUTABLE "python3" CACHE STRING "python executable program")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_LOWER_PROCESSOR)
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ccec_compiler AND SYSTEM_LOWER_PROCESSOR STREQUAL "x86_64")
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler AND SYSTEM_LOWER_PROCESSOR STREQUAL "aarch64")
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
else()
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit)
endif()

set(CCEC_PATH           ${ASCENDC_DEVKIT_PATH}/ccec_compiler/bin)
set(CCEC_LINKER        "${CCEC_PATH}/ld.lld")

set(ASCENDC_RUNTIME_OBJ_TARGET       ascendc_runtime_obj)
set(ASCENDC_RUNTIME_STATIC_TARGET    ascendc_runtime_static)
set(ASCENDC_RUNTIME_CONFIG           ascendc_runtime.cmake)
set(ASCENDC_PACK_KERNEL              ${ASCEND_CANN_PACKAGE_PATH}/bin/ascendc_pack_kernel)
set(ASCENDC_RUNTIME                  ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendc_runtime.a)

set(CMAKE_SKIP_RPATH TRUE)
include(ExternalProject)
