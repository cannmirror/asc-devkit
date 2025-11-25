# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
get_filename_component(ASCENDC_KERNEL_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/legacy_modules" ABSOLUTE)

include(${ASCENDC_KERNEL_CMAKE_DIR}/host_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/host_intf.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/function.cmake)

# If want to launch ACLRT_LAUNCH_KERNEL with ASC plugin, replace function in function.cmake with FindASC.cmake
# _LOWER_SOC_VERSION comes from host_config.cmake
if((_LOWER_SOC_VERSION IN_LIST ascend910b_list) OR (_LOWER_SOC_VERSION IN_LIST ascend310p_list))
    if (ENABLE_ASC)
        option(ENABLE_ACLRT_LAUNCH "Enable ACLRT_LAUNCH_KERNEL for compatibility" ON)
        # plugin support ASC language
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/ASC_CMake")
        include(${CMAKE_CURRENT_LIST_DIR}/ASC_CMake/FindASC.cmake)

        find_package(ASC REQUIRED)
        enable_language(ASC)
    endif()
else()
    message(STATUS "SOC_VERSION ${SOC_VERSION} does not support ASC, use KernelLaunch.")
endif()