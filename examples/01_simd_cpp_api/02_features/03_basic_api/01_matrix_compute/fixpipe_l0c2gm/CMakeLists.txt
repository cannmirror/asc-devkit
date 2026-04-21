# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

if(DEFINED RUN_MODE)
    set(CMAKE_ASC_RUN_MODE ${RUN_MODE} CACHE STRING "Alias of RUN_MODE")
endif()
if(DEFINED NPU_ARCH)
    set(CMAKE_ASC_ARCHITECTURES ${NPU_ARCH} CACHE STRING "Alias of NPU_ARCH")
endif()

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    fixpipe_l0c2gm.asc
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    m
)

# 场景参数（通过命令行传入）
# SCENARIO_NUM: 场景编号
#   1 - 输出格式NZ，输出数据类型float
#   2 - 输出格式ND，输出数据类型float
#   3 - 输出格式DN，输出数据类型float
#   4 - 输出格式ND，输出数据类型int8_t，使能Scalar量化
#   5 - 输出格式ND，输出数据类型int8_t，使能Vector量化
#   6 - 输出格式ND，输出数据类型float，使能relu
#   7 - 输出格式NZ，输出数据类型float，使能channel split
set(SCENARIO_NUM "1" CACHE STRING "Scene number (1-7)")

add_definitions(-DSCENARIO_NUM=${SCENARIO_NUM})

set(NPU_ARCH "dav-2201" CACHE STRING "NPU ARCH, e.g. dav-2201 dav-3510")
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)