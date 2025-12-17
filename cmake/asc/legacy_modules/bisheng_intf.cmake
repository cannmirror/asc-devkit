# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
add_library(device_intf_pub INTERFACE)

target_compile_options(device_intf_pub INTERFACE
    -O3
    -std=c++17
    "SHELL:--cce-aicore-lang"
)

set(CANN_VERSION_HEADER ${ASCENDC_DEVKIT_PATH}/../include/ascendc/asc_devkit_version.h)
if(EXISTS ${CANN_VERSION_HEADER})
    target_compile_options(device_intf_pub INTERFACE
        "SHELL:-include ${CANN_VERSION_HEADER}"
    )
endif()

target_compile_definitions(device_intf_pub INTERFACE
    TILING_KEY_VAR=0
)

target_include_directories(device_intf_pub INTERFACE
    ${ASCENDC_DEVKIT_PATH}/asc/impl/adv_api
    ${ASCENDC_DEVKIT_PATH}/asc/impl/basic_api
    ${ASCENDC_DEVKIT_PATH}/asc/impl/utils
    ${ASCENDC_DEVKIT_PATH}/asc/include
    ${ASCENDC_DEVKIT_PATH}/asc/include/adv_api
    ${ASCENDC_DEVKIT_PATH}/asc/include/basic_api
    ${ASCENDC_DEVKIT_PATH}/asc/include/aicpu_api
    ${ASCENDC_DEVKIT_PATH}/asc/include/utils  
    ${ASCENDC_DEVKIT_PATH}/tikcpp/tikcfw
    ${ASCENDC_DEVKIT_PATH}/tikcpp/tikcfw/interface
    ${ASCENDC_DEVKIT_PATH}/tikcpp/tikcfw/impl
)

add_library(m300_intf_pub INTERFACE)

target_compile_options(m300_intf_pub INTERFACE
    --cce-aicore-arch=dav-m300
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-or-combine=false"
    "SHELL:-mllvm -instcombine-code-sinking=false"
    "SHELL:-mllvm -cce-aicore-jump-expand=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
)

target_link_libraries(m300_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(c220_aiv_intf_pub INTERFACE)

target_compile_options(c220_aiv_intf_pub INTERFACE
    --cce-aicore-arch=dav-c220-vec
    --cce-aicore-only
    --cce-auto-sync
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

target_link_libraries(c220_aiv_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(c220_aic_intf_pub INTERFACE)

target_compile_options(c220_aic_intf_pub INTERFACE
    --cce-aicore-arch=dav-c220-cube
    --cce-aicore-only
    --cce-auto-sync
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

target_link_libraries(c220_aic_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(m200_intf_pub INTERFACE)

target_compile_options(m200_intf_pub INTERFACE
    --cce-aicore-arch=dav-m200
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-fp-ceiling=2"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
)

target_link_libraries(m200_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(m200_vec_intf_pub INTERFACE)

target_compile_options(m200_vec_intf_pub INTERFACE
    --cce-aicore-arch=dav-m200-vec
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-fp-ceiling=2"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
    "SHELL:-D__ENABLE_VECTOR_CORE__"
)

target_link_libraries(m200_vec_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(c100_intf_pub INTERFACE)

target_compile_options(c100_intf_pub INTERFACE
    --cce-aicore-arch=dav-c100
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-jump-expand=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
)

target_link_libraries(c100_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)
