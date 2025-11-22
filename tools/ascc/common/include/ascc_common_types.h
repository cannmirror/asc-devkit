/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ascc_common_types.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_COMMON_TYPES_H__
#define __INCLUDE_ASCC_COMMON_TYPES_H__

#include <string>
#include <cstdint>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

namespace Ascc {

// used for return code in main
constexpr uint32_t ASCC_SUCCESS = 0;
constexpr uint32_t ASCC_FAILURE = 1;

// used for command in ascendc_pack_kernel
constexpr uint8_t ASCC_CORE_TYPE_MIX = 0;
constexpr uint8_t ASCC_CORE_TYPE_AIV = 1;
constexpr uint8_t ASCC_CORE_TYPE_AIC = 2;

// used for return code in normal function
enum class AsccStatus : uint32_t {
    SUCCESS = 0,
    FAILURE = 1
};

enum class PathStatus : uint32_t {
    EXIST = 0,
    READ,
    WRITE,
    NOT_EXIST
};

// preprocess task for ascendfe
enum class PreTaskType : uint32_t {
    NONE = 0,
    HOST,
    DEVICE_AIC,
    DEVICE_AIV
};

enum class OutputFileType: uint32_t {
    FILE_EXECUTABLE = 0,      // default executable
    FILE_O = 1,               // .o when -c
    FILE_SO = 2,              // .so when -shared
    UNKNOWN = 3
};

enum class CoreType: uint32_t {
    SPLIT_VEC = 0,            // dav_c220
    SPLIT_CUBE,               // dav_c220
    AICORE,                   // dav_m200
    VEC,                      // dav_m200
    UNKNOWN
};

enum class CodeMode : uint32_t {
    KERNEL_TYPE_AIV_ONLY = 0,
    KERNEL_TYPE_AIC_ONLY = 1,
    KERNEL_TYPE_MIX_AIV_1_0 = 2,
    KERNEL_TYPE_MIX_AIC_1_0 = 3,
    KERNEL_TYPE_MIX_AIC_1_1 = 4,
    KERNEL_TYPE_MIX_AIC_1_2 = 5,
    INVALID_TYPE = 0xffffffff
};

inline const std::unordered_map<CoreType, std::string> CORE_TYPE_STR_MAP = {
    {CoreType::SPLIT_VEC, "vec"},
    {CoreType::SPLIT_CUBE, "cube"},
    {CoreType::AICORE, "aicore"},
    {CoreType::VEC, "vec"}
};

inline const std::unordered_map<CodeMode, std::vector<const char*>> KERNEL_TYPE_TO_FILE_NAME = {
    {CodeMode::KERNEL_TYPE_AIC_ONLY, {"device_stub_aic.cpp"}},
    {CodeMode::KERNEL_TYPE_AIV_ONLY, {"device_stub_aiv.cpp"}},
    {CodeMode::KERNEL_TYPE_MIX_AIV_1_0, {"device_stub_mix_aiv.cpp"}},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_0, {"device_stub_mix_aic.cpp"}},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_1, {"device_stub_mix_1_1_aic.cpp", "device_stub_mix_1_1_aiv.cpp"}},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_2, {"device_stub_mix_aic.cpp", "device_stub_mix_aiv.cpp"}}
};

// supported soc for bishengcc
const std::unordered_set<std::string> SUPPORTED_SOC_LIST = {
    "Ascend910B1",
    "Ascend910B2",
    "Ascend910B2C",
    "Ascend910B3",
    "Ascend910B4",
    "Ascend910B4-1",
    "Ascend910_9391",
    "Ascend910_9381",
    "Ascend910_9372",
    "Ascend910_9392",
    "Ascend910_9382",
    "Ascend910_9362"
};

}

#endif // __INCLUDE_ASCC_COMMON_TYPES_H__