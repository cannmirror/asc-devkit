/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cos_tiling_impl.cpp
 * \brief
 */
#include <cstdint>

#include "graph/tensor.h"
#include "detail/host_log.h"
#include "math/cos_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace AscendC {
namespace {
constexpr uint32_t COS_HALF_CALC_FAC = 8;
constexpr uint32_t COS_FLOAT_NOREUSE_CALC_FAC = 3;
constexpr uint32_t COS_FLOAT_REUSE_CALC_FAC = 2;
constexpr uint32_t COS_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t COS_EXTRA_BUF = 32;
constexpr uint32_t COS_DOUBLE = 2;

inline uint32_t GetCosMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize, const bool isReuseSource)
{
    uint8_t calcPro = 0;
    if (typeSize == sizeof(float)) {
        calcPro = isReuseSource ? COS_FLOAT_REUSE_CALC_FAC : COS_FLOAT_NOREUSE_CALC_FAC;
    } else {
        ASCENDC_HOST_ASSERT(!isReuseSource, return 0, "when the input data type is half, isReuseSource is not support");
        calcPro = COS_HALF_CALC_FAC;
    }
    return calcPro * std::max(inputSize * typeSize, COS_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetCosMinTmpSize(const uint32_t typeSize, const bool isReuseSource)
{
    uint8_t calcPro = 0;
    if (typeSize == sizeof(float)) {
        calcPro = isReuseSource ? COS_FLOAT_REUSE_CALC_FAC : COS_FLOAT_NOREUSE_CALC_FAC;
    } else {
        ASCENDC_HOST_ASSERT(!isReuseSource, return 0, "when the input data type is half, isReuseSource is not support");
        calcPro = COS_HALF_CALC_FAC;
    }
    return calcPro * COS_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetCosMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetCosMinTmpSize(typeSize, isReuseSource);
    maxValue = GetCosMaxTmpSize(inputSize, typeSize, isReuseSource);
}

void GetCosTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? COS_FLOAT_NOREUSE_CALC_FAC : COS_HALF_CALC_FAC;
}

void GetCosMaxMinTmpSize(const CosConfig& config, const ge::Shape& srcShape, const uint32_t typeSize,
    const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
{
    (void)typeSize;
    (void)isReuseSource;
    platform_ascendc::PlatformAscendC* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ASCENDC_HOST_ASSERT((platform != nullptr), return, "Failed to get PlatformAscendC");

    platform_ascendc::SocVersion socVersion = platform->GetSocVersion();
    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95
                            || socVersion == platform_ascendc::SocVersion::ASCEND910_55),
        return, "Unsupported SocVersion of Cos API.");

    if (config.algo == CosAlgo::POLYNOMIAL_APPROXIMATION) {
        maxValue = 0;
        minValue = 0;
    } else if (config.algo == CosAlgo::RADIAN_REDUCTION) {
        std::vector<int64_t> shapeDims = srcShape.GetDims();
        uint32_t inputSize = 1;
        for (const auto dim : shapeDims) { inputSize *= dim; }
        maxValue = sizeof(float) * inputSize * COS_DOUBLE + COS_EXTRA_BUF;
        minValue = sizeof(float) * inputSize * COS_DOUBLE + COS_EXTRA_BUF;
    }
}

void GetCosTmpBufferFactorSize(
    const CosConfig& config, const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    platform_ascendc::PlatformAscendC* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ASCENDC_HOST_ASSERT((platform != nullptr), return, "Failed to get PlatformAscendC");

    platform_ascendc::SocVersion socVersion = platform->GetSocVersion();
    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95
                            || socVersion == platform_ascendc::SocVersion::ASCEND910_55),
        return, "Unsupported SocVersion of Cos API.");
    if (config.algo == CosAlgo::POLYNOMIAL_APPROXIMATION) {
        extraBuf = 0;
        maxLiveNodeCount = 0;
    } else if (config.algo == CosAlgo::RADIAN_REDUCTION) {
        if (typeSize == sizeof(float)) {
            extraBuf = COS_EXTRA_BUF;
            maxLiveNodeCount = COS_DOUBLE;
        } else {
            extraBuf = 0;
            maxLiveNodeCount = COS_DOUBLE * COS_DOUBLE;
        }
    }
}
} // namespace AscendC
