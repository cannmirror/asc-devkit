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
 * \file sin_tiling_impl.cpp
 * \brief
 */
#include <cstdint>
#include <vector>
#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "include/adv_api/math/sin_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace AscendC {
namespace {
constexpr uint32_t SIN_HALF_CALC_FAC = 8;
constexpr uint32_t SIN_FLOAT_NOREUSE_CALC_FAC = 3;
constexpr uint32_t SIN_FLOAT_REUSE_CALC_FAC = 2;
constexpr uint32_t SIN_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t SIN_EXTRA_BUF = 32;
constexpr uint32_t SIN_DOUBLE = 2;
} // namespace

void GetSinTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? SIN_FLOAT_NOREUSE_CALC_FAC : SIN_HALF_CALC_FAC;
}

inline uint32_t GetSinMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t inputSize = 1;
    for (const auto dim : shapeDims) {
        inputSize *= dim;
    }
    ASCENDC_HOST_ASSERT(inputSize > 0, return 0, "Input Shape size must be greater than 0.");
    uint32_t calcFactor = 0;
    if (typeSize == sizeof(float)) {
        calcFactor = isReuseSource ? SIN_FLOAT_REUSE_CALC_FAC : SIN_FLOAT_NOREUSE_CALC_FAC;
    } else {
        ASCENDC_HOST_ASSERT(isReuseSource == false,
            return 0, "when the input data type is half, isReuseSource is not supported");
        calcFactor = SIN_HALF_CALC_FAC;
    }
    return inputSize * calcFactor * typeSize;
}

inline uint32_t GetSinMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    (void)(srcShape);
    (void)(isReuseSource);
    uint32_t calcFactor = 0;
    if (typeSize == sizeof(float)) {
        calcFactor = isReuseSource ? SIN_FLOAT_REUSE_CALC_FAC : SIN_FLOAT_NOREUSE_CALC_FAC;
    } else {
        ASCENDC_HOST_ASSERT(isReuseSource == false,
            return 0, "when the input data type is half, isReuseSource is not supported");
        calcFactor = SIN_HALF_CALC_FAC;
    }
    return calcFactor * SIN_ONE_REPEAT_BYTE_SIZE;
}

void GetSinMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    uint32_t max = GetSinMaxTmpSize(srcShape, typeSize, isReuseSource);
    minValue = GetSinMinTmpSize(srcShape, typeSize, isReuseSource);
    maxValue = std::max(max, minValue);
}
} // namespace AscendC