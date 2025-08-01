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
 * \file atan_tiling_impl.cpp
 * \brief
 */
#include <cstdint>
#include "math/atan_tiling.h"
#include "graph/tensor.h"
#include "detail/host_log.h"

namespace AscendC {
namespace {
constexpr uint32_t ATAN_HALF_CALC_FAC = 12;
constexpr uint32_t ATAN_FLOAT_CALC_FAC = 5;
constexpr uint32_t ATAN_ONE_REPEAT_BYTE_SIZE = 256;

inline uint32_t GetAtanMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    auto calcFactor = (typeSize == sizeof(float)) ? ATAN_FLOAT_CALC_FAC : ATAN_HALF_CALC_FAC;
    return calcFactor * std::max(inputSize * typeSize, ATAN_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetAtanMinTmpSize(const uint32_t typeSize)
{
    return typeSize == sizeof(float) ? ATAN_FLOAT_CALC_FAC * ATAN_ONE_REPEAT_BYTE_SIZE :
                                       ATAN_HALF_CALC_FAC * ATAN_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetAtanMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetAtanMinTmpSize(typeSize);
    maxValue = GetAtanMaxTmpSize(inputSize, typeSize);
}

void GetAtanTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? ATAN_FLOAT_CALC_FAC : ATAN_HALF_CALC_FAC;
}
} // namespace AscendC
