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
 * \file frac_tiling.cpp
 * \brief
 */
#include "math/frac_tiling.h"

#include <cstdint>

#include "graph/tensor.h"
#include "detail/host_log.h"
namespace AscendC {
namespace {
constexpr uint32_t FRAC_BASIC_BLK_SIZE = 2048;
constexpr uint32_t FRAC_HALF_CALC_FAC = 4;
constexpr uint32_t FRAC_FLOAT_CALC_FAC = 0;
constexpr uint32_t FRAC_ONE_REPEAT_BYTE_SIZE = 256;

inline uint32_t GetFracMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    const uint8_t calcPro = typeSize == sizeof(float) ? FRAC_FLOAT_CALC_FAC : FRAC_HALF_CALC_FAC;
    return calcPro * std::max(inputSize * typeSize, FRAC_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetFracMinTmpSize(const uint32_t typeSize)
{
    return typeSize == sizeof(float) ? FRAC_FLOAT_CALC_FAC * FRAC_ONE_REPEAT_BYTE_SIZE :
                                       FRAC_HALF_CALC_FAC * FRAC_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetFracMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetFracMinTmpSize(typeSize);
    maxValue = GetFracMaxTmpSize(inputSize, typeSize);
}

void GetFracTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? FRAC_FLOAT_CALC_FAC : FRAC_HALF_CALC_FAC;
}
} // namespace AscendC
