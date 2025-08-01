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
 * \file acosh_tiling.cpp
 * \brief
 */
#include "math/acosh_tiling.h"

#include <cstdint>

#include "graph/tensor.h"
#include "detail/host_log.h"
namespace AscendC {
namespace {
constexpr uint32_t ACOSH_HALF_CALC_PROC = 2;
constexpr uint32_t ACOSH_FLOAT_CALC_PROC = 1;
constexpr uint32_t ACOSH_ONE_REPEAT_BYTE_SIZE = 256;

inline uint32_t GetAcoshMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    const uint32_t calcPro = (typeSize == sizeof(float)) ? ACOSH_FLOAT_CALC_PROC : ACOSH_HALF_CALC_PROC;
    return std::max(calcPro * inputSize, ACOSH_ONE_REPEAT_BYTE_SIZE) * typeSize;
}

inline uint32_t GetAcoshMinTmpSize(const uint32_t typeSize)
{
    const uint32_t calcPro = (typeSize == sizeof(float)) ? ACOSH_FLOAT_CALC_PROC : ACOSH_HALF_CALC_PROC;
    return calcPro * ACOSH_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetAcoshMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetAcoshMinTmpSize(typeSize);
    maxValue = GetAcoshMaxTmpSize(inputSize, typeSize);
}

void GetAcoshTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? ACOSH_FLOAT_CALC_PROC : ACOSH_HALF_CALC_PROC;
}
} // namespace AscendC
