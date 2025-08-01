/* *
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file fmod_tiling_impl.cpp
 * \brief
 */
#include "math/fmod_tiling.h"
#include "graph/tensor.h"
#include "detail/host_log.h"
#include "math/trunc_tiling.h"

namespace AscendC {
namespace {
constexpr uint32_t FMOD_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t FMOD_HALF_TENSOR_NUM = 3 * 2; // need extra 3 times of typeSize*2 float
constexpr uint32_t FMOD_FLOAT_TENSOR_NUM = 0;

inline uint32_t GetFmodMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize, const uint8_t truncCalPro)
{
    const uint8_t fmodExtPro = typeSize == sizeof(float) ? FMOD_FLOAT_TENSOR_NUM : FMOD_HALF_TENSOR_NUM;
    return (truncCalPro + fmodExtPro) * std::max(inputSize * typeSize, FMOD_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetFmodMinTmpSize(const uint32_t typeSize, const uint8_t truncCalPro)
{
    const uint8_t fmodExtPro = typeSize == sizeof(float) ? FMOD_FLOAT_TENSOR_NUM : FMOD_HALF_TENSOR_NUM;
    return (truncCalPro + fmodExtPro) * FMOD_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetFmodMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    // Calculate what is actually needed for trunc.
    GetTruncMaxMinTmpSize(srcShape, typeSize, isReuseSource, maxValue, minValue);
    const uint8_t truncCalPro = maxValue / std::max(inputSize * typeSize, FMOD_ONE_REPEAT_BYTE_SIZE);

    maxValue = GetFmodMaxTmpSize(inputSize, typeSize, truncCalPro);
    minValue = GetFmodMinTmpSize(typeSize, truncCalPro);
}

void GetFmodTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    GetTruncTmpBufferFactorSize(typeSize, maxLiveNodeCount, extraBuf);
    maxLiveNodeCount += (typeSize == sizeof(float)) ? FMOD_FLOAT_TENSOR_NUM : FMOD_HALF_TENSOR_NUM;
}
} // namespace AscendC
