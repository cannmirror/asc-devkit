/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "../../detail/host_log.h"
#include "include/adv_api/math/cos_tiling.h"
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
        ASCENDC_HOST_ASSERT(!isReuseSource,
            return 0, "when the input data type is half, isReuseSource is not supported");
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
        ASCENDC_HOST_ASSERT(!isReuseSource,
            return 0, "when the input data type is half, isReuseSource is not supported");
        calcPro = COS_HALF_CALC_FAC;
    }
    return calcPro * COS_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetCosMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t &maxValue, uint32_t &minValue)
{
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetCosMinTmpSize(typeSize, isReuseSource);
    maxValue = GetCosMaxTmpSize(inputSize, typeSize, isReuseSource);
}

void GetCosTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCount, uint32_t &extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? COS_FLOAT_NOREUSE_CALC_FAC : COS_HALF_CALC_FAC;
}
} // namespace AscendC
