/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file digamma_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/math/digamma_tiling.h"
#include "../../detail/host_log.h"

namespace AscendC {
namespace {
constexpr uint32_t DIGAMMA_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t DIGAMMA_HALF_CALC_FAC = 8U * 2U;
constexpr uint32_t DIGAMMA_FLOAT_NOREUSE_CALC_PROC = 7U;
constexpr uint32_t DIGAMMA_FLOAT_REUSE_CALC_PROC = 6U;

inline uint32_t GetMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize, const bool isReuseSource)
{
    uint32_t calcPro = 0U;
    if (typeSize == sizeof(float)) {
        calcPro = isReuseSource ? DIGAMMA_FLOAT_REUSE_CALC_PROC : DIGAMMA_FLOAT_NOREUSE_CALC_PROC;
    } else {
        calcPro = DIGAMMA_HALF_CALC_FAC;
    }
    return calcPro * std::max(inputSize * typeSize, DIGAMMA_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetMinTmpSize(const uint32_t typeSize, const bool isReuseSource)
{
    if (typeSize == sizeof(float)) {
        return isReuseSource ? DIGAMMA_FLOAT_REUSE_CALC_PROC * DIGAMMA_ONE_REPEAT_BYTE_SIZE :
                               DIGAMMA_FLOAT_NOREUSE_CALC_PROC * DIGAMMA_ONE_REPEAT_BYTE_SIZE;
    } else {
        return DIGAMMA_HALF_CALC_FAC * DIGAMMA_ONE_REPEAT_BYTE_SIZE;
    }
}
}  // namespace

void GetDigammaTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCount, uint32_t &extraBuffer)
{
    extraBuffer = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? DIGAMMA_FLOAT_NOREUSE_CALC_PROC : DIGAMMA_HALF_CALC_FAC;
}

void GetDigammaMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isReuseSource,
                             uint32_t &maxValue, uint32_t &minValue)
{
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    maxValue = GetMaxTmpSize(inputSize, typeSize, isReuseSource);
    minValue = GetMinTmpSize(typeSize, isReuseSource);
}
}  // namespace AscendC
