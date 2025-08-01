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
 * \file swiglu_tiling.cpp
 * \brief
 */
#include "activation/swiglu_tiling.h"

#include <cstdint>
#include <set>

#include "graph/tensor.h"
#include "detail/host_log.h"
#include "detail/api_check/host_apicheck.h"

namespace AscendC {
namespace {
constexpr uint32_t SWIGLU_HALF_TMP_SIZE = 6;
constexpr uint32_t SWIGLU_FLOAT_TMP_SIZE = 0;
constexpr uint32_t SWIGLU_ONE_REPEAT_BYTE_SIZE = 256;
static constexpr uint32_t SWI_GELU_HALF_SIEZ = 2;
static constexpr uint32_t SWI_GELU_FLOAT_SIEZ = 4;
static const std::set<uint32_t> SUPPORT_TYPESIZE = {SWI_GELU_HALF_SIEZ, SWI_GELU_FLOAT_SIEZ};
static constexpr const char SWI_GELU_GET_MAX_MIN[] = "GetSwiGLUMaxMinTmpSize";
static constexpr const char SWI_GELU_GET_TMP_BUFFER[] = "GetSwiGLUTmpBufferFactorSize";

inline uint32_t GetSwiGLUMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    const uint32_t calcPro = (typeSize == sizeof(float)) ? SWIGLU_FLOAT_TMP_SIZE : SWIGLU_HALF_TMP_SIZE;
    return calcPro * std::max(inputSize * typeSize, SWIGLU_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetSwiGLUMinTmpSize(const uint32_t typeSize)
{
    const uint32_t calcPro = (typeSize == sizeof(float)) ? SWIGLU_FLOAT_TMP_SIZE : SWIGLU_HALF_TMP_SIZE;
    return calcPro * SWIGLU_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetSwiGLUMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue,
    const bool isReuseSource)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<SWI_GELU_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<SWI_GELU_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<SWI_GELU_GET_MAX_MIN>(isReuseSource);
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetSwiGLUMinTmpSize(typeSize);
    maxValue = GetSwiGLUMaxTmpSize(inputSize, typeSize);
}

void GetSwiGLUTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
{
    HighLevelApiCheck::TypeSizeVerifyingParameters<SWI_GELU_GET_TMP_BUFFER>(typeSize, SUPPORT_TYPESIZE);
    extraBuffer = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? SWIGLU_FLOAT_TMP_SIZE : SWIGLU_HALF_TMP_SIZE;
}
} // namespace AscendC
