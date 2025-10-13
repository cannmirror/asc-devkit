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
 * \file reglu_tiling_impl.cpp
 * \brief
 */

#include <set>

#include "include/adv_api/activation/reglu_tiling.h"
#include "../../detail/host_log.h"
#include "../../detail/api_check/host_apicheck.h"

namespace AscendC {
namespace {
constexpr uint32_t REGLU_HALF_OR_BFLOAT16_CALC_FAC = 6;
constexpr uint32_t REGLU_ONE_REPEAT_BYTE_SIZE = 256;
static constexpr uint32_t RE_GELU_HALF_SIEZ = 2;
static constexpr uint32_t RE_GELU_FLOAT_SIEZ = 4;
static const std::set<uint32_t> SUPPORT_TYPESIZE = { RE_GELU_HALF_SIEZ, RE_GELU_FLOAT_SIEZ };
static constexpr const char RE_GELU_GET_MAX_MIN[] = "GetReGluMaxMinTmpSize";

uint32_t GetReGluMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t inputSize = 1;
    for (const auto dim : shapeDims) {
        inputSize *= dim;
    }
    ASCENDC_HOST_ASSERT(inputSize > 0, return 0, "Input Shape size must be greater than 0.");
    if (typeSize == sizeof(float)) {
        return REGLU_ONE_REPEAT_BYTE_SIZE;
    }
    return inputSize * REGLU_HALF_OR_BFLOAT16_CALC_FAC * typeSize;
}

uint32_t GetReGluMinTmpSize(const uint32_t typeSize)
{
    if (typeSize == sizeof(float)) {
        return REGLU_ONE_REPEAT_BYTE_SIZE;
    }
    return REGLU_HALF_OR_BFLOAT16_CALC_FAC * REGLU_ONE_REPEAT_BYTE_SIZE;
}
}  // namespace

void GetReGluMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<RE_GELU_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<RE_GELU_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<RE_GELU_GET_MAX_MIN>(isReuseSource);
    ASCENDC_HOST_ASSERT(isReuseSource == false, return, "multiplexing of input memory is not supported.");
    uint32_t max = GetReGluMaxTmpSize(srcShape, typeSize);
    minValue = GetReGluMinTmpSize(typeSize);
    maxValue = std::max(max, minValue);
}
}  // namespace AscendC
