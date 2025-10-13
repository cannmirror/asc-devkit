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
 * \file bitwise_and_tiling_impl.cpp
 * \brief
 */
#include <set>
#include "include/adv_api/math/bitwise_and_tiling.h"
#include "../../detail/host_log.h"
#include "../../detail/api_check/host_apicheck.h"
#include "graph/tensor.h"

namespace AscendC {
namespace {
static constexpr uint32_t BITWISE_AND_UINT8_SIZE = 1;
static constexpr uint32_t BITWISE_AND_UINT16_SIZE = 2;
static constexpr uint32_t BITWISE_AND_UINT32_SIZE = 4;
static constexpr uint32_t BITWISE_AND_UINT64_SIZE = 8;
static const std::set<uint32_t> SUPPORT_TYPESIZE = { BITWISE_AND_UINT8_SIZE, BITWISE_AND_UINT16_SIZE, BITWISE_AND_UINT32_SIZE, BITWISE_AND_UINT64_SIZE};
static constexpr const char BITWISE_AND_GET_MAX_MIN[] = "GetBitwiseAndMaxMinTmpSize";
static constexpr const char BITWISE_AND_GET_TMP_BUFFER[] = "GetBitwiseAndTmpBufferFactorSize";
} // namespace

void GetBitwiseAndMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape, 
    const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<BITWISE_AND_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<BITWISE_AND_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<BITWISE_AND_GET_MAX_MIN>(isReuseSource);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();

    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95 ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_55),
        return, "Unsupported SocVersion of BitwiseAnd API.");
    maxValue = 0u;
    minValue = 0u;
}

void GetBitwiseAndTmpBufferFactorSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t typeSize, 
    uint32_t& maxLivedNodeCount, uint32_t& extraBuf)
{
    HighLevelApiCheck::TypeSizeVerifyingParameters<BITWISE_AND_GET_TMP_BUFFER>(typeSize, SUPPORT_TYPESIZE);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95 ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_55),
        return, "Unsupported SocVersion of BitwiseAnd API.");
    extraBuf = 0u;
    maxLivedNodeCount = 0u;
}
} // namespace AscendC
