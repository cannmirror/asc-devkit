/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file bitwise_not_tiling_impl.cpp
 * \brief
 */
#include <set>
#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "../../detail/api_check/host_apicheck.h"
#include "include/adv_api/math/bitwise_not_tiling.h"

namespace AscendC {
namespace {
static const std::set<uint32_t> SUPPORT_TYPESIZE = {1, 2, 4, 8};
static constexpr const char BITWISE_NOT_GET_MAX_MIN[] = "GetBitwiseNotMaxMinTmpSize";
static constexpr const char BITWISE_NOT_GET_TMP_BUFFER[] = "GetBitwiseNotTmpBufferFactorSize";

uint32_t GetBitwiseNotMaxTmpSize(const platform_ascendc::SocVersion& socVersion, const uint32_t inputSize,
                                 const uint32_t typeSize)
{
    (void)socVersion;
    (void)typeSize;
    (void)inputSize;
    return 0;
}

uint32_t GetBitwiseNotMinTmpSize(const platform_ascendc::SocVersion& socVersion, const uint32_t typeSize)
{
    (void)socVersion;
    (void)typeSize;
    return 0;
}
} // namespace
void GetBitwiseNotMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape,
                                const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue,
                                uint32_t& minValue)
{
    (void)isReuseSource;

    const uint32_t inputSize = srcShape.GetShapeSize();
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95
                         || socVersion == platform_ascendc::SocVersion::ASCEND910_55),
                        return, "Unsupported SocVersion of BitwiseNot API.");

    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<BITWISE_NOT_GET_MAX_MIN>(inputSize, typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<BITWISE_NOT_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<BITWISE_NOT_GET_MAX_MIN>(isReuseSource);

    maxValue = GetBitwiseNotMaxTmpSize(socVersion, inputSize, typeSize);
    minValue = GetBitwiseNotMinTmpSize(socVersion, typeSize);
}

void GetBitwiseNotTmpBufferFactorSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t typeSize,
                                      uint32_t& maxLivedNodeCount, uint32_t& extraBuf)
{
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95
                         || socVersion == platform_ascendc::SocVersion::ASCEND910_55),
                        return, "Unsupported SocVersion of BitwiseNot API.");
    HighLevelApiCheck::TypeSizeVerifyingParameters<BITWISE_NOT_GET_TMP_BUFFER>(typeSize, SUPPORT_TYPESIZE);

    extraBuf = 0u;
    maxLivedNodeCount = 0u;
}

} // namespace AscendC