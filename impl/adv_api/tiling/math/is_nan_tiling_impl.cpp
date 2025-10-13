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
 * \file is_nan_tiling_impl.cpp
 * \brief
 */
#include <set>
#include "include/adv_api/math/is_nan_tiling.h"
#include "../../detail/host_log.h"
#include "../../detail/api_check/host_apicheck.h"
#include "graph/tensor.h"

namespace AscendC {
namespace {
static constexpr uint32_t ISNAN_HALF_SIZE = 2;
static constexpr uint32_t ISNAN_FLOAT_SIZE = 4;
static const std::set<uint32_t> SUPPORT_TYPESIZE = { ISNAN_HALF_SIZE, ISNAN_FLOAT_SIZE };
static constexpr const char ISNAN_GET_MAX_MIN[] = "GetIsNanMaxMinTmpSize";
static constexpr const char ISNAN_GET_TMP_BUFFER[] = "GetIsNanTmpBufferFactorSize";
} // namespace

void GetIsNanMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape, 
    const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<ISNAN_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<ISNAN_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<ISNAN_GET_MAX_MIN>(isReuseSource);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();

    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95 ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_55 ||
        socVersion == platform_ascendc::SocVersion::MC62CM12A),
        return, "Unsupported SocVersion of IsNan API.");
    maxValue = 0u;
    minValue = 0u;
}

void GetIsNanTmpBufferFactorSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t typeSize, 
    uint32_t& maxLivedNodeCount, uint32_t& extraBuf)
{
    HighLevelApiCheck::TypeSizeVerifyingParameters<ISNAN_GET_TMP_BUFFER>(typeSize, SUPPORT_TYPESIZE);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    ASCENDC_HOST_ASSERT((socVersion == platform_ascendc::SocVersion::ASCEND910_95 ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_55 ||
        socVersion == platform_ascendc::SocVersion::MC62CM12A),
        return, "Unsupported SocVersion of IsNan API.");
    extraBuf = 0u;
    maxLivedNodeCount = 0u;
}
} // namespace AscendC