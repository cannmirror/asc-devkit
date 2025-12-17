/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
#include <cstdint>
#include <set>
#include <map>

#include "include/adv_api/sort/sort_tiling_intf.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "../../detail/host_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace AscendC {
namespace {
constexpr uint32_t SORT_HALF_SIZE = 2;
constexpr uint32_t SORT_FLOAT_SIZE = 4;

void CheckSortHostCommon(const char *apiName, const char *hostFuncName, 
    const platform_ascendc::PlatformAscendC &ascendcPlatform, const uint32_t elemCount, const uint32_t dataTypeSize)
{
    ASCENDC_HOST_ASSERT(elemCount > 0, return, 
        "[%s][%s] The elemCount must be greater than 0.", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(dataTypeSize == SORT_HALF_SIZE || dataTypeSize == SORT_FLOAT_SIZE, return, 
        "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, dataTypeSize);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ASCENDC_HOST_ASSERT(static_cast<uint64_t>(elemCount * dataTypeSize) < ubSize, continue,
        "[%s][%s] The size of srcShape is %luB, should be less than UB size.", apiName, hostFuncName,
        static_cast<uint64_t>(elemCount * dataTypeSize));
    return;
}
} // namespace

uint32_t GetConcatTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, const uint32_t elemCount,
    const uint32_t dataTypeSize)
{
    CheckSortHostCommon("Sort", "GetConcatTmpSize", ascendcPlatform, elemCount, dataTypeSize);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_95 ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_55) {
        return 0;
    } else {
        return elemCount * REGION_PROPOSAL_DATA_SIZE_V200 * dataTypeSize;
    }
}

uint32_t GetSortTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, const uint32_t elemCount,
    const uint32_t dataTypeSize)
{
    CheckSortHostCommon("Sort", "GetSortTmpSize", ascendcPlatform, elemCount, dataTypeSize);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_95 ||
        socVersion == platform_ascendc::SocVersion::ASCEND910_55) {
        if (dataTypeSize == sizeof(float)) {
            return elemCount * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 * dataTypeSize;
        } else {
            return elemCount * REGION_PROPOSAL_DATA_SIZE_HALF_V220 * dataTypeSize;
        }
    } else {
        return elemCount * REGION_PROPOSAL_DATA_SIZE_V200 * dataTypeSize;
    }
}
} // namespace AscendC
