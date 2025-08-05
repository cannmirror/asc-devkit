/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_COMPILE_INFO_H
#define ATVC_COMPILE_INFO_H
#include <map>
#include "tiling/platform/platform_ascendc.h"

namespace ATVC {
struct OpCompileInfo {
    uint64_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t cacheLineSize = 0;
    uint64_t ubBlockSize = 0;
};

inline OpCompileInfo GetOpCompileInfo()
{
    const auto& platformInfoMgr = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto soc = platformInfoMgr->GetSocVersion();
    static const std::map<platform_ascendc::SocVersion, OpCompileInfo> compileInfoMap = {
    {platform_ascendc::SocVersion::ASCEND910B, {48, 196608, 256, 32}},
    {platform_ascendc::SocVersion::ASCEND310B, {1, 262144, 256, 32}},
    {platform_ascendc::SocVersion::ASCEND310P, {8, 262144, 256, 32}},
    };
    auto findRes = compileInfoMap.find(soc);
    if (findRes != compileInfoMap.cend()) {
        return findRes->second;
    }
    printf("[ERROR] Current framework does not support this current device. Please check chip version.\n");
    return {0, 0, 0, 0};
}
}
#endif
