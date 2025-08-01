/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_tiling.h
 * \brief
 */

#ifndef AICORE_ADV_API_CONV_CONV3D_CONV3D_TILING_H
#define AICORE_ADV_API_CONV_CONV3D_CONV3D_TILING_H

#include "conv3d_tiling_base.h"
#include "conv3d_tilingdata.h"

namespace Conv3dTilingApi {
class Conv3dTiling : public Conv3dTilingBase {
public:
    explicit Conv3dTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform) :
        Conv3dTilingBase(ascendcPlatform){};
    explicit Conv3dTiling(const PlatformInfo& platform) : Conv3dTilingBase(platform){};
    ~Conv3dTiling() override = default;
    int64_t GetTiling(optiling::TConv3DApiTiling& tiling) override;

protected:
    int64_t Compute() override;
};
} // namespace Conv3dTilingApi

#endif
