/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file conv3d_config.h
 * \brief
 */

#ifndef API_CONV3D_CONFIG_H
#define API_CONV3D_CONFIG_H

#include "../../../../impl/adv_api/detail/conv/common/conv_forward_framework_util.h"
#include "../common/conv_forward_config.h"

namespace Conv3dApi {

enum class ConvL0PingPong : uint32_t {
    ALL_CLOSE = 0,
    L0A_OPEN,
    L0B_OPEN,
    ALL_OPEN
};

enum class ConvBL1ByPass : uint32_t {
    BYPASS_OFF = 0,
    BYPASS_ON = 1
};

enum class GroupConvType : uint32_t {
    NoGroup_Conv = 0,
	GroupConv_Weight_Gfz
};

struct Conv3dParam : public ConvApi::ConvParam {
    __aicore__ inline Conv3dParam(){};
};

template <class ConvDataType>
struct Conv3dCfg : public ConvApi::ConvConfig<ConvDataType> {
public:
    __aicore__ inline Conv3dCfg()
    {}

    using ContextData = struct _ : public ConvApi::ConvConfig<ConvDataType>::ContextData {
        __aicore__ inline _()
        {}
    };
};
}  // namespace Conv3dApi

#endif