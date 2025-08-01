/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_filter_config.h
 * \brief
 */

#ifndef AICORE_ADV_API_CONV_BACKPROP_CONV3D_BP_FILTER_CONFIG_H
#define AICORE_ADV_API_CONV_BACKPROP_CONV3D_BP_FILTER_CONFIG_H

#include "common/conv3d_bp_config_base.h"

namespace ConvBackpropApi {

template <class A, class B, class C, class D>
struct Conv3DBpFilterCfg : public ConvBpContext<A, B, C, D> {
public:
    __aicore__ inline Conv3DBpFilterCfg() {}

    using ContextData = struct _ : public ConvBpContext<A, B, C, D>::ContextData {
        __aicore__ inline _() {}
    };
};

} // namespace ConvBackpropApi
#endif // AICORE_ADV_API_CONV_BACKPROP_CONV3D_BP_FILTER_CONFIG_H
