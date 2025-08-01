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
 * \file conv3d_bp_filter_api.h
 * \brief
 */

#ifndef AICORE_ADV_API_CONV_BACKPROP_CONV3D_BP_FILTER_API_H
#define AICORE_ADV_API_CONV_BACKPROP_CONV3D_BP_FILTER_API_H

#include "conv3d_bp_filter_intf.h"
#include "conv3d_bp_filter_config.h"
#include "conv_backprop/conv3d_backprop_filter/conv3d_bp_filter_impl.h"

namespace ConvBackpropApi {
template <class INPUT_TYPE, class WEIGHT_TYPE, class GRAD_OUTPUT_TYPE, class GRAD_WEIGHT_TYPE>
using Conv3DBackpropFilter =
    Conv3DBpFilterIntf<Conv3DBpFilterCfg<INPUT_TYPE, WEIGHT_TYPE, GRAD_OUTPUT_TYPE, GRAD_WEIGHT_TYPE>,
        Conv3DBpFilterImpl>;

} // namespace ConvBackpropApi
#endif // AICORE_ADV_API_CONV_BACKPROP_CONV3D_BP_FILTER_API_H
