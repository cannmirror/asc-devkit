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
 * \file layernorm_check_310.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORM_LAYERNORM_CHECK_310_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORM_LAYERNORM_CHECK_310_H

#include "kernel_tiling/kernel_tiling.h"
#include "normalization/layernorm_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassLayerNorm {
public:
    __aicore__ inline CheckFuncClassLayerNorm(){};
    __aicore__ inline CheckFuncClassLayerNorm(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
        const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma,
        const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon,
        LayerNormTiling& tiling){};
};

template <typename U, typename T, bool isReuseSource = false, const LayerNormConfig& config = LNCFG_NORM>
class CheckFuncClassLayerNormRstd {
public:
    __aicore__ inline CheckFuncClassLayerNormRstd(){};
    __aicore__ inline CheckFuncClassLayerNormRstd(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& output, const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputRstd, const LocalTensor<T>& inputX, const LocalTensor<U>& gamma,
        const LocalTensor<U>& beta, const float epsilon, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const LayerNormPara& para, const LayerNormSeparateTiling& tiling){};
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORM_LAYERNORM_CHECK_310_H
