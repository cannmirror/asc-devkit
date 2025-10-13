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
 * \file welfordfinalize_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_AICORE_H_

#include "include/adv_api/normalization/welfordfinalize_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <bool isReuseSource = false>
class CheckFuncClassWelfordFinalizeCounts {
public:
    __aicore__ inline CheckFuncClassWelfordFinalizeCounts() {};
    __aicore__ inline CheckFuncClassWelfordFinalizeCounts(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
        const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance, const LocalTensor<int32_t> &counts,
        const LocalTensor<uint8_t> &sharedTmpBuffer, const WelfordFinalizePara &para) {};
};

template <bool isReuseSource = false>
class CheckFuncClassWelfordFinalize {
public:
    __aicore__ inline CheckFuncClassWelfordFinalize() {};
    __aicore__ inline CheckFuncClassWelfordFinalize(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
        const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
        const LocalTensor<uint8_t> &sharedTmpBuffer, const WelfordFinalizePara &para) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_AICORE_H_
