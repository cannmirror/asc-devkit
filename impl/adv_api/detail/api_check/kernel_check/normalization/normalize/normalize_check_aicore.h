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
 * \file normalize_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_NORMALIZE_NORMALIZE_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_NORMALIZE_NORMALIZE_CHECK_AICORE_H_

#include "include/adv_api/normalization/normalize_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename U, typename T, bool isReuseSource, const NormalizeConfig& config>
class CheckFuncClassNormalize {
public:
    __aicore__ inline CheckFuncClassNormalize() {};
    __aicore__ inline CheckFuncClassNormalize(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
        const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
        const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const float epsilon, const NormalizePara& para) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_NORMALIZE_NORMALIZE_CHECK_AICORE_H_
