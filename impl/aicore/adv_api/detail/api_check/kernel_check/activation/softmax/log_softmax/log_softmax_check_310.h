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
 * \file log_softmax_check_310.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_LOG_SOFTMAX_LOG_SOFTMAX_CHECK_310_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_LOG_SOFTMAX_LOG_SOFTMAX_CHECK_310_H

#include "kernel_tiling/kernel_tiling.h"
#include "activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
class CheckFuncClassLogSoftMax {
public:
    __aicore__ inline CheckFuncClassLogSoftMax(){};
    __aicore__ inline CheckFuncClassLogSoftMax(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const LocalTensor<T>& sumTensor,
        const LocalTensor<T>& maxTensor, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const LogSoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {}){};
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_LOG_SOFTMAX_LOG_SOFTMAX_CHECK_310_H
