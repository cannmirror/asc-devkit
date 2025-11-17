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
 * \file log_softmax_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_LOG_SOFTMAX_LOG_SOFTMAX_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_LOG_SOFTMAX_LOG_SOFTMAX_CHECK_H_

#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/activation/softmax_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "log_softmax_check_common.h"
#else
#include "log_softmax_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void CheckFuncLogSoftMax(__gm__ const char *apiName, const LocalTensor<T>& dst, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LogSoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    CheckFuncClassLogSoftMax<T, isReuseSource, isDataFormatNZ> checkFun(apiName);
    checkFun.VerifyingParameters(dst, sumTensor, maxTensor, src, sharedTmpBuffer, tiling, softmaxShapeInfo);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_LOG_SOFTMAX_LOG_SOFTMAX_CHECK_H_
