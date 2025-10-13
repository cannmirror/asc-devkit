
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
 * \file reduce_mean_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "reduce_mean_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "reduce_mean_check_c310.h"
#else
#include "reduce_mean_check_aicore.h"
#endif

namespace AscendC {  
namespace HighLevelApiCheck {
template <typename T, class pattern>
__aicore__ inline void CheckFuncReduceMean(__gm__ const char *apiName, const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], const bool srcInnerPad, const uint32_t padLast)
{
    CheckFuncClassReduceMean<T, pattern> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, padLast);
}

} // namespace HighLevelApiCheck
} // AscendC
#endif // IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_H_
 
