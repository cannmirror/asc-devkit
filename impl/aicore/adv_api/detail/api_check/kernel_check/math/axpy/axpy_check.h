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
 * \file axpy_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_AXPY_AXPY_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_AXPY_AXPY_CHECK_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "axpy_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "axpy_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, typename U, bool isReuseSource>
__aicore__ inline void CheckFuncAxpy(__gm__ const char* apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<U>& srcTensor, const U scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t calCount)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassAxpy<T, U, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, scalarValue, sharedTmpBuffer, calCount);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_AXPY_AXPY_CHECK_H
