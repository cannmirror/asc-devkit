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
 * \file power_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_POWER_POWER_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_POWER_POWER_CHECK_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "power_check_common.h"
#else
#include "power_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncPower(__gm__ const char *apiName, const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    CheckFuncClassPower<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncPower(__gm__ const char *apiName, const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    CheckFuncClassPower<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, src0Tensor, scalarValue, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncPower(__gm__ const char *apiName, const LocalTensor<T>& dstTensor, const T& scalarValue,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    CheckFuncClassPower<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, scalarValue, src1Tensor, sharedTmpBuffer, calCount);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_POWER_POWER_CHECK_H_
