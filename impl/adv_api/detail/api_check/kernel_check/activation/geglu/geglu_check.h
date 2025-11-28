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
 * \file geglu_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GEGLU_GEGLU_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GEGLU_GEGLU_CHECK_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "geglu_check_common.h"
#else
#include "geglu_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncGeGLU(__gm__ const char *apiName, const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor0, const LocalTensor<T> &srcTensor1, const LocalTensor<uint8_t> &sharedTmpBuffer,
    uint32_t calCount)
{
    CheckFuncClassGeGLU<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GEGLU_GEGLU_CHECK_H_
