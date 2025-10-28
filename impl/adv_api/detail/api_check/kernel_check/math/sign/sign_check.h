/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sign_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_SIGN_SIGN_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_SIGN_SIGN_CHECK_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "sign_check_common.h"
#else
#include "sign_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncSign(__gm__ const char *apiName, const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckFuncClassSign<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_SIGN_SIGN_CHECK_H_
