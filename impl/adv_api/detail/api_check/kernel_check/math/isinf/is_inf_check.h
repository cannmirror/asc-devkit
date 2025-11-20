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
 * \file isinf_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_ISINF_ISINF_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_ISINF_ISINF_CHECK_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102)
#include "is_inf_check_common.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, typename U, bool isReuseSource>
__aicore__ inline void CheckFuncIsInf(__gm__ const char* name, const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102)
    CheckFuncClassIsInf<T, U, isReuseSource> checkFun(name);
    checkFun.VerifyingParameters(dst, src, sharedTmpBuffer, count);
#endif
}
}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_ISINF_ISINF_CHECK_H
