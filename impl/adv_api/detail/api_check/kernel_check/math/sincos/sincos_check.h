/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sincos_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_SINCOS_SINCOS_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_SINCOS_SINCOS_CHECK_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "sincos_check_common.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncSinCos(__gm__ const char* name, const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, 
    const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
    CheckFuncClassSinCos<T, isReuseSource> checkFun(name);
    checkFun.VerifyingParameters(dst0, dst1, src, sharedTmpBuffer, count);
#endif
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_SINCOS_SINCOS_CHECK_H
