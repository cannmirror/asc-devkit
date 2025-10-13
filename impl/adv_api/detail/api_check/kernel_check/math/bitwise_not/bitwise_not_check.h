/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/* !
 * \file bitwise_not_check.h
 * \brief
 */

#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_BITWISE_NOT_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_BITWISE_NOT_CHECK_H
#include "bitwise_not_check_c310.h"
namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncBitwiseNot(__gm__ const char* apiName, const LocalTensor<T>& dst,
                                           const LocalTensor<T>& src, const uint32_t count)
{
    CheckFuncClassBitwiseNot<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src, count);
}
} // namespace HighLevelApiCheck
} // namespace AscendC
#endif /* IMPL_API_CHECK_KERNEL_CHECK_MATH_BITWISE_NOT_CHECK_H */