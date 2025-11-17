/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
/* !
 * \file bitwise_xor_check.h
 * \brief
 */

#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_BITWISE_XOR_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_BITWISE_XOR_CHECK_H
#include "bitwise_xor_check_c310.h"
namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncBitwiseXor(__gm__ const char* apiName, const LocalTensor<T>& dst,
                                           const LocalTensor<T>& src0, const LocalTensor<T>& src1, const uint32_t count)
{
    CheckFuncClassBitwiseXor<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src0, src1, count);
}
} // namespace HighLevelApiCheck
} // namespace AscendC
#endif /* IMPL_API_CHECK_KERNEL_CHECK_MATH_BITWISE_XOR_CHECK_H */