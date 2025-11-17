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
 * \file logical_xor_check.h
 * \brief
 */

#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_LOGICAL_XOR_LOGICAL_XOR_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_LOGICAL_XOR_LOGICAL_XOR_CHECK_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "logical_xor_check_c310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, typename U, bool isReuseSource>
__aicore__ inline void CheckFuncLogicalXor(__gm__ const char* name, const LocalTensor<T>& dst,
                                           const LocalTensor<U>& src0, const LocalTensor<U>& src1, const uint32_t count)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
    CheckFuncClassLogicalXor<T, U, isReuseSource> checkFun(name);
    checkFun.VerifyingParameters(dst, src0, src1, count);
#endif
}
} // namespace HighLevelApiCheck
} // namespace AscendC

#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_LOGICAL_XOR_LOGICAL_XOR_CHECK_H