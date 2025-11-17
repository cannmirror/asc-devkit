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
 * \file xor_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_XOR_XOR_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_XOR_XOR_CHECK_AICORE_H_

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassXor  {
public:
    __aicore__ inline CheckFuncClassXor() {};
    __aicore__ inline CheckFuncClassXor(__gm__ const char *apiName)  {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T> &src0Tensor,
        const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_XOR_XOR_CHECK_AICORE_H_
