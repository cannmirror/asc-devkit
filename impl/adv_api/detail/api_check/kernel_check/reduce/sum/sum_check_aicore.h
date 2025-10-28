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
* \file sum_check_aicore.h
* \brief
*/
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_SUM_SUM_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_SUM_SUM_CHECK_AICORE_H_

namespace AscendC {  
namespace HighLevelApiCheck {
template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
class CheckFuncClassSum {
public:
    __aicore__ inline CheckFuncClassSum() {};
    __aicore__ inline CheckFuncClassSum(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const SumParams &sumParams) {};
};

} // namespace HighLevelApiCheck
} // AscendC
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_SUM_SUM_CHECK_AICORE_H_
