/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clamp_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_CLAMP_CLAMP_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_CLAMP_CLAMP_CHECK_COMMON_H

#include "../math_common_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassClampMax : public CheckFuncClassMathCommon {
public:
    __aicore__ inline CheckFuncClassClampMax(){};
    __aicore__ inline CheckFuncClassClampMax(__gm__ const char* apiName) : CheckFuncClassMathCommon(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
    {
        CheckFuncClassMathCommon::CommonVerifyingParameters<T, isReuseSource>(
            dstTensor, srcTensor, sharedTmpBuffer, calCount);
    };
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassClampMin : public CheckFuncClassMathCommon {
public:
    __aicore__ inline CheckFuncClassClampMin(){};
    __aicore__ inline CheckFuncClassClampMin(__gm__ const char* apiName) : CheckFuncClassMathCommon(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
    {
        CheckFuncClassMathCommon::CommonVerifyingParameters<T, isReuseSource>(
            dstTensor, srcTensor, sharedTmpBuffer, calCount);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_CLAMP_CLAMP_CHECK_COMMON_H
