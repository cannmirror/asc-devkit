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
 * \file sinh_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_SINH_SINH_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_SINH_SINH_CHECK_COMMON_H_

#include "../math_common_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassSinh : public CheckFuncClassMathCommon {
public:
    __aicore__ inline CheckFuncClassSinh() {};
    __aicore__ inline CheckFuncClassSinh(__gm__ const char *apiName) : CheckFuncClassMathCommon(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount) {
        CheckFuncClassMathCommon::CommonVerifyingParameters<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_SINH_SINH_CHECK_COMMON_H_
