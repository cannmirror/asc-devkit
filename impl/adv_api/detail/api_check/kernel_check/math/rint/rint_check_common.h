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
 * \file rint_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_RINT_RINT_CHECK_COMMON_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_RINT_RINT_CHECK_COMMON_H

#include "../math_common_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassRint : public CheckFuncClassMathCommon {
public:
    __aicore__ inline CheckFuncClassRint() {};
    __aicore__ inline CheckFuncClassRint(__gm__ const char* name) : CheckFuncClassMathCommon(name) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const LocalTensor<T>& src,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count) {
        CheckFuncClassMathCommon::CommonVerifyingParameters<T, isReuseSource>(dst, src, sharedTmpBuffer, count);
    };
};
}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_RINT_RINT_CHECK_COMMON_H
