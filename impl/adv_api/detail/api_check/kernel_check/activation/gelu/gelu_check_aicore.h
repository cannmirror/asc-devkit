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
 * \file gelu_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GELU_GELU_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GELU_GELU_CHECK_AICORE_H_

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool highPrecision = false, bool highPerformance = false>
class CheckFuncClassGelu {
public:
    __aicore__ inline CheckFuncClassGelu() {};
    __aicore__ inline CheckFuncClassGelu(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GELU_GELU_CHECK_AICORE_H_
 