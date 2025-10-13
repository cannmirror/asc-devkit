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
 * \file floor_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_FLOOR_FLOOR_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_FLOOR_FLOOR_CHECK_AICORE_H_

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassFloor  {
public:
    __aicore__ inline CheckFuncClassFloor() {};
    __aicore__ inline CheckFuncClassFloor(__gm__ const char *apiName)  {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount) {};
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassFloorNoTmpBuffer {
public:
    __aicore__ inline CheckFuncClassFloorNoTmpBuffer() {};
    __aicore__ inline CheckFuncClassFloorNoTmpBuffer(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const uint32_t calCount) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_FLOOR_FLOOR_CHECK_AICORE_H_
 