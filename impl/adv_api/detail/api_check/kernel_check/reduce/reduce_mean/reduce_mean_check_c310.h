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
 * \file reduce_mean_check
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_C310_H_
#define IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_C310_H_

#include "../reduce_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, class pattern>
class CheckFuncClassReduceMean : public CheckFuncClassReduce<T, pattern> {
public:
    __aicore__ inline CheckFuncClassReduceMean(__gm__ const char *name) : CheckFuncClassReduce<T, pattern>(name) {};

    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], const bool srcInnerPad, const uint32_t padLast) {
        CheckFuncClassReduce<T, pattern>::VerifyingParameters(dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, padLast);

        ASCENDC_ASSERT((srcShape[1] * sizeof(T) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[ReduceMean] The result of srcShape[1] * sizeof(T) cannot be %u, "
            "should be an integer multiple of 32.", srcShape[1] * sizeof(T)); });
    };
};
} // namespace HighLevelApiCheck
} // AscendC
#endif // IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_C310_H_
 