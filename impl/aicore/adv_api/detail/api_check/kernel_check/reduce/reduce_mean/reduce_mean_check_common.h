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
 * \file reduce_mean_check
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_COMMON_H

#include "../reduce_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, class pattern>
class CheckFuncClassReduceMean : public CheckFuncClassReduce<T, pattern> {
public:
    __aicore__ inline CheckFuncClassReduceMean(__gm__ const char* apiName) :
        CheckFuncClassReduce<T, pattern>(apiName){};

    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], const bool srcInnerPad,
        const uint32_t padLast)
    {
        CheckFuncClassReduce<T, pattern>::VerifyingParameters(
            dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, padLast);
    };
};
} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_MEAN_REDUCE_MEAN_CHECK_COMMON_H
