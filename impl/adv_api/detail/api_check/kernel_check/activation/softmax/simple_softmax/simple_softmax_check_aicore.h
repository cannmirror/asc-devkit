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
 * \file simple_softmax_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_SIMPLE_SOFTMAX_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_SIMPLE_SOFTMAX_CHECK_AICORE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
class CheckFuncClassSimpleSoftMax {
public:
    __aicore__ inline CheckFuncClassSimpleSoftMax() {};
    __aicore__ inline CheckFuncClassSimpleSoftMax(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<float>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo) {
        CheckTensorPos<T1>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
        CheckTensorPos<T2>(inSumTensor, Hardware::UB, "inSumTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
        CheckTensorPos<T2>(inMaxTensor, Hardware::UB, "inMaxTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
        CheckTensorPos<T1>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T1) % ONE_BLK_SIZE == 0), {
            KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK);
        });
        static_assert(std::is_same<T1, half>::value || std::is_same<T1, float>::value,
                "This Related Api of Softmax only support half/float input dtype");
        static_assert(std::is_same<T2, half>::value || std::is_same<T2, float>::value,
                "This Related Api of Softmax only support half/float input dtype");
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_SIMPLE_SOFTMAX_CHECK_AICORE_H_
 