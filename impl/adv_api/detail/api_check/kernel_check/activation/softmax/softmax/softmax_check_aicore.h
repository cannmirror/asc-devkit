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
 * \file softmax_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_SOFTMAX_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_SOFTMAX_CHECK_AICORE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
class CheckFuncClassSoftMax {
public:
    __aicore__ inline CheckFuncClassSoftMax() {};
    __aicore__ inline CheckFuncClassSoftMax(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo) {
        CheckTensorPos<float>(workLocal, Hardware::UB, "workLocal", "VECIN / VECCALC / VECOUT", "SoftMax");
        CheckTensorPos<T1>(dst, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
        CheckTensorPos<T2>(sumTensor, Hardware::UB, "sumTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
        CheckTensorPos<T2>(maxTensor, Hardware::UB, "maxTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
        CheckTensorPos<T1>(src, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T1) % ONE_BLK_SIZE == 0), {
            KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK);
        });
        static_assert(std::is_same<T1, half>::value || std::is_same<T1, float>::value,
            "This Related Api of Softmax only support half/float input dtype");
        static_assert(std::is_same<T1, half>::value || std::is_same<T2, float>::value,
            "This Related Api of Softmax only support half/float input dtype");
    };
};

template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
class CheckFuncClassSoftMaxSameT {
public:
    __aicore__ inline CheckFuncClassSoftMaxSameT() {};
    __aicore__ inline CheckFuncClassSoftMaxSameT(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo) {
        CheckTensorPos<float>(workLocal, Hardware::UB, "workLocal", "VECIN / VECCALC / VECOUT", "SoftMax");
        CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
        CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0), {
            KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK);
        });
        static_assert(std::is_same<T, half>::value || std::is_same<T, float>::value,
            "This Related Api of Softmax only support half/float input dtype");
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_SOFTMAX_CHECK_AICORE_H_
