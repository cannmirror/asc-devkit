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
 * \file softmax_grad_front_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_FRONT_SOFTMAX_GRAD_FRONT_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_FRONT_SOFTMAX_GRAD_FRONT_CHECK_COMMON_H

#include "../../../basic_check/datatype_check.h"
#include "../../../basic_check/single_tensor_check.h"
#include "../../../basic_check/multiple_tensor_check.h"
#include "../../../basic_check/reuse_source_check.h"
#include "activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckSoftmaxGradFrontParamsClass {
public:
    template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
    __aicore__ inline void CheckSoftmaxGradFrontParams(const LocalTensor<T>& dstTensor,
        const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const LocalTensor<float>& sharedTmpBuffer,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        VerifyingParameters<T, isBasicBlock, isDataFormatNZ>(
            dstTensor, gradTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, isBasicBlock, isDataFormatNZ>(
                dstTensor, gradTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
        }
    }

private:
    template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
        const LocalTensor<T>& srcTensor, const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SoftmaxGradFront] The softmaxShapeInfo.srcK is %u, should be 32B aligned.",
                softmaxShapeInfo.srcK);
        });
        ASCENDC_ASSERT(
            (softmaxShapeInfo.srcK * softmaxShapeInfo.srcM <= srcTensor.GetSize() || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[SoftmaxGradFront] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
                    "the product of softmaxShapeInfo.srcM and softmaxShapeInfo.srcK should not be greater than srcTensor size %u.",
                    softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, srcTensor.GetSize());
            });
        ASCENDC_ASSERT(((srcTensor.GetSize() == gradTensor.GetSize()) || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[SoftmaxGradFront] The srcTensor size %u, gradTensor size is %u, they should be equal.",
                srcTensor.GetSize(), gradTensor.GetSize());
        });
    }

    template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
    __aicore__ inline void PrintParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
        const LocalTensor<T>& srcTensor, const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        KERNEL_LOG(KERNEL_INFO,
            "[SoftmaxGradFront] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
            "softmaxShapeInfo.oriSrcM is %u, softmaxShapeInfo.oriSrcK is %u.",
            softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK);
    }
};

template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
class CheckFuncClassSoftmaxGradFront : public DataTypeCheckFuncBasicClass,
                                       public SingleTensorCheckFuncBasicClass,
                                       public ReuseSourceCheckFuncBasicClass,
                                       public MultipleTensorCheckFuncBasicClass,
                                       public CheckSoftmaxGradFrontParamsClass {
public:
    __aicore__ inline CheckFuncClassSoftmaxGradFront(){};
    __aicore__ inline CheckFuncClassSoftmaxGradFront(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName),
        MultipleTensorCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
        const LocalTensor<T>& srcTensor, const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half or float");
#if __CCE_AICORE__ == 300
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isBasicBlock));
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isDataFormatNZ));
#endif
        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, gradTensor, srcTensor, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CheckSoftmaxGradFrontParamsClass::CheckSoftmaxGradFrontParams<T, isBasicBlock, isDataFormatNZ>(
            dstTensor, gradTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_FRONT_SOFTMAX_GRAD_FRONT_CHECK_COMMON_H
