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
 * \file layernormgrad_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRAD_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRAD_CHECK_COMMON_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"
#include "include/adv_api/normalization/layernormgrad_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckLayerNormGradParamsClass {
public:
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void CheckLayerNormGradParams(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma,
        const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance,
        const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, LocalTensor<uint8_t> &sharedTmpBuffer, T epsilon,
        LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {}) {
        VerifyingParameters<T, isReuseSource>(outputPdX, resForGamma, inputDy, inputX, inputVariance, inputMean,
                                              inputGamma, sharedTmpBuffer, epsilon, tiling, shapeInfo);
    }

private:
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma,
        const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance,
        const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, LocalTensor<uint8_t> &sharedTmpBuffer, T epsilon,
        LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {}) {
        ASCENDC_ASSERT(((inputGamma.GetSize() * sizeof(T)) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
            "[LayerNormGrad] The inputGamma length is %lu, should be 32B aligned.", inputGamma.GetSize() * sizeof(T)); });
        ASCENDC_ASSERT((shapeInfo.dataFormat == DataFormat::ND || HighLevelAPIParametersPrint),
                       { KERNEL_LOG(KERNEL_ERROR, "[LayerNormGrad] Only support ND data format."); });
    }
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassLayerNormGrad : public DataTypeCheckFuncBasicClass,
    public ReuseSourceCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass,
    public MultipleTensorCheckFuncBasicClass, public CheckLayerNormGradParamsClass {
public:
    __aicore__ inline CheckFuncClassLayerNormGrad() {};
    __aicore__ inline CheckFuncClassLayerNormGrad(__gm__ const char *apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma,
        const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance,
        const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, LocalTensor<uint8_t> &sharedTmpBuffer, T epsilon,
        LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {}) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half or float");

        if (std::is_same<T, half>::value) {
            ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));
        }

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputPdX, resForGamma, inputDy, inputX, inputVariance, inputMean, inputGamma, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputPdX, resForGamma));

        CheckLayerNormGradParamsClass::CheckLayerNormGradParams(outputPdX, resForGamma, inputDy, inputX, inputVariance,
                                                                inputMean, inputGamma, sharedTmpBuffer, epsilon,
                                                                tiling, shapeInfo);
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRAD_CHECK_COMMON_H_
 