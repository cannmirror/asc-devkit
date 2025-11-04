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
 * \file batchnorm_check_c310.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_BATCHNORM_BATCHNORM_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_BATCHNORM_BATCHNORM_CHECK_COMMON_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckBatchNormParamsClass {
public:
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void CheckBatchNormParams(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
        const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
        const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling) {
        VerifyingParameters<T, isReuseSource>(output, outputMean,
            outputVariance, inputX, gamm, beta, sharedTmpBuffer, epsilon, tiling);
    }

private:
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
        const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
        const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling){
        ShapeInfo outputShape = output.GetShapeInfo();
        ShapeInfo outputMeanShape = outputMean.GetShapeInfo();
        ShapeInfo outputVarianceShape = outputVariance.GetShapeInfo();
        ShapeInfo inputXShape = inputX.GetShapeInfo();
        ShapeInfo gammShape = gamm.GetShapeInfo();
        ShapeInfo betaShape = beta.GetShapeInfo();
        ASCENDC_ASSERT((((outputShape.shape[0] == inputXShape.shape[0]) && (outputShape.shape[0] == gammShape.shape[0]) &&
            (outputShape.shape[0] == betaShape.shape[0])) || HighLevelAPIParametersPrint), {KERNEL_LOG(KERNEL_ERROR,
            "[BatchNorm] The B dims of output, inputX, gamm, beta are %u, %u, %u, %u. They should be same.",
            outputShape.shape[0], inputXShape.shape[0], gammShape.shape[0], betaShape.shape[0]); });
        ASCENDC_ASSERT((((outputShape.shape[1] == inputXShape.shape[1]) && (outputShape.shape[1] == outputMeanShape.shape[1]) &&
            (outputShape.shape[1] == outputVarianceShape.shape[1])) || HighLevelAPIParametersPrint), {KERNEL_LOG(KERNEL_ERROR,
            "[BatchNorm] The S dims of output, inputX, outputMean, outputVariance are %u, %u, %u, %u. They should be same.",
            outputShape.shape[1], inputXShape.shape[1], outputMeanShape.shape[1], outputVarianceShape.shape[1]); });
        ASCENDC_ASSERT((((outputShape.shape[2] == inputXShape.shape[2]) && (outputShape.shape[2] == outputMeanShape.shape[2]) &&
            (outputShape.shape[2] == outputVarianceShape.shape[2])) || HighLevelAPIParametersPrint), {KERNEL_LOG(KERNEL_ERROR,
            "[BatchNorm] The H dims of output, inputX, outputMean, outputVariance are %u, %u, %u, %u. They should be same.",
            outputShape.shape[2], inputXShape.shape[2], outputMeanShape.shape[2], outputVarianceShape.shape[2]); });
        ASCENDC_ASSERT(((inputXShape.shape[1] * inputXShape.shape[2] * sizeof(T) % ONE_BLK_SIZE == 0) ||
            HighLevelAPIParametersPrint), { KERNEL_LOG(KERNEL_ERROR,
            "[BatchNorm] The inputX slength, hlength are %u, %u, slength * hlength should be 32B aligned.",
            inputXShape.shape[1] * sizeof(T), inputXShape.shape[2]* sizeof(T)); });
        ASCENDC_ASSERT(((gammShape.shape[0] * sizeof(T) % ONE_BLK_SIZE == 0) ||
            HighLevelAPIParametersPrint), { KERNEL_LOG(KERNEL_ERROR,
            "[BatchNorm] The gamm length is %u, should be 32B aligned.", gammShape.shape[0] * sizeof(T)); });
        ASCENDC_ASSERT(((betaShape.shape[0] * sizeof(T) % ONE_BLK_SIZE == 0) ||
            HighLevelAPIParametersPrint), { KERNEL_LOG(KERNEL_ERROR,
            "[BatchNorm] The beta length is %u, should be 32B aligned.", betaShape.shape[0] * sizeof(T)); });
    }
};

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
class CheckFuncClassBatchNorm : public DataTypeCheckFuncBasicClass, public ReuseSourceCheckFuncBasicClass,
    public SingleTensorCheckFuncBasicClass, public MultipleTensorCheckFuncBasicClass, public CheckBatchNormParamsClass {
public:
    __aicore__ inline CheckFuncClassBatchNorm() {};
    __aicore__ inline CheckFuncClassBatchNorm(__gm__ const char *apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName) {};
public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
        const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
        const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half or float");
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));
        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(output, outputMean, outputVariance, inputX, gamm, beta, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(output, outputMean, outputVariance));
        CheckBatchNormParamsClass::CheckBatchNormParams(output, outputMean, outputVariance, inputX, gamm, beta,
                                                        sharedTmpBuffer, epsilon, tiling);
    };
};
} // namespace HeighLevelApiCheck
} // namespace AscendC
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_BATCHNORM_BATCHNORM_CHECK_COMMON_H_