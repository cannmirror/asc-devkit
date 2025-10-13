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
 * \file simple_softmax_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_SIMPLE_SOFTMAX_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_SIMPLE_SOFTMAX_CHECK_COMMON_H_

#include "../../../basic_check/datatype_check.h"
#include "../../../basic_check/reuse_source_check.h"
#include "../../../basic_check/single_tensor_check.h"
#include "../../../basic_check/multiple_tensor_check.h"
#include "include/adv_api/activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
constexpr uint8_t CHECK_SIMPLESOFTMAX_K_SIZE = 64;
constexpr uint8_t CHECK_SIMPLESOFTMAX_SRCM_SIZE = 8;

class CheckSimpleSoftMaxParamsClass {
public:
    template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
        const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void CheckSimpleSoftMaxParams(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& inSumTensor,
        const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<float>& sharedTmpBuffer,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo) {
        VerifyingParameters<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
            dstTensor, inSumTensor, inMaxTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
                dstTensor, inSumTensor, inMaxTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
        }
    }

private:
    template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
        const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& inSumTensor,
        const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<float>& sharedTmpBuffer,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo) {
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T1) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
            "[SimpleSoftMax] The softmaxShapeInfo.srcK is %u, should be 32B aligned.", softmaxShapeInfo.srcK); });
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * softmaxShapeInfo.srcM <= srcTensor.GetSize() || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SimpleSoftMax] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
            "the product of softmaxShapeInfo.srcM and softmaxShapeInfo.srcK should not be greater than srcTensor size %u.",
            softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, srcTensor.GetSize()); });
        ASCENDC_ASSERT((dstTensor.GetSize() == srcTensor.GetSize() || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SimpleSoftMax] The dstTensor size is %u, should be equal to srcTensor size %u.",
            dstTensor.GetSize(), srcTensor.GetSize()); });
#if __NPU_ARCH__ == 3002
        ASCENDC_LOG_IF_CHECK((isBasicBlock == false || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_WARN, "[SimpleSoftMax] The isBasicBlock is true, may not be effective in this device."); });
        ASCENDC_LOG_IF_CHECK((isDataFormatNZ == false || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_WARN, "[SimpleSoftMax] The isDataFormatNZ is true, may not be effective in this device."); });
#endif
#if __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201
        if (isBasicBlock == true && (softmaxShapeInfo.srcK != 0 || softmaxShapeInfo.srcM != 0)) {
            bool ans = (softmaxShapeInfo.srcK < 2048) && (softmaxShapeInfo.srcK >= DEFAULT_BLOCK_SIZE / sizeof(T1)) &&
                (softmaxShapeInfo.srcK % CHECK_SIMPLESOFTMAX_K_SIZE == 0 );
            ASCENDC_ASSERT((ans || HighLevelAPIParametersPrint), {  KERNEL_LOG(KERNEL_ERROR,
                "[SimpleSoftMax] The softmaxShapeInfo.srcK is %u, should be less than 2048 and greater than or equal to "
                "256/sizeof(T), and should be an integer multiple of 64.", softmaxShapeInfo.srcK); });
            ASCENDC_ASSERT((softmaxShapeInfo.srcM % CHECK_SIMPLESOFTMAX_SRCM_SIZE == 0 || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR, "[SimpleSoftMax] The softmaxShapeInfo.srcM is %u, should be an integer multiple of 8.",
                softmaxShapeInfo.srcM); });
        }
#endif
    }

    template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
        const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void PrintParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& inSumTensor,
        const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<float>& sharedTmpBuffer,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo) {
        KERNEL_LOG(KERNEL_INFO, "[SimpleSoftMax] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
            "softmaxShapeInfo.oriSrcM is %u, softmaxShapeInfo.oriSrcK is %u.", softmaxShapeInfo.srcK,
            softmaxShapeInfo.srcM, softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK);
    }
};

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
class CheckFuncClassSimpleSoftMax : public DataTypeCheckFuncBasicClass, public ReuseSourceCheckFuncBasicClass,
    public SingleTensorCheckFuncBasicClass, public MultipleTensorCheckFuncBasicClass, public CheckSimpleSoftMaxParamsClass {
public:
    __aicore__ inline CheckFuncClassSimpleSoftMax() {};
    __aicore__ inline CheckFuncClassSimpleSoftMax(__gm__ const char *apiName) : DataTypeCheckFuncBasicClass(apiName),
        ReuseSourceCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName),
        MultipleTensorCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& inSumTensor,
        const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<float>& sharedTmpBuffer,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T1, half, float>(
            "first template parameter (T1) is not half or float");

        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T2, half, float>(
            "second template parameter (T2) is not half or float");

        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor, inSumTensor, inMaxTensor, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CheckSimpleSoftMaxParamsClass::CheckSimpleSoftMaxParams<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
            dstTensor, inSumTensor, inMaxTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_SIMPLE_SOFTMAX_CHECK_COMMON_H_
