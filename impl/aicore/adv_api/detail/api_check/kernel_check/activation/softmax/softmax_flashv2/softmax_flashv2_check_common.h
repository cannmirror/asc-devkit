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
 * \file softmax_flashv2_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_SOFTMAX_FLASHV2_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_SOFTMAX_FLASHV2_CHECK_COMMON_H

#include "../../../basic_check/datatype_check.h"
#include "../../../basic_check/reuse_source_check.h"
#include "../../../basic_check/single_tensor_check.h"
#include "../../../basic_check/multiple_tensor_check.h"
#include "activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
constexpr uint8_t CHECK_SOFTMAXFLASHV2_SRCM_SIZE = 8;
constexpr uint8_t CHECK_SOFTMAXFLASHV2_K_SIZE = 64;

class CheckSoftMaxFlashV2ParamsClass {
public:
    template <typename T1, typename T2, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ,
        const SoftmaxConfig& config>
    __aicore__ inline void CheckSoftMaxFlashV2Params(const LocalTensor<T1>& dstTensor,
        const LocalTensor<T2>& expSumTensor, const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor,
        const LocalTensor<T1>& expMaxTensor, const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor,
        const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        VerifyingParameters<T1, T2, isUpdate, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dstTensor,
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer, tiling,
            softmaxShapeInfo);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T1, T2, isUpdate, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dstTensor,
                expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer, tiling,
                softmaxShapeInfo);
        }
    }

private:
    template <typename T1, typename T2, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ,
        const SoftmaxConfig& config>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
        const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor,
        const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T1) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SoftMaxFlashV2] The softmaxShapeInfo.srcK is %u, should be 32B aligned.",
                softmaxShapeInfo.srcK);
        });
        ASCENDC_ASSERT(
            (softmaxShapeInfo.srcK * softmaxShapeInfo.srcM <= srcTensor.GetSize() || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[SoftMaxFlashV2] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
                    "the product of softmaxShapeInfo.srcM and softmaxShapeInfo.srcK should not be greater than srcTensor size %u.",
                    softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, srcTensor.GetSize());
            });
        ASCENDC_ASSERT((dstTensor.GetSize() == srcTensor.GetSize() || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SoftMaxFlashV2] The dstTensor size is %u, should be equal to srcTensor size %u.",
                dstTensor.GetSize(), srcTensor.GetSize());
        });
#if __CCE_AICORE__ == 300
        ASCENDC_LOG_IF_CHECK((isBasicBlock == false || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_WARN, "[SoftMaxFlashV2] The isBasicBlock is true, may not be effective in this device.");
        });
        ASCENDC_LOG_IF_CHECK((isDataFormatNZ == false || HighLevelAPIParametersPrint), {
            KERNEL_LOG(
                KERNEL_WARN, "[SoftMaxFlashV2] The isDataFormatNZ is true, may not be effective in this device.");
        });

        bool ans = config.isCheckTiling == true && config.oriSrcM == 0 && config.oriSrcK == 0
                   && config.mode == SoftmaxMode::SOFTMAX_NORMAL;
        ASCENDC_LOG_IF_CHECK((ans || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_WARN,
                "[SoftMaxFlashV2] The config.isCheckTiling is %d, config.oriSrcM is %u, "
                "config.oriSrcK is %u, config.mode is %d, should remain at the default value in this device.",
                config.isCheckTiling, config.oriSrcM, config.oriSrcK, config.mode);
        });
#endif
#if __CCE_AICORE__ == 200
        ASCENDC_LOG_IF_CHECK((config.mode == SoftmaxMode::SOFTMAX_NORMAL || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_WARN,
                "[SoftMaxFlashV2] The config.mode is %d, should be defalut value SoftmaxMode::SOFTMAX_NORMAL.",
                config.mode);
        });
#endif
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
        if (isBasicBlock == true && (softmaxShapeInfo.srcK != 0 || softmaxShapeInfo.srcM != 0)) {
            bool ans = (softmaxShapeInfo.srcK < 2048) && (softmaxShapeInfo.srcK >= DEFAULT_BLOCK_SIZE / sizeof(T1))
                       && (softmaxShapeInfo.srcK % CHECK_SOFTMAXFLASHV2_K_SIZE == 0);
            ASCENDC_ASSERT((ans || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[SoftMaxFlashV2] The softmaxShapeInfo.srcK is %u, should be less than 2048 and greater than or equal to "
                    "256/sizeof(T), and should be an integer multiple of 64.",
                    softmaxShapeInfo.srcK);
            });
            ASCENDC_ASSERT(
                (softmaxShapeInfo.srcM % CHECK_SOFTMAXFLASHV2_SRCM_SIZE == 0 || HighLevelAPIParametersPrint), {
                    KERNEL_LOG(KERNEL_ERROR,
                        "[SoftMaxFlashV2] The softmaxShapeInfo.srcM is %u, should be an integer multiple of 8.",
                        softmaxShapeInfo.srcM);
                });
        }
#endif
    }

    template <typename T1, typename T2, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ,
        const SoftmaxConfig& config>
    __aicore__ inline void PrintParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
        const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor,
        const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        KERNEL_LOG(KERNEL_INFO,
            "[SoftMaxFlashV2] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
            "softmaxShapeInfo.oriSrcM is %u, softmaxShapeInfo.oriSrcK is %u.",
            softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK);
    }
};

template <typename T1, typename T2, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ,
    const SoftmaxConfig& config>
class CheckFuncClassSoftmaxFlashV2 : public DataTypeCheckFuncBasicClass,
                                     public CheckSoftMaxFlashV2ParamsClass,
                                     public ReuseSourceCheckFuncBasicClass,
                                     public SingleTensorCheckFuncBasicClass,
                                     public MultipleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassSoftmaxFlashV2(){};
    __aicore__ inline CheckFuncClassSoftmaxFlashV2(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
        const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor,
        const LocalTensor<float>& sharedTmpBuffer, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T1, half, float>(
            "template parameter (T) is not half or float");

        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T2, half, float>(
            "template parameter (T) is not half or float");

        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CheckSoftMaxFlashV2ParamsClass::CheckSoftMaxFlashV2Params<T1, T2, isUpdate, isReuseSource, isBasicBlock,
            isDataFormatNZ, config>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, sharedTmpBuffer, tiling, softmaxShapeInfo);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_SOFTMAX_FLASHV2_CHECK_COMMON_H
