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
 * \file welfordfinalize_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_COMMON_H

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"
#include "normalization/welfordfinalize_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
constexpr float CHECK_WELFINAL_FLOAT_RTOL = float(1e-4);
class CheckWelfordFinalizeParamsClass {
public:
    template <bool isReuseSource = false>
    __aicore__ inline void CheckWelfordFinalizeParams(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        VerifyingParameters<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
        }
    }

private:
    template <bool isReuseSource = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        ASCENDC_ASSERT((para.abLength == para.headCountLength + para.tailCountLength || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[WelfordFinalize] The para.abLength is %u, should be the sum of para.headCountLength %u and para.tailCountLength %u.",
                para.abLength, para.headCountLength, para.tailCountLength);
        });
        bool ans = ((para.abRec - 1.0 / para.abLength > -CHECK_WELFINAL_FLOAT_RTOL)
                    && (para.abRec - 1.0 / para.abLength < CHECK_WELFINAL_FLOAT_RTOL));
        ASCENDC_ASSERT((ans || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[WelfordFinalize] The para.abRec is %u, should be equant to 1 / para.abLength, para.abLength is %u.",
                para.abRec, para.abLength);
        });
        ASCENDC_ASSERT((para.abLength * sizeof(float) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[WelfordFinalize] The para.abLength is %u, should be an integral multiple of 32/sizeof(float).",
                para.abLength);
        });
        if (para.tailCount == 0) {
            ASCENDC_ASSERT((para.tailCountLength == 0 || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[WelfordFinalize] The para.tailCountLength is %u, should be 0 when para.tailCount is 0.",
                    para.tailCountLength);
            });
        }
    }

    template <bool isReuseSource = false>
    __aicore__ inline void PrintParameters(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        KERNEL_LOG(KERNEL_INFO,
            "[WelfordFinalize] The para.rnLength is %u, para.abLength is %u, para.headCount is %u, "
            "para.headCountLength is %u, para.tailCount is %u, para.tailCountLength is %u, para.abRec is %f, para.rRec is %f.",
            para.rnLength, para.abLength, para.headCount, para.headCountLength, para.tailCount, para.tailCountLength,
            para.abRec, para.rRec);
    }
};

class CheckWelfordFinalizeParamsClassNoCounts : public CheckWelfordFinalizeParamsClass {
public:
    template <bool isReuseSource = false>
    __aicore__ inline void CheckWelfordFinalizeParamsNoCounts(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        VerifyingParametersNoCounts<isReuseSource>(
            outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
    }

private:
    template <bool isReuseSource = false>
    __aicore__ inline void VerifyingParametersNoCounts(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        CheckWelfordFinalizeParamsClass::CheckWelfordFinalizeParams<isReuseSource>(
            outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
        bool ans = (para.rRec - 1.0 / (para.rnLength * para.abLength) > -CHECK_WELFINAL_FLOAT_RTOL)
                   && (para.rRec - 1.0 / (para.rnLength * para.abLength) < CHECK_WELFINAL_FLOAT_RTOL);
        ASCENDC_ASSERT((ans || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[WelfordFinalize] The para.rRec is %f, should be equal to 1 / (para.rnLength * para.abLength), "
                "para.rnLength is %u, para.abLength is %u.",
                para.rRec, para.rnLength, para.abLength);
        });
    }
};

class CheckWelfordFinalizeParamsClassCounts : public CheckWelfordFinalizeParamsClass {
public:
    template <bool isReuseSource = false>
    __aicore__ inline void CheckWelfordFinalizeParamsCounts(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        VerifyingParametersCounts<isReuseSource>(
            outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
    }

private:
    template <bool isReuseSource = false>
    __aicore__ inline void VerifyingParametersCounts(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        CheckWelfordFinalizeParamsClass::CheckWelfordFinalizeParams<isReuseSource>(
            outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
    }
};

template <bool isReuseSource = false>
class CheckFuncClassWelfordFinalizeCounts : public DataTypeCheckFuncBasicClass,
                                            public ReuseSourceCheckFuncBasicClass,
                                            public SingleTensorCheckFuncBasicClass,
                                            public MultipleTensorCheckFuncBasicClass,
                                            public CheckWelfordFinalizeParamsClassCounts {
public:
    __aicore__ inline CheckFuncClassWelfordFinalizeCounts(){};
    __aicore__ inline CheckFuncClassWelfordFinalizeCounts(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const WelfordFinalizePara& para)
    {
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputMean, outputVariance, inputMean, inputVariance, counts, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputMean, outputVariance, inputMean, sharedTmpBuffer));
        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputMean, outputVariance, inputVariance, sharedTmpBuffer));
        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputMean, outputVariance, counts, sharedTmpBuffer));

        CheckWelfordFinalizeParamsClassCounts::CheckWelfordFinalizeParamsCounts<isReuseSource>(
            outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
    };
};

template <bool isReuseSource = false>
class CheckFuncClassWelfordFinalize : public DataTypeCheckFuncBasicClass,
                                      public ReuseSourceCheckFuncBasicClass,
                                      public SingleTensorCheckFuncBasicClass,
                                      public MultipleTensorCheckFuncBasicClass,
                                      public CheckWelfordFinalizeParamsClassNoCounts {
public:
    __aicore__ inline CheckFuncClassWelfordFinalize(){};
    __aicore__ inline CheckFuncClassWelfordFinalize(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<float>& outputMean,
        const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
        const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const WelfordFinalizePara& para)
    {
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer));

        CheckWelfordFinalizeParamsClassNoCounts::CheckWelfordFinalizeParamsNoCounts<isReuseSource>(
            outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_COMMON_H
