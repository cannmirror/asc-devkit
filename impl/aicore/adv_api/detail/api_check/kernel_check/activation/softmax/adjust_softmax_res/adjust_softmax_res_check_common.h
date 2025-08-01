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
 * \file adjust_softmax_res_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_COMMON_H

#include "../../../basic_check/datatype_check.h"
#include "../../../basic_check/single_tensor_check.h"
#include "../../../basic_check/multiple_tensor_check.h"
#include "activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckAdjustSoftMaxResParamsClass {
public:
    template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
    __aicore__ inline void CheckAdjustSoftMaxResParams(const LocalTensor<T1>& softMaxRes,
        const LocalTensor<T2>& maxTensor, const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        VerifyingParameters<T1, T2, isDataFormatNZ, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T1, T2, isDataFormatNZ, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
        }
    }

private:
    template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
        const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T1) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[AdjustSoftMaxRes] The softmaxShapeInfo.srcK is %u, should be 32B aligned.",
                softmaxShapeInfo.srcK);
        });
        ASCENDC_ASSERT(
            (softmaxShapeInfo.srcK * softmaxShapeInfo.srcM <= softMaxRes.GetSize() || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[AdjustSoftMaxRes] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
                    "the product of softmaxShapeInfo.srcM and softmaxShapeInfo.srcK should not be greater than softMaxRes size %u.",
                    softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, softMaxRes.GetSize());
            });
    }

    template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
    __aicore__ inline void PrintParameters(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
        const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        KERNEL_LOG(KERNEL_INFO,
            "[AdjustSoftMaxRes] The softmaxShapeInfo.srcK is %u, softmaxShapeInfo.srcM is %u, "
            "softmaxShapeInfo.oriSrcM is %u, softmaxShapeInfo.oriSrcK is %u.",
            softmaxShapeInfo.srcK, softmaxShapeInfo.srcM, softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK);
    }
};

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
class CheckFuncClassAdjustSoftMaxRes : public DataTypeCheckFuncBasicClass,
                                       public SingleTensorCheckFuncBasicClass,
                                       public MultipleTensorCheckFuncBasicClass,
                                       public CheckAdjustSoftMaxResParamsClass {
public:
    __aicore__ inline CheckFuncClassAdjustSoftMaxRes(){};
    __aicore__ inline CheckFuncClassAdjustSoftMaxRes(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName),
        MultipleTensorCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
        const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
    {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T1, half, float>(
            "first template parameter (T1) is not half or float");

        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T2, half, float>(
            "second template parameter (T2) is not half or float");

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(softMaxRes, maxTensor),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CheckAdjustSoftMaxResParamsClass::CheckAdjustSoftMaxResParams<T1, T2, isDataFormatNZ, stepSizeMode>(
            softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_COMMON_H
