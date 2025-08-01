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
 * \file reglu_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_REGLU_REGLU_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_REGLU_REGLU_CHECK_COMMON_H

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/calcount_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckReGluParamsClass {
public:
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void CheckReGluParams(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
        const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
        VerifyingParameters<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
        }
    }

private:
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
        const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
        bool ans = (dstTensor.GetSize() == srcTensor0.GetSize() && srcTensor0.GetSize() == srcTensor1.GetSize());
        ASCENDC_ASSERT((ans || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[ReGlu] The dstTensor size is %u, srcTensor0 size is %u, srcTensor1 is %u, those should be equal.",
                dstTensor.GetSize(), srcTensor0.GetSize(), srcTensor1.GetSize());
        });
    }

    template <typename T, bool isReuseSource = false>
    __aicore__ inline void PrintParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
        const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
        KERNEL_LOG(KERNEL_INFO, "[ReGlu] The dstTensor size is %u, srcTensor0 size is %u, srcTensor1 is %u.",
            dstTensor.GetSize(), srcTensor0.GetSize(), srcTensor1.GetSize());
    }
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassReGlu : public DataTypeCheckFuncBasicClass,
                            public CalCountCheckFuncBasicClass,
                            public ReuseSourceCheckFuncBasicClass,
                            public SingleTensorCheckFuncBasicClass,
                            public MultipleTensorCheckFuncBasicClass,
                            public CheckReGluParamsClass {
public:
    __aicore__ inline CheckFuncClassReGlu(){};
    __aicore__ inline CheckFuncClassReGlu(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), CalCountCheckFuncBasicClass(apiName),
        ReuseSourceCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName),
        MultipleTensorCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
        const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
#if __CCE_AICORE__ == 220
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float, bfloat16_t>(
            "template parameter (T) is not half/float/bfloat16_t");
#elif __CCE_AICORE__ == 200
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half/float");
#endif
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(srcTensor0, srcTensor1, dstTensor, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor1, sharedTmpBuffer));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor0, sharedTmpBuffer));

        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(
            ARG_AND_STRING(calCount), VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor0, srcTensor1));

        CheckReGluParamsClass::CheckReGluParams<T, isReuseSource>(
            dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_REGLU_REGLU_CHECK_COMMON_H
