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
 * \file faster_gelu_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GELU_FASTER_GELU_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GELU_FASTER_GELU_CHECK_COMMON_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/calcount_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckFasterGeluParamsClass {
public:
    template <typename T, bool highPrecision = false, bool highPerformance = false>
    __aicore__ inline void CheckFasterGeluParams(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize) {
        VerifyingParameters<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
        }
    }

private:
    template <typename T, bool highPrecision = false, bool highPerformance = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize) {
        if (std::is_same<T, float>::value) {
            ASCENDC_ASSERT((highPrecision == false || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_WARN,
                "[FasterGelu] The highPrecision is true, may not be effective."); });
        }
    }

    template <typename T, bool highPrecision = false, bool highPerformance = false>
    __aicore__ inline void PrintParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize) {
        KERNEL_LOG(KERNEL_INFO,
            "[FasterGelu] The highPrecision is %d, highPerformance is %d.", highPrecision, highPerformance);
    }
};

template <typename T, bool highPrecision = false, bool highPerformance = false>
class CheckFuncClassFasterGelu : public DataTypeCheckFuncBasicClass, public CalCountCheckFuncBasicClass,
    public SingleTensorCheckFuncBasicClass, public MultipleTensorCheckFuncBasicClass, public CheckFasterGeluParamsClass {
public:
    __aicore__ inline CheckFuncClassFasterGelu() {};
    __aicore__ inline CheckFuncClassFasterGelu(__gm__ const char *apiName) : MultipleTensorCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName),DataTypeCheckFuncBasicClass(apiName),
        CalCountCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half or float");

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(dstLocal, sharedTmpBuffer, srcLocal),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));
        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(dataSize),
            VA_ARGS_TO_MAKE_TUPLE(dstLocal, srcLocal));
        CheckFasterGeluParamsClass::CheckFasterGeluParams<T, highPrecision, highPerformance>(
            dstLocal, srcLocal, sharedTmpBuffer, dataSize);
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GELU_FASTER_GELU_CHECK_COMMON_H_
