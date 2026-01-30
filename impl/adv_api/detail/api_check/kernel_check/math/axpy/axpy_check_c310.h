/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file axpy_check_c310.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_AXPY_AXPY_CHECK_C310_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_AXPY_AXPY_CHECK_C310_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/calcount_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, typename U, bool isReuseSource>
class CheckFuncClassAxpy : public CalCountCheckFuncBasicClass, public DataTypeCheckFuncBasicClass,
    public ReuseSourceCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass, public MultipleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassAxpy() {};
    __aicore__ inline CheckFuncClassAxpy(__gm__ const char *apiName) :
        DataTypeCheckFuncBasicClass(apiName), CalCountCheckFuncBasicClass(apiName),
        ReuseSourceCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor,
        const U scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<Std::tuple<T, U>, Std::tuple<float, half>, Std::tuple<half, half>,
            Std::tuple<float, float>>(" (T, U) is not (float, half)/(half, half)/(float, float)");

        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(calCount), VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor, sharedTmpBuffer));
    };
};
}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_AXPY_AXPY_CHECK_C310_H_
