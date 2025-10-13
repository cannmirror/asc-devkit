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
 * \file geglu_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GEGLU_GEGLU_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GEGLU_GEGLU_CHECK_COMMON_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/calcount_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
class CheckFuncClassGeGLU : public DataTypeCheckFuncBasicClass, public CalCountCheckFuncBasicClass,
    public ReuseSourceCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass,
    public MultipleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassGeGLU() {};
    __aicore__ inline CheckFuncClassGeGLU(__gm__ const char *apiName) : DataTypeCheckFuncBasicClass(apiName),
        CalCountCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
        const LocalTensor<T> &srcTensor1, const LocalTensor<uint8_t> &sharedTmpBuffer, uint32_t calCount) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half or float");

        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor0,
            srcTensor1, sharedTmpBuffer), VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor1));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor0));

        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(
            ARG_AND_STRING(calCount), VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor0, srcTensor1));
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_GEGLU_GEGLU_CHECK_COMMON_H_
