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
 * \file isnan_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_ISNAN_ISNAN_CHECK_COMMON_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_ISNAN_ISNAN_CHECK_COMMON_H

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/calcount_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, typename U, bool isReuseSource>
class CheckFuncClassIsNan : public DataTypeCheckFuncBasicClass, public CalCountCheckFuncBasicClass,
    public ReuseSourceCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass, public MultipleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassIsNan() {};
    __aicore__ inline CheckFuncClassIsNan(__gm__ const char* name) :
        DataTypeCheckFuncBasicClass(name), CalCountCheckFuncBasicClass(name),
        ReuseSourceCheckFuncBasicClass(name), SingleTensorCheckFuncBasicClass(name), MultipleTensorCheckFuncBasicClass(name) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const LocalTensor<U>& src,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count) {
        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(count), 
            VA_ARGS_TO_MAKE_TUPLE(dst, src));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dst, src, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dst, src));
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_ISNAN_ISNAN_CHECK_COMMON_H
