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
 * \file where_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_WHERE_WHERE_CHECK_COMMON_H
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_WHERE_WHERE_CHECK_COMMON_H

#include "../../basic_check/calcount_check.h"
#include "../../basic_check/single_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, typename U, typename S, typename V>
class CheckFuncClassWhere : public CalCountCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassWhere() {};
    __aicore__ inline CheckFuncClassWhere(__gm__ const char* name) :
        CalCountCheckFuncBasicClass(name), SingleTensorCheckFuncBasicClass(name) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const U& src0,
        const S& src1, const LocalTensor<V>& condition, const uint32_t count) {
        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(count), 
            VA_ARGS_TO_MAKE_TUPLE(dst, condition));
        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dst, condition),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));
        if constexpr (TypeUtils::IsLocalTensorType<U>()) {
            CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(count), 
                VA_ARGS_TO_MAKE_TUPLE(src0));
            SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
                VA_ARGS_TO_MAKE_TUPLE(src0),
                VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));
        }
        if constexpr (TypeUtils::IsLocalTensorType<S>()) {
            CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(count), 
                VA_ARGS_TO_MAKE_TUPLE(src1));
            SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
                VA_ARGS_TO_MAKE_TUPLE(src1),
                VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));
        }
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_WHERE_WHERE_CHECK_COMMON_H
