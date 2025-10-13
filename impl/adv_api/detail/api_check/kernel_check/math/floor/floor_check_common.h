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
 * \file floor_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_FLOOR_FLOOR_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_FLOOR_FLOOR_CHECK_COMMON_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/calcount_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"
#include "../math_common_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassFloor : public CheckFuncClassMathCommon {
public:
    __aicore__ inline CheckFuncClassFloor() {};
    __aicore__ inline CheckFuncClassFloor(__gm__ const char *apiName) : CheckFuncClassMathCommon(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount) {
        CheckFuncClassMathCommon::CommonVerifyingParameters<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
    };
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassFloorNoTmpBuffer : public DataTypeCheckFuncBasicClass, public CalCountCheckFuncBasicClass,
    public ReuseSourceCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass, public MultipleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassFloorNoTmpBuffer() {};
    __aicore__ inline CheckFuncClassFloorNoTmpBuffer(__gm__ const char *apiName) :
        DataTypeCheckFuncBasicClass(apiName), CalCountCheckFuncBasicClass(apiName),
        ReuseSourceCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const uint32_t calCount) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "template parameter (T) is not half or float");

        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(calCount), VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(srcTensor, dstTensor),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, srcTensor));
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_FLOOR_FLOOR_CHECK_COMMON_H_
