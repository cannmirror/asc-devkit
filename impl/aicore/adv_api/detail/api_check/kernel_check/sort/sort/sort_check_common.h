/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_SORT_SORT_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_SORT_SORT_CHECK_COMMON_H

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/multiple_tensor_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckSortParamsClass {
public:
    __aicore__ inline CheckSortParamsClass() {}
    __aicore__ inline CheckSortParamsClass(__gm__ const char* apiName)
    {
        this->apiName = apiName;
    }

public:
    template <typename T, bool isFullSort>
    __aicore__ inline void CheckSortParams(const LocalTensor<T>& dst, const LocalTensor<T>& concat,
        const LocalTensor<uint32_t>& index, const LocalTensor<T>& tmp, const int32_t repeatTime)
    {
        VerifyingParameters<T, isFullSort>(dst, concat, index, tmp, repeatTime);

        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, isFullSort>(dst, concat, index, tmp, repeatTime);
        }
    }

private:
    template <typename T, bool isFullSort>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const LocalTensor<T>& concat,
        const LocalTensor<uint32_t>& index, const LocalTensor<T>& tmp, const int32_t repeatTime)
    {
        ASCENDC_ASSERT(((repeatTime >= 0 && repeatTime <= MAX_REPEAT_TIMES) || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[%s] The repeatTime is %d, should in range [0, %u].", this->apiName, repeatTime,
                MAX_REPEAT_TIMES);
        });
    }

    template <typename T, bool isFullSort>
    __aicore__ inline void PrintParameters(const LocalTensor<T>& dst, const LocalTensor<T>& concat,
        const LocalTensor<uint32_t>& index, const LocalTensor<T>& tmp, const int32_t repeatTime)
    {
        KERNEL_LOG(KERNEL_INFO, "[%s] The repeatTime is %d.", this->apiName, repeatTime);
    }

private:
    __gm__ const char* apiName = nullptr;
};

template <typename T, bool isFullSort>
class CheckFuncClassSort : public DataTypeCheckFuncBasicClass,
                           public ReuseSourceCheckFuncBasicClass,
                           public SingleTensorCheckFuncBasicClass,
                           public MultipleTensorCheckFuncBasicClass,
                           public CheckSortParamsClass {
public:
    __aicore__ inline CheckFuncClassSort() {}
    __aicore__ inline CheckFuncClassSort(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName),
        CheckSortParamsClass(apiName)
    {}

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const LocalTensor<T>& concat,
        const LocalTensor<uint32_t>& index, const LocalTensor<T>& tmp, const int32_t repeatTime)
    {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>("template parameter (T) is not half "
                                                                                 "or float");

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(dst, concat, index, tmp),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CheckSortParamsClass::CheckSortParams<T, isFullSort>(dst, concat, index, tmp, repeatTime);
    }
};

} // namespace HighLevelApiCheck
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_SORT_SORT_CHECK_COMMON_H