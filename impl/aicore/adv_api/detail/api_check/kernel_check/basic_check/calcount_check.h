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
 * \file calcount_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_BASIC_CHECK_CALCOUNT_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_BASIC_CHECK_CALCOUNT_CHECK_H

#include "basic_check_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

class CalCountCheckFuncBasicClass {
public:
    __aicore__ inline CalCountCheckFuncBasicClass(){};
    __aicore__ inline CalCountCheckFuncBasicClass(__gm__ const char* apiName)
    {
        this->apiName = apiName;
    };

public:
    template <typename T, typename U>
    __aicore__ inline void CalCountVerifyingParameters(
        const uint32_t calCount, __gm__ const char* paraName, T tensorTuple, U tensorStringTuple)
    {
        static_assert((Std::is_tuple_v<T> && Std::is_tuple_v<U>), "Input template T or U is not tuple!");
        static_assert((Std::tuple_size_v<decltype(tensorTuple)> == Std::tuple_size_v<decltype(tensorStringTuple)>),
            "CalCountVerifyingParameters func, tensorTuple.Size != tensorStringTuple.Size !");

        CalCountLoopCheck(calCount, paraName, tensorTuple, tensorStringTuple, tuple_sequence<T>{});
    }

private:
    template <typename T, typename U, size_t... Is>
    __aicore__ inline void CalCountLoopCheck(
        const uint32_t calCount, __gm__ const char* paraName, T checkTensor, U tensorInfo, Std::index_sequence<Is...>)
    {
        (CalCountCheck<decltype(Std::get<Is>(checkTensor)), decltype(Std::get<Is>(tensorInfo))>(
             calCount, paraName, Std::get<Is>(checkTensor), Std::get<Is>(tensorInfo)),
            ...);
    }

    template <typename T, typename U>
    __aicore__ inline void CalCountCheck(
        const uint32_t calCount, __gm__ const char* paraName, const T& checkTensor, const U& tensorInfo)
    {
        ASCENDC_ASSERT((calCount <= checkTensor.GetSize() || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[%s] Failed to check %s, %s is %u, should be less than or equal to %s size %u.",
                apiName, paraName, paraName, calCount, tensorInfo, checkTensor.GetSize());
        });

        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, U>(calCount, paraName, checkTensor, tensorInfo);
        }
    }

    template <typename T, typename U>
    __aicore__ inline void PrintParameters(
        const uint32_t calCount, __gm__ const char* paraName, const T& checkTensor, const U& tensorInfo)
    {
        KERNEL_LOG(KERNEL_INFO, "[%s] The tensor size of %s is %u, %s is %u.", apiName, tensorInfo,
            checkTensor.GetSize(), paraName, calCount);
    }

private:
    __gm__ const char* apiName = nullptr;
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_BASIC_CHECK_CALCOUNT_CHECK_H
