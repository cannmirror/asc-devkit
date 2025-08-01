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
 * \file init_global_memory_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_COMMON_H

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/single_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckInitGlobalMemoryParamsClass {
public:
    template <typename T>
    __aicore__ inline void CheckInitGlobalMemoryParams(
        GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
    {
        VerifyingParameters<T>(gmWorkspaceAddr, size, value);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T>(gmWorkspaceAddr, size, value);
        }
    }

private:
    template <typename T>
    __aicore__ inline void VerifyingParameters(GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
    {
        ASCENDC_ASSERT((gmWorkspaceAddr.GetSize() > 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[InitGlobalMemory] Failed to check tensor size of gmWorkspaceAddr, current tensor size is %u, "
                "should be greater than 0.",
                gmWorkspaceAddr.GetSize());
        });

        ASCENDC_ASSERT(((size <= gmWorkspaceAddr.GetSize()) || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
                "[InitGlobalMemory] The value of size is %u, should not be greater than gmWorkspaceAddr size %u", size,
                gmWorkspaceAddr.GetSize());
        });
    }

    template <typename T>
    __aicore__ inline void PrintParameters(GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
    {
        KERNEL_LOG(KERNEL_INFO, "[InitGlobalMemory] The size of gmWorkspaceAddr is %u.", gmWorkspaceAddr.GetSize());
    }
};

template <typename T>
class CheckFuncClassInitGlobalMemory : public DataTypeCheckFuncBasicClass, public CheckInitGlobalMemoryParamsClass {
public:
    __aicore__ inline CheckFuncClassInitGlobalMemory(){};
    __aicore__ inline CheckFuncClassInitGlobalMemory(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
    {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float, uint16_t, int16_t, uint32_t, int32_t>(
            "template parameter (T) is not half/float/uint16_t/int16_t/uint32_t/int32_t");

        CheckInitGlobalMemoryParamsClass::CheckInitGlobalMemoryParams<T>(gmWorkspaceAddr, size, value);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_COMMON_H
