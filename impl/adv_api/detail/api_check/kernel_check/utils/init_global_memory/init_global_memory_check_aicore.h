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
 * \file init_global_memory_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_AICORE_H_

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T>
class CheckFuncClassInitGlobalMemory {
public:
    __aicore__ inline CheckFuncClassInitGlobalMemory() {};
    __aicore__ inline CheckFuncClassInitGlobalMemory(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(GlobalTensor<T> &gmWorkspaceAddr, const uint64_t size, const T value) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_AICORE_H_
 