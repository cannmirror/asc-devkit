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
 * \file init_global_memory_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "init_global_memory_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "init_global_memory_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T>
__aicore__ inline void CheckFuncInitGlobalMemory(
    __gm__ const char* apiName, GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassInitGlobalMemory<T> checkFun(apiName);
    checkFun.VerifyingParameters(gmWorkspaceAddr, size, value);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_CHECK_H
