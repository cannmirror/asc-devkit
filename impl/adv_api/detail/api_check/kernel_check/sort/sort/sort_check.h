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

#ifndef IMPL_API_CHECK_KERNEL_CHECK_SORT_SORT_SORT_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_SORT_SORT_SORT_CHECK_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3002)
#include "sort_check_common.h"
#else
#include "sort_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isFullSort, typename... Args>
__aicore__ inline void CheckFuncSort(__gm__ const char* apiName, Args... args)
{
    CheckFuncClassSort<T, isFullSort> checkFun(apiName);
    checkFun.VerifyingParameters(args...);
}
}  // namespace HighLevelApiCheck
}  // namespace AscendC
#endif  // IMPL_API_CHECK_KERNEL_CHECK_SORT_SORT_SORT_CHECK_H_
