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
 * \file pad_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_PAD_PAD_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_PAD_PAD_CHECK_H

#include "kernel_tiling/kernel_tiling.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "pad_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "pad_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T>
__aicore__ inline void CheckFuncPad(__gm__ const char* apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor, PadParams& padParams, const LocalTensor<uint8_t>& sharedTmpBuffer,
    PadTiling& tiling)
{
#if (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)) || defined(__DAV_C310__)             \
    || defined(__DAV_310R6__)
    CheckFuncClassPad<T> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, padParams, sharedTmpBuffer, tiling);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_PAD_PAD_CHECK_H
