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
 * \file broadcast_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "broadcast_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "broadcast_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void CheckFuncBroadcast(__gm__ const char* apiName, const LocalTensor<T>& dstLocal,
    const LocalTensor<T>& srcLocal, const uint32_t dstShape[dim], const uint32_t srcShape[dim],
    LocalTensor<uint8_t>& sharedTmpBuffer)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassBroadcast<T, dim, axis, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_H
