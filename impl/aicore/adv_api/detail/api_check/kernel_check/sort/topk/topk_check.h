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
 * \file topk_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_TOPK_TOPK_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_TOPK_TOPK_CHECK_H

#include "kernel_tiling/kernel_tiling.h"
#include "sort/topk_utils.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "topk_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "topk_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false,
    enum TopKMode topkMode = TopKMode::TOPK_NORMAL>
__aicore__ inline void CheckFuncTopK(__gm__ const char* apiName, const LocalTensor<T>& dstValueLocal,
    const LocalTensor<int32_t>& dstIndexLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<int32_t>& srcIndexLocal, const LocalTensor<bool>& finishLocal,
    const LocalTensor<uint8_t>& tmpLocal, const int32_t k, const TopkTiling& tilling, const TopKInfo& topKInfo,
    const bool isLargest = true)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassTopKTmpBuf<T, isInitIndex, isHasfinish, isReuseSrc, topkMode> checkFun(apiName);
    checkFun.VerifyingParameters(
        dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal, finishLocal, tmpLocal, k, tilling, topKInfo, isLargest);
#endif
}

template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false,
    enum TopKMode topkMode = TopKMode::TOPK_NORMAL>
__aicore__ inline void CheckFuncTopK(__gm__ const char* apiName, const LocalTensor<T>& dstValueLocal,
    const LocalTensor<int32_t>& dstIndexLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<int32_t>& srcIndexLocal, const LocalTensor<bool>& finishLocal, const int32_t k,
    const TopkTiling& tilling, const TopKInfo& topKInfo, const bool isLargest = true)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassTopK<T, isInitIndex, isHasfinish, isReuseSrc, topkMode> checkFun(apiName);
    checkFun.VerifyingParameters(
        dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal, finishLocal, k, tilling, topKInfo, isLargest);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_TOPK_TOPK_CHECK_H
