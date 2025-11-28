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
 * \file topk_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_SORT_TOPK_TOPK_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_SORT_TOPK_TOPK_CHECK_AICORE_H_

#include "include/adv_api/sort/topk_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false,
    enum TopKMode topkMode = TopKMode::TOPK_NORMAL>
class CheckFuncClassTopKTmpBuf {
public:
    __aicore__ inline CheckFuncClassTopKTmpBuf() {};
    __aicore__ inline CheckFuncClassTopKTmpBuf(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
        const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
        const LocalTensor<uint8_t> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
        const bool isLargest = true) {};
};

template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false,
    enum TopKMode topkMode = TopKMode::TOPK_NORMAL>
class CheckFuncClassTopK {
public:
    __aicore__ inline CheckFuncClassTopK() {};
    __aicore__ inline CheckFuncClassTopK(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
        const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
        const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
        const bool isLargest = true) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_SORT_TOPK_TOPK_CHECK_AICORE_H_
