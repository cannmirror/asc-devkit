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
 * \file broadcast_check_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_COMMON_H_

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
class CheckFuncClassBroadcast {
public:
    __aicore__ inline CheckFuncClassBroadcast() {};
    __aicore__ inline CheckFuncClassBroadcast(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
        const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t> &sharedTmpBuffer) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_AICORE_H_
