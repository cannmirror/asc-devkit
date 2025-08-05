/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef ATVC_KERNEL_UTILS_H
#define ATVC_KERNEL_UTILS_H

#ifndef __ASCC_HOST__
#include "common/const_def.h"
#include "kernel_operator.h"
namespace ATVC {
__BLOCK_LOCAL__  static AscendC::TPipe g_pipe;
template <AscendC::HardEvent EVENT>
__aicore__ inline void SetEvent(AscendC::HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}
template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncDataQueue()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}

namespace KernelUtils {
template <class T, class U>
struct IsSameV {
};
template <class T>
struct IsSameV<T, T> {
    using Type = T;
};


template <typename DType>
struct GetPromoteType {
};

template <>
struct GetPromoteType<float> {
    using T = float;
};

template <>
struct GetPromoteType<int32_t> {
    using T = int32_t;
};

template <>
struct GetPromoteType<int64_t> {
    using T = int64_t;
};

template <>
struct GetPromoteType<uint8_t> {
    using T = uint8_t;
};


template <>
struct GetPromoteType<half> {
    using T = float;
};

template <>
struct GetPromoteType<bfloat16_t> {
    using T = float;
};

__aicore__ inline int64_t FindNearestPower2(const int64_t value)
{
    if (value == 0) {
        return 0;
    } else if (value <= CONST2) {
        return 1;
    } else if (value <= CONST4) {
        return CONST2;
    } else {
        const int64_t num = value - 1;
        const int64_t pow = CONST63 - clz(num);
        return (1 << pow);
    }
}

__aicore__ inline int64_t CalLog2(int64_t value)
{
    int64_t res = 0;
    while (value > 1) {
        value = value >> 1;
        res++;
    }
    return res;
}
}
}
#endif // __ASCC_HOST__
#endif // ATVC_KERNEL_UTILS_H