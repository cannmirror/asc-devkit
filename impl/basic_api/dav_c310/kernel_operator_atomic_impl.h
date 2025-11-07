/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_MODULE_OPERATOR_ATOMIC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_ATOMIC_IMPL_H

namespace AscendC {
// atomic_add
template <typename T>
__aicore__ inline T AtomicAddImpl(__gm__ T *address, T value)
{
    static_assert(SupportType<T, uint32_t, int32_t, uint64_t, int64_t, float>(),
        "AtomicAdd only support uint32_t/int32_t/uint64_t/int64_t/float data type on current device!");
    return atomicAdd(address, value);
}

// atomic_max
template <typename T>
__aicore__ inline T AtomicMaxImpl(__gm__ T *address, T value)
{
    static_assert(SupportType<T, uint32_t, int32_t, uint64_t, int64_t, float>(),
        "AtomicMax only support uint32_t/int32_t/uint64_t/int64_t/float data type on current device!");
    return atomicMax(address, value);
}

// atomic_min
template <typename T>
__aicore__ inline T AtomicMinImpl(__gm__ T *address, T value)
{
    static_assert(SupportType<T, uint32_t, int32_t, uint64_t, int64_t, float>(),
        "AtomicMin only support uint32_t/int32_t/uint64_t/int64_t/float data type on current device!");
    return atomicMin(address, value);
}

// atomic_cas
template <typename T>
__aicore__ inline T AtomicCasImpl(__gm__ T *address, T value1, T value2)
{
    static_assert(SupportType<T, uint32_t, uint64_t>(),
        "AtomicCas only support uint32_t/uint64_t data type on current device!");
    return atomicCAS(address, value1, value2);
}

// atomic_exch
template <typename T>
__aicore__ inline T AtomicExchImpl(__gm__ T *address, T value)
{
    static_assert(SupportType<T, uint32_t, uint64_t>(),
        "AtomicExch only support uint32_t/uint64_t data type on current device!");
    return atomicExch(address, value);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_ATOMIC_ADD_IMPL_H