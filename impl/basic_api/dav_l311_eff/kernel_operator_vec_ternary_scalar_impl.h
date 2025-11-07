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

/*!
 * \file kernel_operator_vec_ternary_scalar_impl.h
 * \brief AscendC v311 eff support vaxpy level 0/2 api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_operator_common_impl.h"
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
                                          uint64_t mask, const uint8_t repeatTime,
                                          const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
                                          uint64_t mask, const uint8_t repeatTime,
                                          const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
                                          uint64_t mask[2], const uint8_t repeatTime,
                                          const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
                                          uint64_t mask[2], const uint8_t repeatTime,
                                          const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
                                          const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
                                          const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    uint64_t mask, const uint8_t repeatTime,
                                    const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    uint64_t mask[2], const uint8_t repeatTime,
                                    const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// Axpy::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                uint64_t mask[2], const uint8_t repeatTime,
                                const UnaryRepeatParams& repeatParams)
{
    if constexpr (sizeof(T) == sizeof(U)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) > sizeof(U)) {
        return AxpyFmixImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                uint64_t mask, const uint8_t repeatTime,
                                const UnaryRepeatParams& repeatParams)
{
    if constexpr (sizeof(T) == sizeof(U)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) > sizeof(U)) {
        return AxpyFmixImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

// Add::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                const int32_t& count)
{
    if constexpr (sizeof(T) == sizeof(U)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, count);
    } else if constexpr (sizeof(T) > sizeof(U)) {
        return AxpyFmixImpl(dst, src, scalarValue, count);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}
}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H