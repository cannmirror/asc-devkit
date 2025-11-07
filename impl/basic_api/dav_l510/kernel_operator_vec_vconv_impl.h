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
 * \file kernel_operator_vec_vconv_impl.h
 * \brief AscendC l510 support vector cast api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator.h"

namespace AscendC {
// Cast::Level 2
template <typename T, typename U>
__aicore__ inline void CastImpl(__ubuf__ T* dst, __ubuf__ U* src, const RoundMode& roundMode,
    const uint32_t count)
{
    ASCENDC_ASSERT(
        (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
}

// Cast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ T* dst, __ubuf__ U* src, const RoundMode& roundMode,
    const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(
        (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
}

// Cast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ T* dst, __ubuf__ U* src, const RoundMode& roundMode,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false),
        { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
}

template <typename T, typename U, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ T* dst, __ubuf__ U* src,
    const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, typename U, bool isSetMask, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ T* dst, __ubuf__ U* src,
    const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, typename U, bool isSetMask, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ T* dst, __ubuf__ U* src,
    const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// AddReluCast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// AddReluCast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// AddReluCast::Level 2
template <typename T, typename U>
__aicore__ inline void AddReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// SubReluCast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// SubReluCast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

// SubReluCast::Level 2
template <typename T, typename U>
__aicore__ inline void SubReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint32_t count)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on current device");
}

__aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on current device");
}

template <typename T>
__aicore__ inline void SetDeqScaleImpl(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on current device");
}

template<typename T>
__aicore__ inline void SetDeqScaleImpl(T config)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on current device");
}
}
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
