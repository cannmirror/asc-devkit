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
 * \file kernel_operator_vec_vpadding_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {

template <typename T>
__aicore__ inline void VectorPaddingIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((SupportType<T, half, int16_t, uint16_t, float, int32_t, uint32_t>()), { KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in VectorPadding, current api support dtype combination is src and dst both: half / "
        "int16_t / uint16_t / float / int32_t / uint32_t.");});
    ASCENDC_CHECK_VALUE_RANGE(padMode, 0, 2, "padMode", "VectorPadding");
    if constexpr(sizeof(T) == B16_BYTE_SIZE) {
        vpadding((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src, repeatTime, repeatParams.dstBlkStride,
            repeatParams.srcBlkStride, repeatParams.dstRepStride, repeatParams.srcRepStride,
            repeatParams.repeatStrideMode, repeatParams.strideSizeMode, padMode, padSide);
    } else if constexpr(sizeof(T) == B32_BYTE_SIZE) {
        vpadding((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, repeatTime, repeatParams.dstBlkStride,
            repeatParams.srcBlkStride, repeatParams.dstRepStride, repeatParams.srcRepStride,
            repeatParams.repeatStrideMode, repeatParams.strideSizeMode, padMode, padSide);
    }
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint8_t padMode, const bool padSide,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    VectorPaddingIntrinsicsImpl(dst, src, padMode, padSide, repeatTime, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint8_t padMode, const bool padSide,
    const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    VectorPaddingIntrinsicsImpl(dst, src, padMode, padSide, repeatTime, repeatParams);
}

template <typename T>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint32_t count)
{
    set_mask_count();
    set_vector_mask(0, count);
    UnaryRepeatParams repeatParams;
    VectorPaddingIntrinsicsImpl(dst, src, padMode, padSide, 1, repeatParams);
    set_mask_norm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H