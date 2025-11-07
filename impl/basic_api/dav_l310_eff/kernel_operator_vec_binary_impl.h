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
 * \file kernel_operator_vec_binary_impl.h
 * \brief AscendC l310 eff support vector binary api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_binary_continuous_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
// Add::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
// Sub::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
// Mul::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
// Div::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
// Max::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
// Min::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
// And::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
// Or::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "BilinearInterpolation data type is not supported!"); });
}

/* **************************************************************************************************
 * AddRelu                                             *
 * ************************************************************************************************* */
// AddRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
// FusedMulAdd::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulAdd is not supported on this version"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulAdd is not supported on this version"); });
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
// FusedMulAddRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulAddRelu is not supported on this version"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulAddRelu is not supported on this version"); });
}
/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on this version"); });
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on this version"); });
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
// SubRelu::Level 2
template <typename T>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on this version"); });
}

// SubRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on this version"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on this version"); });
}
/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const int32_t &count)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)count;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu on this version"); });
}

// AddDeqRelu::Level 0
template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTime;
    (void)repeatParams;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu on this version"); });
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTime;
    (void)repeatParams;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu on this version"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
