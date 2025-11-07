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
 * \file kernel_operator_vec_binary_scalar_impl.h
 * \brief AscendC l310 eff support vector binary scalar api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
// Adds::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// Adds::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
// Muls::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// Muls::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
// Maxs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// Maxs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
// Mins::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// Mins::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
// ShiftLeft::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
// ShiftRight::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
// LeakyRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],
    uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

// LeakyRelu::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
