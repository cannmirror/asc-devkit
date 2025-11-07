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
 * \file kernel_operator_vec_reduce_impl.h
 * \brief AscendC l510 support reduce api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H

namespace AscendC {
#define VCPADD_FUNC() vcpadd(vreg1, vreg0, preg, MODE_ZEROING)
#define VCGADD_FUNC() vcgadd(vreg1, vreg0, preg, MODE_ZEROING)
#define VCGMAX_FUNC() vcgmax(vreg1, vreg0, preg, MODE_ZEROING)
#define VCGMIN_FUNC() vcgmin(vreg1, vreg0, preg, MODE_ZEROING)
#define VCMAX_FUNC() vcmax(vreg1, vreg0, preg, MODE_ZEROING)
#define VCMIN_FUNC() vcmin(vreg1, vreg0, preg, MODE_ZEROING)
#define VCADD_FUNC() vcadd(vreg1, vreg0, preg, MODE_ZEROING)

#define CONTINUOUS_MODE_REDUCE_VF(reducefunc, vregType, pltType, dstStrideOffset) \
        ASCENDC_ASSERT((false), {                                                 \
                KERNEL_LOG(KERNEL_ERROR,                                          \
                    " vector calculate is not support on current device!");             \
            })

#define BITBYBIT_MODE_HALF_REDUCE_VF(reducefunc, dstStrideOffset)                 \
        ASCENDC_ASSERT((false), {                                                 \
                KERNEL_LOG(KERNEL_ERROR,                                          \
                    " vector calculate is not support on current device!");             \
            })

#define BITBYBIT_MODE_FLOAT_REDUCE_VF(reducefunc, dstStrideOffset)                \
        ASCENDC_ASSERT((false), {                                                 \
                KERNEL_LOG(KERNEL_ERROR,                                          \
                    " vector calculate is not support on current device!");             \
            })

/* **************************************** Pair Reduce Impl ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime, const uint64_t mask[2],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

/* **************************************** Block Reduce Impl ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void RepeatReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeatTime,
    const int32_t elemsInOneRepeate, const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

/* **************************************** Whole Reduce Interface ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t mask,
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

/* **************************************** Reduce Interface ****************************************** */
template <typename T>
__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ T* work, __ubuf__ T* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxIntrinsicsImpl is not supported!"); });
}

template <typename T>
__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ T* work, __ubuf__ T* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxIntrinsicsImpl is not supported!"); });
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ half* work, __ubuf__ half* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ half* work, __ubuf__ half* src,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ float* work, __ubuf__ float* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ float* work, __ubuf__ float* src,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ T* work, __ubuf__ T* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinIntrinsicsImpl is not supported!"); });
}

template <typename T>
__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ T* work, __ubuf__ T* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinIntrinsicsImpl is not supported!"); });
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ half* work, __ubuf__ half* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ half* work, __ubuf__ half* src,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ float* work, __ubuf__ float* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ float* work, __ubuf__ float* src,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ T* work, __ubuf__ T* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumIntrinsicsImpl is not supported!"); });
}

template <typename T>
__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ T* work, __ubuf__ T* src, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumIntrinsicsImpl is not supported!"); });
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ half* work, __ubuf__ half* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ half* work, __ubuf__ half* src,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ float* work, __ubuf__ float* src, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ float* work, __ubuf__ float* src,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceSumSecondStep(__ubuf__ T* dst, __ubuf__ T* work,
    struct ReduceRepeatParams& params)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void CreateSpecialFormatMask(const int32_t& maskLen, uint64_t& highMask, uint64_t& lowMask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceOperation(__ubuf__ T* work, __ubuf__ T* src, struct ReduceRepeatParams& params,
    const ReduceMode& mode)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplFirstStep(__ubuf__ T* work, __ubuf__ T* src,
    struct ReduceRepeatParams& params, const ReduceMode& mode, int32_t& curData)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplSecondStep(__ubuf__ T* work, const ReduceMode& mode, int32_t& curData,
    int32_t preStartPos, int32_t secondStartPos)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void GetIndex(__ubuf__ T* work, int32_t secondStartPos, int32_t& secondIndex,
    int32_t& thirdIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void GetIndex(__ubuf__ T* work, int32_t secondStartPos, int32_t thirdStartPos,
    int32_t& firstIndex, int32_t& secondIndex, int32_t& thirdIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void GetIndex(__ubuf__ T* work, int32_t secondStartPos, int32_t thirdStartPos,
    int32_t fourthStartPos, int32_t& firstIndex, int32_t& secondIndex, int32_t& thirdIndex, int32_t& fourthIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplThirdStep(__ubuf__ T* dst, __ubuf__ T* work, const int32_t srcRepStride,
    const ReduceMode& mode, int32_t& curData, int32_t& secondStartPos, int32_t& thirdStartPos)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceSumFirstStep(__ubuf__ T* work, __ubuf__ T* src,
    struct ReduceRepeatParams& params)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceSumFinalStep(__ubuf__ T* dst, __ubuf__ T* work, int32_t& secondResultNum)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    struct ReduceRepeatParams& params)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplSecondStepNoIndex(__ubuf__ T* work, const ReduceMode& mode, int32_t& curData)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplThirdStepNoIndex(__ubuf__ T* dst, __ubuf__ T* work, const ReduceMode& mode,
    int32_t& curData)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplWithIndex(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    struct ReduceRepeatParams& params, const ReduceMode& mode)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImplNoIndex(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    struct ReduceRepeatParams& params, const ReduceMode& mode)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    struct ReduceRepeatParams& params, bool calIndex, const ReduceMode& mode)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void ReduceTailCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& work, const int32_t count, bool calIndex, const ReduceMode& mode)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on current device!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue, uint32_t &maxMinIndex)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue, T &maxMinIndex)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline T GetAccValImpl()
{
    ASCENDC_ASSERT((false), "GetAccVal is not supported on current device");
    return 0;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
