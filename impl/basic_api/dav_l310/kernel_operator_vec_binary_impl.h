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
 * \brief AscendC l310 support vector binary api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_binary_continuous_impl.h"

namespace AscendC {
// for Level 0 binary op
#define BINARY_OP_IMPL_NOT_SUPPORT(FUNC_NAME)                                                                      \
    template <typename T, bool isSetMask = true>                                                                   \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],  \
        const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)                                         \
    {                                                                                                              \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });               \
    }
// for Level 0 binary op
#define BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(FUNC_NAME)                                                      \
    template <typename T, bool isSetMask = true>                                                                   \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,     \
        const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)                                         \
    {                                                                                                              \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });               \
    }
/* **************************************************************************************************
 * bit mask                                         *
 * ************************************************************************************************* */
// Level 0
#define BINARY_OP_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                                        \
    template <typename T, bool isSetMask = true>                                                                             \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src0, __ubuf__ DATA_TYPE* src1,            \
        const uint64_t mask[2],                                                                                              \
        const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)                                                   \
    {                                                                                                                        \
        if constexpr (isSetMask) {                                                                                           \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                                      \
        }                                                                                                                    \
        __VEC_SCOPE__                                                                                                        \
        {                                                                                                                    \
            RegTensor<DATA_TYPE> vreg0;                                                                                      \
            RegTensor<DATA_TYPE> vreg1;                                                                                      \
            RegTensor<DATA_TYPE> vreg2;                                                                                      \
            MaskReg preg = MovePredicate<DATA_TYPE>();                                                                       \
            uint32_t strideConfig0 = (uint32_t)repeatParams.src0BlkStride;                                                   \
            uint32_t repeatStrideConfig0 = (uint32_t)repeatParams.src0RepStride;                                             \
            uint32_t strideConfig1 = (uint32_t)repeatParams.src1BlkStride;                                                   \
            uint32_t repeatStrideConfig1 = (uint32_t)repeatParams.src1RepStride;                                             \
            uint32_t strideConfig2 = (uint32_t)repeatParams.dstBlkStride;                                                    \
            uint32_t repeatStrideConfig2 = (uint32_t)repeatParams.dstRepStride;                                              \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                           \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg0, src0, strideConfig0, repeatStrideConfig0, preg);    \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg1, src1, strideConfig1, repeatStrideConfig1, preg);    \
                OP_NAME(vreg2, vreg0, vreg1, preg);                                                                          \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, vreg2, strideConfig2, repeatStrideConfig2, preg);     \
            }                                                                                                                \
        }                                                                                                                    \
    }

/* **************************************************************************************************
 * continuous mask                                            *
 * ************************************************************************************************* */
// Level 0
#define BINARY_OP_CONTINUOUS_MASK_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                       \
    template <typename T, bool isSetMask = true>                                                                             \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src0, __ubuf__ DATA_TYPE* src1,           \
        const uint64_t mask,                                                                                                \
        const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)                                                  \
    {                                                                                                                       \
        __VEC_SCOPE__                                                                                                       \
        {                                                                                                                   \
            RegTensor<DATA_TYPE> vreg0;                                                                                     \
            RegTensor<DATA_TYPE> vreg1;                                                                                     \
            RegTensor<DATA_TYPE> vreg2;                                                                                     \
            uint32_t sreg = (uint32_t)mask;                                                                                 \
            MaskReg preg = CreatePredicate<DATA_TYPE>(sreg);                                                                \
            uint32_t strideConfig0 = (uint32_t)repeatParams.src0BlkStride;                                                  \
            uint32_t repeatStrideConfig0 = (uint32_t)repeatParams.src0RepStride;                                            \
            uint32_t strideConfig1 = (uint32_t)repeatParams.src1BlkStride;                                                  \
            uint32_t repeatStrideConfig1 = (uint32_t)repeatParams.src1RepStride;                                            \
            uint32_t strideConfig2 = (uint32_t)repeatParams.dstBlkStride;                                                   \
            uint32_t repeatStrideConfig2 = (uint32_t)repeatParams.dstRepStride;                                             \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                          \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg0, src0, strideConfig0, repeatStrideConfig0, preg);   \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg1, src1, strideConfig1, repeatStrideConfig1, preg);   \
                OP_NAME(vreg2, vreg0, vreg1, preg);                                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, vreg2, strideConfig2, repeatStrideConfig2, preg);    \
            }                                                                                                               \
        }                                                                                                                   \
    }
/* **************************************************************************************************
 * Add                                                                                              *
 * **************************************************************************************************/
// Add::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(AddImpl)
BINARY_OP_IMPL(AddImpl, Add, int16_t)
BINARY_OP_IMPL(AddImpl, Add, uint16_t)
BINARY_OP_IMPL(AddImpl, Add, int32_t)
BINARY_OP_IMPL(AddImpl, Add, uint32_t)
BINARY_OP_IMPL(AddImpl, Add, half)
BINARY_OP_IMPL(AddImpl, Add, float)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(AddImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, half)
BINARY_OP_CONTINUOUS_MASK_IMPL(AddImpl, Add, float)
/* **************************************************************************************************
 * Sub                                                                                              *
 * **************************************************************************************************/
// Sub::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(SubImpl)
BINARY_OP_IMPL(SubImpl, Sub, int16_t)
BINARY_OP_IMPL(SubImpl, Sub, uint16_t)
BINARY_OP_IMPL(SubImpl, Sub, int32_t)
BINARY_OP_IMPL(SubImpl, Sub, uint32_t)
BINARY_OP_IMPL(SubImpl, Sub, half)
BINARY_OP_IMPL(SubImpl, Sub, float)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(SubImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, half)
BINARY_OP_CONTINUOUS_MASK_IMPL(SubImpl, Sub, float)

/* **************************************************************************************************
 * Mul                                                                                              *
 * **************************************************************************************************/
// Mul::Level
BINARY_OP_IMPL_NOT_SUPPORT(MulImpl)
BINARY_OP_IMPL(MulImpl, Mul, int16_t)
BINARY_OP_IMPL(MulImpl, Mul, uint16_t)
BINARY_OP_IMPL(MulImpl, Mul, int32_t)
BINARY_OP_IMPL(MulImpl, Mul, uint32_t)
BINARY_OP_IMPL(MulImpl, Mul, half)
BINARY_OP_IMPL(MulImpl, Mul, float)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(MulImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, half)
BINARY_OP_CONTINUOUS_MASK_IMPL(MulImpl, Mul, float)
/* **************************************************************************************************
 * Div                                                                                              *
 * **************************************************************************************************/
// Div::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(DivImpl)
BINARY_OP_IMPL(DivImpl, Div, int16_t)
BINARY_OP_IMPL(DivImpl, Div, uint16_t)
BINARY_OP_IMPL(DivImpl, Div, int32_t)
BINARY_OP_IMPL(DivImpl, Div, uint32_t)
BINARY_OP_IMPL(DivImpl, Div, half)
BINARY_OP_IMPL(DivImpl, Div, float)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(DivImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(DivImpl, Div, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(DivImpl, Div, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(DivImpl, Div, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(DivImpl, Div, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(DivImpl, Div, half)
BINARY_OP_CONTINUOUS_MASK_IMPL(DivImpl, Div, float)

/* **************************************************************************************************
 * Max                                                                                              *
 * **************************************************************************************************/
// Max::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(MaxImpl)
BINARY_OP_IMPL(MaxImpl, Max, int16_t)
BINARY_OP_IMPL(MaxImpl, Max, uint16_t)
BINARY_OP_IMPL(MaxImpl, Max, int32_t)
BINARY_OP_IMPL(MaxImpl, Max, uint32_t)
BINARY_OP_IMPL(MaxImpl, Max, half)
BINARY_OP_IMPL(MaxImpl, Max, float)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(MaxImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, half)
BINARY_OP_CONTINUOUS_MASK_IMPL(MaxImpl, Max, float)
/* **************************************************************************************************
 * Min                                                                                              *
 * **************************************************************************************************/
// Min::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(MinImpl)
BINARY_OP_IMPL(MinImpl, Min, int16_t)
BINARY_OP_IMPL(MinImpl, Min, uint16_t)
BINARY_OP_IMPL(MinImpl, Min, int32_t)
BINARY_OP_IMPL(MinImpl, Min, uint32_t)
BINARY_OP_IMPL(MinImpl, Min, half)
BINARY_OP_IMPL(MinImpl, Min, float)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(MinImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, half)
BINARY_OP_CONTINUOUS_MASK_IMPL(MinImpl, Min, float)
/* **************************************************************************************************
 * And                                                                                              *
 * **************************************************************************************************/
// And::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(AndImpl)
BINARY_OP_IMPL(AndImpl, And, int16_t)
BINARY_OP_IMPL(AndImpl, And, uint16_t)
BINARY_OP_IMPL(AndImpl, And, int32_t)
BINARY_OP_IMPL(AndImpl, And, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(AndImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(AndImpl, And, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AndImpl, And, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AndImpl, And, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AndImpl, And, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AndImpl, And, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(AndImpl, And, uint32_t)
/* **************************************************************************************************
 * Or                                                                                               *
 * **************************************************************************************************/
// Or::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(OrImpl)
BINARY_OP_IMPL(OrImpl, Or, int16_t)
BINARY_OP_IMPL(OrImpl, Or, uint16_t)
BINARY_OP_IMPL(OrImpl, Or, int32_t)
BINARY_OP_IMPL(OrImpl, Or, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(OrImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, uint32_t)

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
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu"); });
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
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu"); });
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
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu"); });
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported MulAddDst"); });
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported MulAddDst"); });
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
    const int32_t &count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported MulAddDst"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAdd"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAdd"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAdd"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported SubRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported SubRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported SubRelu"); });
}
}

#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H