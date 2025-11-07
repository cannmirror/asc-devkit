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
 * \file kernel_operator_vec_unary_impl.h
 * \brief AscendC l311 eff support vector unary api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
// Macros for block & repeat size
#define B8_BLOCK_SIZE 32
#define B16_BLOCK_SIZE 16
#define B32_BLOCK_SIZE 8

#define B8_REPEAT_SIZE 256
#define B16_REPEAT_SIZE 128
#define B32_REPEAT_SIZE 64

// Macros for level-0 api with type not support
#define UNARY_VEC_NORMAL_NOT_SUPPORT(FUNC_NAME)                                                                 \
    template <typename T, bool isSetMask = true>                                                                \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask,                     \
        const uint8_t repeatTime, const UnaryRepeatParams& reapeatParams)                                      \
    {                                                                                                           \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported"); });             \
    }                                                                                                           \

#define UNARY_VEC_BITWISE_NOT_SUPPORT(FUNC_NAME)                                                                \
    template <typename T, bool isSetMask = true>                                                                \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[2],                  \
        const uint8_t repeatTime, const UnaryRepeatParams& reapeatParams)                                      \
    {                                                                                                           \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported"); });             \
    }                                                                                                           \

// Macros for level-2 api with type not support
#define UNARY_VEC_COUNTER_NOT_SUPPORT(FUNC_NAME)                                                                \
    template <typename T>                                                                                       \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& count)                 \
    {                                                                                                           \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported"); });             \
    }                                                                                                           \

// Macros for level-0 api
// for normal 8-bit types: s8, u8
#define UNARY_VEC_NORMAL_T8(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                            \
template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask,         \
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                           \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, " vector calculate is not support on current device!"); });    \
}

// for normal 16-bit types: s16, f16
#define UNARY_VEC_NORMAL_T16(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                           \
template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask,         \
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                           \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, \
    " vector calculate is not support on current device!"); });     \
}

// for normal 32-bit types: s32, f32
#define UNARY_VEC_NORMAL_T32(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                           \
template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask,         \
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                           \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, \
    " vector calculate is not support on current device!"); });     \
}

// for bit-wise 8-bit types: u8, s8. 4个mask，plds用normal连续存。preg长度256.
#define UNARY_VEC_BITWISE_T8(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                           \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask[4],  \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                       \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, " vector calculate is not support on current device!"); });    \
}

// for bit-wise 16-bit types: s16, f16
#define UNARY_VEC_BITWISE_T16(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                          \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask[2],  \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                       \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, " vector calculate is not support on current device!"); });    \
}

// for bit-wise 32-bit types: s32, f32
#define UNARY_VEC_BITWISE_T32(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                          \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask[2],  \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                       \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, " vector calculate is not support on current device!"); });    \
}

// for counter level-2 8-bit data
#define UNARY_VEC_COUNTER_T8(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                           \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t& count)     \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, " vector calculate is not support on current device!"); });    \
}

// for counter level-2 16-bit data
#define UNARY_VEC_COUNTER_T16(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                          \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t& count)     \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, " vector calculate is not support on current device!"); });    \
}

// for counter level-2 32-bit data
#define UNARY_VEC_COUNTER_T32(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE)                                          \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t& count)     \
{                                                                                                               \
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, \
    " vector calculate is not support on current device!"); });     \
}

/* **************************************************************************************************
 * Exp                                             *
 * ************************************************************************************************* */
// Exp::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(ExpImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(ExpImpl);

UNARY_VEC_NORMAL_T16(ExpImpl, vexp, half, vector_f16);
UNARY_VEC_NORMAL_T32(ExpImpl, vexp, float, vector_f32);

UNARY_VEC_BITWISE_T16(ExpImpl, vexp, half, vector_f16);
UNARY_VEC_BITWISE_T32(ExpImpl, vexp, float, vector_f32);

// // Exp::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ExpImpl);

UNARY_VEC_COUNTER_T16(ExpImpl, vexp, half, vector_f16);
UNARY_VEC_COUNTER_T32(ExpImpl, vexp, float, vector_f32);


// /* **************************************************************************************************
//  * Ln                                             *
//  * ************************************************************************************************* */
// // Ln::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(LnImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(LnImpl);

UNARY_VEC_NORMAL_T16(LnImpl, vln, half, vector_f16);
UNARY_VEC_NORMAL_T32(LnImpl, vln, float, vector_f32);

UNARY_VEC_BITWISE_T16(LnImpl, vln, half, vector_f16);
UNARY_VEC_BITWISE_T32(LnImpl, vln, float, vector_f32);

// // Ln::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(LnImpl);

UNARY_VEC_COUNTER_T16(LnImpl, vln, half, vector_f16);
UNARY_VEC_COUNTER_T32(LnImpl, vln, float, vector_f32);


// /* **************************************************************************************************
//  * Abs                                             *
//  * ************************************************************************************************* */
// // Abs::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(AbsImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(AbsImpl);

UNARY_VEC_NORMAL_T8(AbsImpl, vabs, int8_t, vector_s8);
UNARY_VEC_NORMAL_T16(AbsImpl, vabs, half, vector_f16);
UNARY_VEC_NORMAL_T32(AbsImpl, vabs, float, vector_f32);
UNARY_VEC_NORMAL_T16(AbsImpl, vabs, int16_t, vector_s16);
UNARY_VEC_NORMAL_T32(AbsImpl, vabs, int32_t, vector_s32);

UNARY_VEC_BITWISE_T8(AbsImpl, vabs, int8_t, vector_s8);
UNARY_VEC_BITWISE_T16(AbsImpl, vabs, half, vector_f16);
UNARY_VEC_BITWISE_T32(AbsImpl, vabs, float, vector_f32);
UNARY_VEC_BITWISE_T16(AbsImpl, vabs, int16_t, vector_s16);
UNARY_VEC_BITWISE_T32(AbsImpl, vabs, int32_t, vector_s32);

// // Abs::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(AbsImpl);

UNARY_VEC_COUNTER_T8(AbsImpl, vabs, int8_t, vector_s8);
UNARY_VEC_COUNTER_T16(AbsImpl, vabs, half, vector_f16);
UNARY_VEC_COUNTER_T32(AbsImpl, vabs, float, vector_f32);
UNARY_VEC_COUNTER_T16(AbsImpl, vabs, int16_t, vector_s16);
UNARY_VEC_COUNTER_T32(AbsImpl, vabs, int32_t, vector_s32);

// /* **************************************************************************************************
//  * Rec                                             *
//  * ************************************************************************************************* */
// // Rec::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(ReciprocalImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(ReciprocalImpl);

UNARY_VEC_NORMAL_T16(ReciprocalImpl, vrec, half, vector_f16);
UNARY_VEC_NORMAL_T32(ReciprocalImpl, vrec, float, vector_f32);

UNARY_VEC_BITWISE_T16(ReciprocalImpl, vrec, half, vector_f16);
UNARY_VEC_BITWISE_T32(ReciprocalImpl, vrec, float, vector_f32);

// // Rec::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ReciprocalImpl);

UNARY_VEC_COUNTER_T16(ReciprocalImpl, vrec, half, vector_f16);
UNARY_VEC_COUNTER_T32(ReciprocalImpl, vrec, float, vector_f32);


// /* **************************************************************************************************
//  * Sqrt                                             *
//  * ************************************************************************************************* */
// // Sqrt::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(SqrtImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(SqrtImpl);

UNARY_VEC_NORMAL_T16(SqrtImpl, vsqrt, half, vector_f16);
UNARY_VEC_NORMAL_T32(SqrtImpl, vsqrt, float, vector_f32);

UNARY_VEC_BITWISE_T16(SqrtImpl, vsqrt, half, vector_f16);
UNARY_VEC_BITWISE_T32(SqrtImpl, vsqrt, float, vector_f32);

// // Sqrt::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(SqrtImpl);

UNARY_VEC_COUNTER_T16(SqrtImpl, vsqrt, half, vector_f16);
UNARY_VEC_COUNTER_T32(SqrtImpl, vsqrt, float, vector_f32);


// // /* **************************************************************************************************
// //  * Rsqrt                                             *
// //  * ************************************************************************************************* */
// // // Rsqrt::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(RsqrtImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(RsqrtImpl);

UNARY_VEC_NORMAL_T16(RsqrtImpl, vrsqrt, half, vector_f16);
UNARY_VEC_NORMAL_T32(RsqrtImpl, vrsqrt, float, vector_f32);

UNARY_VEC_BITWISE_T16(RsqrtImpl, vrsqrt, half, vector_f16);
UNARY_VEC_BITWISE_T32(RsqrtImpl, vrsqrt, float, vector_f32);

// // Rec::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(RsqrtImpl);

UNARY_VEC_COUNTER_T16(RsqrtImpl, vrsqrt, half, vector_f16);
UNARY_VEC_COUNTER_T32(RsqrtImpl, vrsqrt, float, vector_f32);


// // /* **************************************************************************************************
// //  * Not                                             *
// //  * ************************************************************************************************* */
// // // Not::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(NotImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(NotImpl);

UNARY_VEC_NORMAL_T8(NotImpl, vnot, uint8_t, vector_u8)
UNARY_VEC_NORMAL_T8(NotImpl, vnot, int8_t, vector_s8)
UNARY_VEC_NORMAL_T16(NotImpl, vnot, uint16_t, vector_u16);
UNARY_VEC_NORMAL_T16(NotImpl, vnot, int16_t, vector_s16);
UNARY_VEC_NORMAL_T16(NotImpl, vnot, half, vector_f16);
UNARY_VEC_NORMAL_T32(NotImpl, vnot, float, vector_f32);
UNARY_VEC_NORMAL_T32(NotImpl, vnot, uint32_t, vector_u32);
UNARY_VEC_NORMAL_T32(NotImpl, vnot, int32_t, vector_s32);

UNARY_VEC_BITWISE_T8(NotImpl, vnot, uint8_t, vector_u8)
UNARY_VEC_BITWISE_T8(NotImpl, vnot, int8_t, vector_s8)
UNARY_VEC_BITWISE_T16(NotImpl, vnot, uint16_t, vector_u16);
UNARY_VEC_BITWISE_T16(NotImpl, vnot, int16_t, vector_s16);
UNARY_VEC_BITWISE_T16(NotImpl, vnot, half, vector_f16);
UNARY_VEC_BITWISE_T32(NotImpl, vnot, float, vector_f32);
UNARY_VEC_BITWISE_T32(NotImpl, vnot, uint32_t, vector_u32);
UNARY_VEC_BITWISE_T32(NotImpl, vnot, int32_t, vector_s32);

// // // Not::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(NotImpl);

UNARY_VEC_COUNTER_T8(NotImpl, vnot, uint8_t, vector_u8);
UNARY_VEC_COUNTER_T8(NotImpl, vnot, int8_t, vector_s8);
UNARY_VEC_COUNTER_T16(NotImpl, vnot, uint16_t, vector_u16);
UNARY_VEC_COUNTER_T16(NotImpl, vnot, int16_t, vector_s16);
UNARY_VEC_COUNTER_T16(NotImpl, vnot, half, vector_f16);
UNARY_VEC_COUNTER_T32(NotImpl, vnot, float, vector_f32);
UNARY_VEC_COUNTER_T32(NotImpl, vnot, uint32_t, vector_u32);
UNARY_VEC_COUNTER_T32(NotImpl, vnot, int32_t, vector_s32);


// /* **************************************************************************************************
//  * Relu                                             *
//  * ************************************************************************************************* */
// // Relu::Level 0
// // bit mode
UNARY_VEC_NORMAL_NOT_SUPPORT(ReluImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(ReluImpl);

UNARY_VEC_NORMAL_T16(ReluImpl, vrelu, half, vector_f16);
UNARY_VEC_NORMAL_T32(ReluImpl, vrelu, float, vector_f32);
UNARY_VEC_NORMAL_T32(ReluImpl, vrelu, int32_t, vector_s32);

UNARY_VEC_BITWISE_T16(ReluImpl, vrelu, half, vector_f16);
UNARY_VEC_BITWISE_T32(ReluImpl, vrelu, float, vector_f32);
UNARY_VEC_BITWISE_T32(ReluImpl, vrelu, int32_t, vector_s32);

// // Relu::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ReluImpl);

UNARY_VEC_COUNTER_T16(ReluImpl, vrelu, half, vector_f16);
UNARY_VEC_COUNTER_T32(ReluImpl, vrelu, float, vector_f32);
UNARY_VEC_COUNTER_T32(ReluImpl, vrelu, int32_t, vector_s32);
}
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H