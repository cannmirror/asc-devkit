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
 * \brief AscendC l310 support vector unary api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {
// Macros for level-0 api with type not support
#define UNARY_VEC_NORMAL_NOT_SUPPORT(FUNC_NAME)                                                                                  \
    template <typename T, bool isSetMask = true>                                                                                 \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask,                                      \
        const uint8_t repeatTime, const UnaryRepeatParams& reapeatParams)                                                       \
    {                                                                                                                            \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                             \
    }                                                                                                                            \

#define UNARY_VEC_BITWISE_NOT_SUPPORT(FUNC_NAME)                                                                                 \
    template <typename T, bool isSetMask = true>                                                                                 \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[2],                                   \
        const uint8_t repeatTime, const UnaryRepeatParams& reapeatParams)                                                       \
    {                                                                                                                            \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                             \
    }                                                                                                                            \

// Macros for level-2 api with type not support
#define UNARY_VEC_COUNTER_NOT_SUPPORT(FUNC_NAME)                                                                                 \
    template <typename T>                                                                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& count)                                  \
    {                                                                                                                            \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                             \
    }                                                                                                                            \

// Macros for level-0 api
// for normal op
#define UNARY_VEC_NORMAL_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                                     \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                                         \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask,                          \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                                            \
    {                                                                                                                                \
        __VEC_SCOPE__                                                                                                                \
        {                                                                                                                            \
            RegTensor<DATA_TYPE> vreg0;                                                                                              \
            RegTensor<DATA_TYPE> vreg1;                                                                                              \
            uint32_t sreg = (uint32_t)mask;                                                                                          \
            MaskReg preg = CreatePredicate<DATA_TYPE>(sreg);                                                                         \
            uint32_t strideConfig0 = (uint32_t)repeatParams.srcBlkStride;                                                            \
            uint32_t repeatStrideConfig0 = (uint32_t)repeatParams.srcRepStride;                                                      \
            uint32_t strideConfig1 = (uint32_t)repeatParams.dstBlkStride;                                                            \
            uint32_t repeatStrideConfig1 = (uint32_t)repeatParams.dstRepStride;                                                      \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                                   \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg0, src, strideConfig0, repeatStrideConfig0, preg);            \
                OP_NAME(vreg1, vreg0, preg);                                                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, strideConfig1, repeatStrideConfig1, preg);            \
            }                                                                                                                        \
        }                                                                                                                            \
    }                                                                                                                                \

// for bit-wise op
#define UNARY_VEC_BITWISE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                                    \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                                     \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const uint64_t mask[2],                   \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                                        \
    {                                                                                                                            \
        if constexpr (isSetMask) {                                                                                               \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                                          \
        }                                                                                                                        \
        __VEC_SCOPE__                                                                                                            \
        {                                                                                                                        \
            RegTensor<DATA_TYPE> vreg0;                                                                                          \
            RegTensor<DATA_TYPE> vreg1;                                                                                          \
            MaskReg preg = MovePredicate<DATA_TYPE>();                                                                           \
            uint32_t strideConfig0 = (uint32_t)repeatParams.srcBlkStride;                                                        \
            uint32_t repeatStrideConfig0 = (uint32_t)repeatParams.srcRepStride;                                                  \
            uint32_t strideConfig1 = (uint32_t)repeatParams.dstBlkStride;                                                        \
            uint32_t repeatStrideConfig1 = (uint32_t)repeatParams.dstRepStride;                                                  \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                               \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg0, src, strideConfig0, repeatStrideConfig0, preg);        \
                OP_NAME(vreg1, vreg0, preg);                                                                                     \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, strideConfig1, repeatStrideConfig1, preg);        \
            }                                                                                                                    \
        }                                                                                                                        \
    }                                                                                                                            \

// for counter level-2 op
#define UNARY_VEC_COUNTER_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t& count)                      \
    {                                                                                                                                \
        __VEC_SCOPE__                                                                                                                \
        {                                                                                                                            \
            RegTensor<DATA_TYPE> vreg0;                                                                                              \
            RegTensor<DATA_TYPE> vreg1;                                                                                              \
            uint32_t sreg = (uint32_t)count;                                                                                      \
            MaskReg preg;                                                                                                            \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(DATA_TYPE));                                                   \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                                \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                                   \
                preg = CreatePredicate<DATA_TYPE>(sreg);                                                                             \
                DataCopy(vreg0, src, i * sregLower);                                                                                 \
                OP_NAME(vreg1, vreg0, preg);                                                                                         \
                DataCopy(dst, vreg1, i * sregLower, preg);                                                                           \
            }                                                                                                                        \
        }                                                                                                                            \
    }                                                                                                                                \

/* **************************************************************************************************
 * Abs                                             *
 * ************************************************************************************************* */
// Abs::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(AbsImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(AbsImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(AbsImpl, Abs, int8_t);
UNARY_VEC_NORMAL_IMPL(AbsImpl, Abs, half);
UNARY_VEC_NORMAL_IMPL(AbsImpl, Abs, float);
UNARY_VEC_NORMAL_IMPL(AbsImpl, Abs, int16_t);
UNARY_VEC_NORMAL_IMPL(AbsImpl, Abs, int32_t);
// bit mode
UNARY_VEC_BITWISE_IMPL(AbsImpl, Abs, half);
UNARY_VEC_BITWISE_IMPL(AbsImpl, Abs, float);
UNARY_VEC_BITWISE_IMPL(AbsImpl, Abs, int16_t);
UNARY_VEC_BITWISE_IMPL(AbsImpl, Abs, int32_t);
// Abs::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(AbsImpl);
UNARY_VEC_COUNTER_IMPL(AbsImpl, Abs, int8_t);
UNARY_VEC_COUNTER_IMPL(AbsImpl, Abs, half);
UNARY_VEC_COUNTER_IMPL(AbsImpl, Abs, float);
UNARY_VEC_COUNTER_IMPL(AbsImpl, Abs, int16_t);
UNARY_VEC_COUNTER_IMPL(AbsImpl, Abs, int32_t);

/* **************************************************************************************************
 * Relu                                             *
 * ************************************************************************************************* */
// Relu::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(ReluImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(ReluImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(ReluImpl, Relu, half);
UNARY_VEC_NORMAL_IMPL(ReluImpl, Relu, float);
UNARY_VEC_NORMAL_IMPL(ReluImpl, Relu, int32_t);
// bit mode
UNARY_VEC_BITWISE_IMPL(ReluImpl, Relu, half);
UNARY_VEC_BITWISE_IMPL(ReluImpl, Relu, float);
UNARY_VEC_BITWISE_IMPL(ReluImpl, Relu, int32_t);
// Relu::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ReluImpl);
UNARY_VEC_COUNTER_IMPL(ReluImpl, Relu, half);
UNARY_VEC_COUNTER_IMPL(ReluImpl, Relu, float);
UNARY_VEC_COUNTER_IMPL(ReluImpl, Relu, int32_t);

/* **************************************************************************************************
 * Exp                                             *
 * ************************************************************************************************* */
// Exp::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(ExpImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(ExpImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(ExpImpl, Exp, half);
UNARY_VEC_NORMAL_IMPL(ExpImpl, Exp, float);
// bit mode
UNARY_VEC_BITWISE_IMPL(ExpImpl, Exp, half);
UNARY_VEC_BITWISE_IMPL(ExpImpl, Exp, float);
// Exp::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ExpImpl);
UNARY_VEC_COUNTER_IMPL(ExpImpl, Exp, half);
UNARY_VEC_COUNTER_IMPL(ExpImpl, Exp, float);

/* **************************************************************************************************
 * Sqrt                                             *
 * ************************************************************************************************* */
// Sqrt::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(SqrtImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(SqrtImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(SqrtImpl, Sqrt, half);
UNARY_VEC_NORMAL_IMPL(SqrtImpl, Sqrt, float);
// bit mode
UNARY_VEC_BITWISE_IMPL(SqrtImpl, Sqrt, half);
UNARY_VEC_BITWISE_IMPL(SqrtImpl, Sqrt, float);
// Sqrt::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(SqrtImpl);
UNARY_VEC_COUNTER_IMPL(SqrtImpl, Sqrt, half);
UNARY_VEC_COUNTER_IMPL(SqrtImpl, Sqrt, float);

/* **************************************************************************************************
 * Rsqrt                                             *
 * ************************************************************************************************* */
// Rsqrt::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(RsqrtImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(RsqrtImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(RsqrtImpl, Rsqrt, half);
UNARY_VEC_NORMAL_IMPL(RsqrtImpl, Rsqrt, float);
// bit mode
UNARY_VEC_BITWISE_IMPL(RsqrtImpl, Rsqrt, half);
UNARY_VEC_BITWISE_IMPL(RsqrtImpl, Rsqrt, float);
// Rsqrt::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(RsqrtImpl);
UNARY_VEC_COUNTER_IMPL(RsqrtImpl, Rsqrt, half);
UNARY_VEC_COUNTER_IMPL(RsqrtImpl, Rsqrt, float);

/* **************************************************************************************************
 * Rec                                             *
 * ************************************************************************************************* */
// Rec::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(ReciprocalImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(ReciprocalImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(ReciprocalImpl, Rec, half);
UNARY_VEC_NORMAL_IMPL(ReciprocalImpl, Rec, float);
// bit mode
UNARY_VEC_BITWISE_IMPL(ReciprocalImpl, Rec, half);
UNARY_VEC_BITWISE_IMPL(ReciprocalImpl, Rec, float);
// Rec::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ReciprocalImpl);
UNARY_VEC_COUNTER_IMPL(ReciprocalImpl, Rec, half);
UNARY_VEC_COUNTER_IMPL(ReciprocalImpl, Rec, float);

/* **************************************************************************************************
 * Ln                                             *
 * ************************************************************************************************* */
// Ln::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(LnImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(LnImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(LnImpl, Ln, half);
UNARY_VEC_NORMAL_IMPL(LnImpl, Ln, float);
// bit mode
UNARY_VEC_BITWISE_IMPL(LnImpl, Ln, half);
UNARY_VEC_BITWISE_IMPL(LnImpl, Ln, float);
// Ln::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(LnImpl);
UNARY_VEC_COUNTER_IMPL(LnImpl, Ln, half);
UNARY_VEC_COUNTER_IMPL(LnImpl, Ln, float);

/* **************************************************************************************************
 * Not                                             *
 * ************************************************************************************************* */
// Not::Level 0
UNARY_VEC_NORMAL_NOT_SUPPORT(NotImpl);
UNARY_VEC_BITWISE_NOT_SUPPORT(NotImpl);
// normal mode
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, uint8_t)
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, int8_t)
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, uint16_t);
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, int16_t);
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, half);
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, float);
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, uint32_t);
UNARY_VEC_NORMAL_IMPL(NotImpl, Not, int32_t);
// bit mode
UNARY_VEC_BITWISE_IMPL(NotImpl, Not, uint16_t);
UNARY_VEC_BITWISE_IMPL(NotImpl, Not, int16_t);
UNARY_VEC_BITWISE_IMPL(NotImpl, Not, half);
UNARY_VEC_BITWISE_IMPL(NotImpl, Not, float);
UNARY_VEC_BITWISE_IMPL(NotImpl, Not, uint32_t);
UNARY_VEC_BITWISE_IMPL(NotImpl, Not, int32_t);
// Not::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(NotImpl);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, uint8_t);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, int8_t);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, uint16_t);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, int16_t);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, half);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, float);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, uint32_t);
UNARY_VEC_COUNTER_IMPL(NotImpl, Not, int32_t);
}
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H