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
 * \brief AscendC l311 support vector binary scalar api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {
#define BIT_BY_BIT_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTime, repeatParams)                 \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<DATA_TYPE> srcReg, dstReg;                                                                    \
            MaskReg preg = MovePredicate<DATA_TYPE>();                                                              \
            uint32_t srcBlkStride = (uint32_t)(repeatParams.srcBlkStride);                                          \
            uint32_t dstBlkStride = (uint32_t)(repeatParams.dstBlkStride);                                          \
            uint32_t srcRepStride = (uint32_t)(repeatParams.srcRepStride);                                          \
            uint32_t dstRepStride = (uint32_t)(repeatParams.dstRepStride);                                          \
            for (uint16_t i = 0; i < (uint16_t)(repeatTime); ++i) {                                                \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(srcReg, src, srcBlkStride, srcRepStride, preg);  \
                OP_NAME(dstReg, srcReg, scalarValue, preg);                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, dstReg, dstBlkStride, dstRepStride, preg);  \
            }                                                                                                       \
        }                                                                                                           \

#define CONTINUOUS_MODE_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTime, repeatParams)            \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<DATA_TYPE> srcReg, dstReg;                                                                    \
            uint32_t sreg = (uint32_t)(mask);                                                                       \
            MaskReg preg = CreatePredicate<DATA_TYPE>(sreg);                                                        \
            uint32_t srcBlkStride = (uint32_t)(repeatParams.srcBlkStride);                                          \
            uint32_t dstBlkStride = (uint32_t)(repeatParams.dstBlkStride);                                          \
            uint32_t srcRepStride = (uint32_t)(repeatParams.srcRepStride);                                          \
            uint32_t dstRepStride = (uint32_t)(repeatParams.dstRepStride);                                          \
            for (uint16_t i = 0; i < (uint16_t)(repeatTime); ++i) {                                                \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(srcReg, src, srcBlkStride, srcRepStride, preg);  \
                OP_NAME(dstReg, srcReg, scalarValue, preg);                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, dstReg, dstBlkStride, dstRepStride, preg);  \
            }                                                                                                       \
        }

// for Level 0 bit-by-bit mode binary scalar op
#define BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                                         \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],       \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                           \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                              \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],       \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false)                     \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }


#define BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                 \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                   \
    {                                                                                                               \
        if constexpr (isSetMask) {                                                                                  \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                             \
        }                                                                                                           \
        BIT_BY_BIT_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTime, repeatParams);                \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                      \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask[2], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)     \
    {                                                                                                               \
        if constexpr (isSetMask) {                                                                                  \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                             \
        }                                                                                                           \
        BIT_BY_BIT_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTime, repeatParams);                \
    }

// for Level 0 continuous mode binary scalar op
#define BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                                         \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,          \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                                           \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                              \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,          \
        const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false)                     \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }

#define BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                 \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                      \
    {                                                                                                               \
        CONTINUOUS_MODE_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTime, repeatParams)            \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                      \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)        \
    {                                                                                                               \
        CONTINUOUS_MODE_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTime, repeatParams)            \
    }

// for Level 2 binary scalar op
#define BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(FUNC_NAME)                                                         \
    template <typename T, bool isSetMask>                                                                           \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count)      \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }


#define BINARY_SCALAR_OP_LEVEL2_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                 \
    template <typename T = DATA_TYPE, bool isSetMask>                                                               \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
                                     const int32_t& count)                                                       \
    {                                                                                                               \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<DATA_TYPE> srcReg, dstReg;                                                                    \
            MaskReg preg;                                                                                           \
            uint32_t sreg = (uint32_t)count;                                                                     \
            constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(DATA_TYPE));                        \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                               \
            for (uint16_t i = 0; i < repeatTime; ++i) {                                                            \
                preg = CreatePredicate<DATA_TYPE>(sreg);                                                            \
                DataCopy(srcReg, src, i * sregLower);                                                               \
                OP_NAME(dstReg, srcReg, scalarValue, preg);                                                         \
                DataCopy(dst, dstReg, i * sregLower, preg);                                                         \
            }                                                                                                       \
        }                                                                                                           \
    }

/* **************************************************************************************************
 * Adds                                                                                             *
 * **************************************************************************************************/
// Adds::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(AddsImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(AddsImpl, Adds, uint16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(AddsImpl, Adds, int16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(AddsImpl, Adds, uint32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(AddsImpl, Adds, int32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(AddsImpl, Adds, half)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(AddsImpl, Adds, float)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(AddsImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, uint8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, int8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, uint16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, int16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, uint32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, int32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, half)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(AddsImpl, Adds, float)

// Adds::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(AddsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, float)

/* **************************************************************************************************
 * Muls                                                                                             *
 * **************************************************************************************************/
// Muls::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(MulsImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MulsImpl, Muls, uint16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MulsImpl, Muls, int16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MulsImpl, Muls, uint32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MulsImpl, Muls, int32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MulsImpl, Muls, half)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MulsImpl, Muls, float)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(MulsImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, uint8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, int8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, uint16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, int16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, uint32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, int32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, half)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MulsImpl, Muls, float)

// Muls::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(MulsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, float)

/* **************************************************************************************************
 * Maxs                                                                                             *
 * **************************************************************************************************/
// Maxs::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(MaxsImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MaxsImpl, Maxs, uint16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MaxsImpl, Maxs, int16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MaxsImpl, Maxs, uint32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MaxsImpl, Maxs, int32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MaxsImpl, Maxs, half)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MaxsImpl, Maxs, float)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(MaxsImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, uint8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, int8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, uint16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, int16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, uint32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, int32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, half)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MaxsImpl, Maxs, float)

// Maxs::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(MaxsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, float)

/* **************************************************************************************************
 * Mins                                                                                             *
 * **************************************************************************************************/
// Mins::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(MinsImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MinsImpl, Mins, uint16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MinsImpl, Mins, int16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MinsImpl, Mins, uint32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MinsImpl, Mins, int32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MinsImpl, Mins, half)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(MinsImpl, Mins, float)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(MinsImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, uint8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, int8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, uint16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, int16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, uint32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, int32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, half)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(MinsImpl, Mins, float)

// Mins::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(MinsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, float)

/* **************************************************************************************************
 * LeakyRelu                                                                                        *
 * **************************************************************************************************/
// LeakyRelu::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(LeakyReluImpl, LeakyRelu, half)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(LeakyReluImpl, LeakyRelu, float)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(LeakyReluImpl, LeakyRelu, half)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(LeakyReluImpl, LeakyRelu, float)

// LeakyRelu::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(LeakyReluImpl, LeakyRelu, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(LeakyReluImpl, LeakyRelu, float)

/* **************************************************************************************************
 * ShiftLeft                                                                                        *
 * **************************************************************************************************/
// ShiftLeft::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(ShiftLeftImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftLeftImpl, ShiftLefts, uint16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftLeftImpl, ShiftLefts, int16_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftLeftImpl, ShiftLefts, uint32_t)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftLeftImpl, ShiftLefts, int32_t)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(ShiftLeftImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftLeftImpl, ShiftLefts, uint8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftLeftImpl, ShiftLefts, int8_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftLeftImpl, ShiftLefts, uint16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftLeftImpl, ShiftLefts, int16_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftLeftImpl, ShiftLefts, uint32_t)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftLeftImpl, ShiftLefts, int32_t)

// ShiftLeft::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(ShiftLeftImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, int32_t)

/* **************************************************************************************************
 * ShiftRight                                                                                       *
 * **************************************************************************************************/
// ShiftRight::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(ShiftRightImpl)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftRightImpl, ShiftRights, uint16_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftRightImpl, ShiftRights, int16_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftRightImpl, ShiftRights, uint32_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL(ShiftRightImpl, ShiftRights, int32_t)
// continuous mode
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(ShiftRightImpl)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftRightImpl, ShiftRights, uint8_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftRightImpl, ShiftRights, int8_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftRightImpl, ShiftRights, uint16_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftRightImpl, ShiftRights, int16_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftRightImpl, ShiftRights, uint32_t)
BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(ShiftRightImpl, ShiftRights, int32_t)

// ShiftRight::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(ShiftRightImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, int32_t)

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
