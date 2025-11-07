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
 * \brief AscendC l210 support vector binary scalar memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
// for Level 2 binary scalar op
#define BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(FUNC_NAME)                                                           \
    template <typename T, bool isSetMask>                                                                      \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& count) \
    {                                                                                                          \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });           \
    }

// for Level 2 binary scalar op
#define BINARY_SCALAR_OP_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE, CAST_SCALAR_TYPE, REG_TYPE, BIT_WIDTH)            \
    template <typename T = DATA_TYPE, bool isSetMask>                                                          \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,  \
                                     const int32_t& count)                                                  \
    {                                                                                                          \
        __VEC_SCOPE__                                                                                          \
        {                                                                                                      \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                          \
                REG_TYPE vreg0;                                                                                \
                REG_TYPE vreg1;                                                                                \
                vector_bool preg;                                                                              \
                vector_address offset;                                                                         \
                preg = vpd_b##BIT_WIDTH();                                                                     \
                offset = vag_b##BIT_WIDTH(VECTOR_REG_WIDTH / B##BIT_WIDTH##_BYTE_SIZE);                        \
                vld(vreg0, src, offset, NORM);                                                                 \
                OP_NAME(vreg1, vreg0, (CAST_SCALAR_TYPE)scalarValue, preg);                                    \
                vst(vreg1, dst, offset, NORM_B##BIT_WIDTH, preg);                                              \
            }                                                                                                  \
        }                                                                                                      \
    }

/* **************************************************************************************************
 * Adds                                                                                             *
 * **************************************************************************************************/
// Adds::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(AddsImpl)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, uint8_t, uint8_t, vector_u8, 8)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, int8_t, int8_t, vector_s8, 8)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, uint16_t, uint16_t, vector_u16, 16)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, int16_t, int16_t, vector_s16, 16)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, uint32_t, uint32_t, vector_u32, 32)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, int32_t, int32_t, vector_s32, 32)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, half, half, vector_f16, 16)
BINARY_SCALAR_OP_IMPL(AddsImpl, vadds, float, float, vector_f32, 32)

/* **************************************************************************************************
 * Muls                                                                                             *
 * **************************************************************************************************/
// Muls::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(MulsImpl)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, uint8_t, uint8_t, vector_u8, 8)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, int8_t, int8_t, vector_s8, 8)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, uint16_t, uint16_t, vector_u16, 16)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, int16_t, int16_t, vector_s16, 16)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, int32_t, int32_t, vector_s32, 32)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, half, half, vector_f16, 16)
BINARY_SCALAR_OP_IMPL(MulsImpl, vmuls, float, float, vector_f32, 32)

/* **************************************************************************************************
 * Maxs                                                                                             *
 * **************************************************************************************************/
// Maxs::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(MaxsImpl)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, uint8_t, uint8_t, vector_u8, 8)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, int8_t, int8_t, vector_s8, 8)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, uint16_t, uint16_t, vector_u16, 16)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, int16_t, int16_t, vector_s16, 16)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, uint32_t, uint32_t, vector_u32, 32)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, int32_t, int32_t, vector_s32, 32)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, half, half, vector_f16, 16)
BINARY_SCALAR_OP_IMPL(MaxsImpl, vmaxs, float, float, vector_f32, 32)

/* **************************************************************************************************
 * Mins                                                                                             *
 * **************************************************************************************************/
// Mins::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(MinsImpl)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, uint8_t, uint8_t, vector_u8, 8)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, int8_t, int8_t, vector_s8, 8)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, uint16_t, uint16_t, vector_u16, 16)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, int16_t, int16_t, vector_s16, 16)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, uint32_t, uint32_t, vector_u32, 32)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, int32_t, int32_t, vector_s32, 32)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, half, half, vector_f16, 16)
BINARY_SCALAR_OP_IMPL(MinsImpl, vmins, float, float, vector_f32, 32)

/* **************************************************************************************************
 * LeakyRelu                                                                                        *
 * **************************************************************************************************/
// LeakyRelu::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_IMPL(LeakyReluImpl, vlrelu, half, half, vector_f16, 16)
BINARY_SCALAR_OP_IMPL(LeakyReluImpl, vlrelu, float, float, vector_f32, 32)

/* **************************************************************************************************
 * ShiftLeft                                                                                        *
 * **************************************************************************************************/
// ShiftLeft::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(ShiftLeftImpl)
BINARY_SCALAR_OP_IMPL(ShiftLeftImpl, vshls, uint8_t, int16_t, vector_u8, 8)
BINARY_SCALAR_OP_IMPL(ShiftLeftImpl, vshls, uint16_t, int16_t, vector_u16, 16)
BINARY_SCALAR_OP_IMPL(ShiftLeftImpl, vshls, int16_t, int16_t, vector_s16, 16)
BINARY_SCALAR_OP_IMPL(ShiftLeftImpl, vshls, uint32_t, int16_t, vector_u32, 32)
BINARY_SCALAR_OP_IMPL(ShiftLeftImpl, vshls, int32_t, int16_t, vector_s32, 32)

/* **************************************************************************************************
 * ShiftRight                                                                                        *
 * **************************************************************************************************/
// ShiftRight::Level 2
BINARY_SCALAR_OP_IMPL_NOT_SUPPORT(ShiftRightImpl)
BINARY_SCALAR_OP_IMPL(ShiftRightImpl, vshrs, uint8_t, int16_t, vector_u8, 8)
BINARY_SCALAR_OP_IMPL(ShiftRightImpl, vshrs, uint16_t, int16_t, vector_u16, 16)
BINARY_SCALAR_OP_IMPL(ShiftRightImpl, vshrs, int16_t, int16_t, vector_s16, 16)
BINARY_SCALAR_OP_IMPL(ShiftRightImpl, vshrs, uint32_t, int16_t, vector_u32, 32)
BINARY_SCALAR_OP_IMPL(ShiftRightImpl, vshrs, int32_t, int16_t, vector_s32, 32)
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
