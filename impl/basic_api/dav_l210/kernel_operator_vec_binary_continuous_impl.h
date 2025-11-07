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
 * \file kernel_operator_vec_binary_continuous_impl.h
 * \brief AscendC l210 support vec binary continuous data api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {

// for Level 2 binary op
#define BINARY_OP_IMPL_NOT_SUPPORT(FUNC_NAME)                                                                      \
    template <typename T>                                                                                          \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count) \
    {                                                                                                              \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });               \
    }

// for Level 2 binary op
#define BINARY_OP_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE, BIT_WIDTH)                                         \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src0, __ubuf__ DATA_TYPE* src1,  \
                                     const int32_t& count)                                                      \
    {                                                                                                              \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                              \
                REG_TYPE vreg0;                                                                                    \
                REG_TYPE vreg1;                                                                                    \
                REG_TYPE vreg2;                                                                                    \
                vector_bool preg;                                                                                  \
                vector_address offset;                                                                             \
                preg = vpd_b##BIT_WIDTH();                                                                         \
                offset = vag_b##BIT_WIDTH(VECTOR_REG_WIDTH / B##BIT_WIDTH##_BYTE_SIZE);                            \
                vld(vreg0, src0, offset, NORM);                                                                    \
                vld(vreg1, src1, offset, NORM);                                                                    \
                OP_NAME(vreg2, vreg0, vreg1, preg);                                                                \
                vst(vreg2, dst, offset, NORM_B##BIT_WIDTH, preg);                                                  \
            }                                                                                                      \
        }                                                                                                          \
    }

/* **************************************************************************************************
 * Add                                                                                              *
 * **************************************************************************************************/
// Add::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(AddImpl)
BINARY_OP_IMPL(AddImpl, vadd, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(AddImpl, vadd, int8_t, vector_s8, 8)
BINARY_OP_IMPL(AddImpl, vadd, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(AddImpl, vadd, int16_t, vector_s16, 16)
BINARY_OP_IMPL(AddImpl, vadd, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(AddImpl, vadd, int32_t, vector_s32, 32)
BINARY_OP_IMPL(AddImpl, vadd, half, vector_f16, 16)
BINARY_OP_IMPL(AddImpl, vadd, float, vector_f32, 32)

/* **************************************************************************************************
 * Sub                                                                                              *
 * **************************************************************************************************/
// Sub::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(SubImpl)
BINARY_OP_IMPL(SubImpl, vsub, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(SubImpl, vsub, int8_t, vector_s8, 8)
BINARY_OP_IMPL(SubImpl, vsub, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(SubImpl, vsub, int16_t, vector_s16, 16)
BINARY_OP_IMPL(SubImpl, vsub, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(SubImpl, vsub, int32_t, vector_s32, 32)
BINARY_OP_IMPL(SubImpl, vsub, half, vector_f16, 16)
BINARY_OP_IMPL(SubImpl, vsub, float, vector_f32, 32)

/* **************************************************************************************************
 * Mul                                                                                              *
 * **************************************************************************************************/
// Mul::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(MulImpl)
BINARY_OP_IMPL(MulImpl, vmul, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(MulImpl, vmul, int8_t, vector_s8, 8)
BINARY_OP_IMPL(MulImpl, vmul, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(MulImpl, vmul, int16_t, vector_s16, 16)
BINARY_OP_IMPL(MulImpl, vmul, int32_t, vector_s32, 32)
BINARY_OP_IMPL(MulImpl, vmul, half, vector_f16, 16)
BINARY_OP_IMPL(MulImpl, vmul, float, vector_f32, 32)

/* **************************************************************************************************
 * Div                                                                                              *
 * **************************************************************************************************/
// Div::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(DivImpl)
BINARY_OP_IMPL(DivImpl, vdiv, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(DivImpl, vdiv, int16_t, vector_s16, 16)
BINARY_OP_IMPL(DivImpl, vdiv, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(DivImpl, vdiv, int32_t, vector_s32, 32)
BINARY_OP_IMPL(DivImpl, vdiv, half, vector_f16, 16)
BINARY_OP_IMPL(DivImpl, vdiv, float, vector_f32, 32)

/* **************************************************************************************************
 * Max                                                                                              *
 * **************************************************************************************************/
// Max::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(MaxImpl)
BINARY_OP_IMPL(MaxImpl, vmax, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(MaxImpl, vmax, int8_t, vector_s8, 8)
BINARY_OP_IMPL(MaxImpl, vmax, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(MaxImpl, vmax, int16_t, vector_s16, 16)
BINARY_OP_IMPL(MaxImpl, vmax, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(MaxImpl, vmax, int32_t, vector_s32, 32)
BINARY_OP_IMPL(MaxImpl, vmax, half, vector_f16, 16)
BINARY_OP_IMPL(MaxImpl, vmax, float, vector_f32, 32)

/* **************************************************************************************************
 * Min                                                                                              *
 * **************************************************************************************************/
// Min::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(MinImpl)
BINARY_OP_IMPL(MinImpl, vmin, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(MinImpl, vmin, int8_t, vector_s8, 8)
BINARY_OP_IMPL(MinImpl, vmin, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(MinImpl, vmin, int16_t, vector_s16, 16)
BINARY_OP_IMPL(MinImpl, vmin, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(MinImpl, vmin, int32_t, vector_s32, 32)
BINARY_OP_IMPL(MinImpl, vmin, half, vector_f16, 16)
BINARY_OP_IMPL(MinImpl, vmin, float, vector_f32, 32)

/* **************************************************************************************************
 * And                                                                                              *
 * **************************************************************************************************/
// And::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(AndImpl)
BINARY_OP_IMPL(AndImpl, vand, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(AndImpl, vand, int8_t, vector_s8, 8)
BINARY_OP_IMPL(AndImpl, vand, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(AndImpl, vand, int16_t, vector_s16, 16)
BINARY_OP_IMPL(AndImpl, vand, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(AndImpl, vand, int32_t, vector_s32, 32)
BINARY_OP_IMPL(AndImpl, vand, half, vector_f16, 16)
BINARY_OP_IMPL(AndImpl, vand, float, vector_f32, 32)

/* **************************************************************************************************
 * Or                                                                                               *
 * **************************************************************************************************/
// Or::Level 2
BINARY_OP_IMPL_NOT_SUPPORT(OrImpl)
BINARY_OP_IMPL(OrImpl, vor, uint8_t, vector_u8, 8)
BINARY_OP_IMPL(OrImpl, vor, int8_t, vector_s8, 8)
BINARY_OP_IMPL(OrImpl, vor, uint16_t, vector_u16, 16)
BINARY_OP_IMPL(OrImpl, vor, int16_t, vector_s16, 16)
BINARY_OP_IMPL(OrImpl, vor, uint32_t, vector_u32, 32)
BINARY_OP_IMPL(OrImpl, vor, int32_t, vector_s32, 32)
BINARY_OP_IMPL(OrImpl, vor, half, vector_f16, 16)
BINARY_OP_IMPL(OrImpl, vor, float, vector_f32, 32)

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
