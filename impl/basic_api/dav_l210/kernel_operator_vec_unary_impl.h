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
 * \brief AscendC l210 support vector unary memory base api.
 */

#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {

#define UNARY_VEC_COUNTER_NOT_SUPPORT(FUNC_NAME)                                                                \
    template <typename T>                                                                                       \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& count)                 \
    {                                                                                                           \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported"); });             \
    }                                                                                                           \

// for counter level-2
#define UNARY_VEC_COUNTER(FUNC_NAME, OP_NAME, DATA_TYPE, REG_TYPE, BIT_WIDTH)                                   \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t& count)     \
{                                                                                                               \
    __VEC_SCOPE__ {                                                                                             \
        for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                               \
            REG_TYPE vreg0;                                                                                     \
            REG_TYPE vreg1;                                                                                     \
            vector_bool preg;                                                                                   \
            vector_address offset;                                                                              \
            preg = vpd_b##BIT_WIDTH();                                                                          \
            offset = vag_b##BIT_WIDTH(VECTOR_REG_WIDTH / B##BIT_WIDTH##_BYTE_SIZE);                             \
            vld(vreg0, src, offset, NORM);                                                                      \
            OP_NAME(vreg1, vreg0, preg);                                                                        \
            vst(vreg1, dst, offset, NORM_B##BIT_WIDTH, preg);                                                   \
        }                                                                                                       \
    }                                                                                                           \
}                                                                                                               \

/* **************************************************************************************************
 * Abs                                             *
 * ************************************************************************************************* */
// Abs::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(AbsImpl);

UNARY_VEC_COUNTER(AbsImpl, vabs, int8_t, vector_s8, 8);
UNARY_VEC_COUNTER(AbsImpl, vabs, int16_t, vector_s16, 16);
UNARY_VEC_COUNTER(AbsImpl, vabs, int32_t, vector_s32, 32);
UNARY_VEC_COUNTER(AbsImpl, vabs, half, vector_f16, 16);
UNARY_VEC_COUNTER(AbsImpl, vabs, float, vector_f32, 32);

/* **************************************************************************************************
 * Relu                                             *
 * ************************************************************************************************* */
// Relu::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ReluImpl);

UNARY_VEC_COUNTER(ReluImpl, vrelu, int32_t, vector_s32, 32);
UNARY_VEC_COUNTER(ReluImpl, vrelu, half, vector_f16, 16);
UNARY_VEC_COUNTER(ReluImpl, vrelu, float, vector_f32, 32);

/* **************************************************************************************************
 * Exp                                             *
 * ************************************************************************************************* */
// Exp::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ExpImpl);

UNARY_VEC_COUNTER(ExpImpl, vexp, half, vector_f16, 16);
UNARY_VEC_COUNTER(ExpImpl, vexp, float, vector_f32, 32);

/* **************************************************************************************************
 * Sqrt                                             *
 * ************************************************************************************************* */
// Sqrt::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(SqrtImpl);

UNARY_VEC_COUNTER(SqrtImpl, vsqrt, half, vector_f16, 16);
UNARY_VEC_COUNTER(SqrtImpl, vsqrt, float, vector_f32, 32);

/* **************************************************************************************************
 * Reciprocal                                            *
 * ************************************************************************************************* */
// Reciprocal::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(ReciprocalImpl);

UNARY_VEC_COUNTER(ReciprocalImpl, vrec, half, vector_f16, 16);
UNARY_VEC_COUNTER(ReciprocalImpl, vrec, float, vector_f32, 32);

/* **************************************************************************************************
 * Rsqrt                                            *
 * ************************************************************************************************* */
// Rsqrt::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(RsqrtImpl);

UNARY_VEC_COUNTER(RsqrtImpl, vrsqrt, half, vector_f16, 16);
UNARY_VEC_COUNTER(RsqrtImpl, vrsqrt, float, vector_f32, 32);

/* **************************************************************************************************
 * Ln                                            *
 * ************************************************************************************************* */
// Ln::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(LnImpl);

UNARY_VEC_COUNTER(LnImpl, vln, half, vector_f16, 16);
UNARY_VEC_COUNTER(LnImpl, vln, float, vector_f32, 32);

/* **************************************************************************************************
 * Not                                            *
 * ************************************************************************************************* */
// Not::Level 2
UNARY_VEC_COUNTER_NOT_SUPPORT(NotImpl);

UNARY_VEC_COUNTER(NotImpl, vnot, uint8_t, vector_u8, 8);
UNARY_VEC_COUNTER(NotImpl, vnot, int8_t, vector_s8, 8);
UNARY_VEC_COUNTER(NotImpl, vnot, uint16_t, vector_u16, 16);
UNARY_VEC_COUNTER(NotImpl, vnot, int16_t, vector_s16, 16);
UNARY_VEC_COUNTER(NotImpl, vnot, uint32_t, vector_u32, 32);
UNARY_VEC_COUNTER(NotImpl, vnot, int32_t, vector_s32, 32);
UNARY_VEC_COUNTER(NotImpl, vnot, half, vector_f16, 16);
UNARY_VEC_COUNTER(NotImpl, vnot, float, vector_f32, 32);

} // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H