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
 * \file kernel_operator_vec_ternary_scalar_impl.h
 * \brief AscendC l210 support vector ternary scalar memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * Axpy                                             *
 * ************************************************************************************************* */
// Axpy::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

#define AXPY_OP_IMPL(DATA_TYPE, REG_TYPE, BIT_WIDTH)                                                     \
template <typename T = DATA_TYPE, typename U = DATA_TYPE>                                                \
__aicore__ inline void AxpyImpl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue, \
                                const int32_t& count)                                                 \
{                                                                                                        \
    __VEC_SCOPE__ {                                                                                      \
        for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                        \
            vector_##REG_TYPE vreg0;                                                                     \
            vector_##REG_TYPE vreg1;                                                                     \
            vector_bool preg;                                                                            \
            vector_address offset;                                                                       \
            preg = vpd_b##BIT_WIDTH();                                                                   \
            offset = vag_b##BIT_WIDTH(ELE_CNT_B##BIT_WIDTH);                                             \
            vld(vreg0, src, offset, NORM);                                                               \
            vld(vreg1, dst, offset, NORM);                                                               \
            vaxpy(vreg1, vreg0, scalarValue, preg);                                                      \
            vst(vreg1, dst, offset, NORM_B##BIT_WIDTH, preg);                                            \
        }                                                                                                \
    }                                                                                                    \
}

AXPY_OP_IMPL(half, f16, 16)
AXPY_OP_IMPL(float, f32, 32)

template <typename T = float, typename U = half>
__aicore__ inline void AxpyImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue, const int32_t& count)
{
    __VEC_SCOPE__ {
        for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); i++) {
            vector_f16 vreg_in;
            vector_f32 vreg_even;
            vector_f32 vreg_odd;
            vector_f32 vreg_lower;
            vector_f32 vreg_higher;
            vector_f32 vreg_out0;
            vector_f32 vreg_out1;
            vector_bool p0 = vpd_b16();
            vector_bool p1;
            vector_bool p2;
            vector_address offset_in = vag_b16(ELE_CNT_B16);
            vector_address offset_out = vag_b32(ELE_CNT_B16);
            vld(vreg_in, src, offset_in, NORM);
            vld(vreg_out0, dst, offset_out, NORM);
            vld(vreg_out1, dst + ELE_CNT_B32, offset_out, NORM);
            vmuls(vreg_in, vreg_in, scalarValue, p0);
            vfcvt(vreg_even, vreg_in, PART_EVEN);
            vfcvt(vreg_odd, vreg_in, PART_ODD);
            vintlv(vreg_lower, vreg_higher, vreg_even, vreg_odd);
            punpack(p1, p0, LOWER);
            punpack(p2, p0, HIGHER);
            vadd(vreg_out0, vreg_out0, vreg_lower, p1);
            vadd(vreg_out1, vreg_out1, vreg_higher, p2);
            vst(vreg_out0, dst, offset_out, NORM_B32, p1);
            vst(vreg_out1, dst + ELE_CNT_B32, offset_out, NORM_B32, p2);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
