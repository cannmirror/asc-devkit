/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_vec_createvecindex_impl.h
 * \brief AscendC l210 support vector create vector index memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> dst, const T firstValue, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

// for Level 2 createvecindex op inner
#define CREATEVECINDEXCALC_OP_IMPL_INNER(DATA_TYPE, FIRST_VAL_TYPE, REG_TYPE, BIT_WIDTH)                      \
    __aicore__ inline void CreateVecIndexCalcInner(__ubuf__ DATA_TYPE* dst, const FIRST_VAL_TYPE firstValue,  \
                                                   uint32_t count, DATA_TYPE scalar_offset)                \
    {                                                                                                         \
        __VEC_SCOPE__ {                                                                                       \
            vector_##REG_TYPE vreg;                                                                           \
            vci(vreg, firstValue);                                                                            \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                         \
                vector_address offset;                                                                        \
                vector_bool preg;                                                                             \
                preg = vpd_b##BIT_WIDTH();                                                                    \
                offset = vag_b##BIT_WIDTH(ELE_CNT_B##BIT_WIDTH);                                              \
                vadds(vreg, vreg, scalar_offset, preg);                                                       \
                vst(vreg, dst, offset, NORM_B##BIT_WIDTH, preg);                                              \
            }                                                                                                 \
        }                                                                                                     \
    }

CREATEVECINDEXCALC_OP_IMPL_INNER(int8_t, int8_t, s8, 8)
CREATEVECINDEXCALC_OP_IMPL_INNER(int16_t, int16_t, s16, 16)
CREATEVECINDEXCALC_OP_IMPL_INNER(int32_t, int32_t, s32, 32)
CREATEVECINDEXCALC_OP_IMPL_INNER(half, uint16_t, f16, 16)
CREATEVECINDEXCALC_OP_IMPL_INNER(float, uint32_t, f32, 32)

template <typename T = int8_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int8_t> dst, const int8_t firstValue, uint32_t count)
{
    __ubuf__ int8_t* dst = (__ubuf__ int8_t*)dst.GetPhyAddr();
    CreateVecIndexCalcInner(dst, firstValue - ELE_CNT_B8, count, ELE_CNT_B8);
}

template <typename T = int16_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int16_t> dst, const int16_t firstValue, uint32_t count)
{
    __ubuf__ int16_t* dst = (__ubuf__ int16_t*)dst.GetPhyAddr();
    CreateVecIndexCalcInner(dst, firstValue - ELE_CNT_B16, count, ELE_CNT_B16);
}

template <typename T = int32_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int32_t> dst, const int32_t firstValue, uint32_t count)
{
    __ubuf__ int32_t* dst = (__ubuf__ int32_t*)dst.GetPhyAddr();
    CreateVecIndexCalcInner(dst, firstValue - ELE_CNT_B32, count, ELE_CNT_B32);
}

template <typename T = half>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<half> dst, const half firstValue, uint32_t count)
{
    __ubuf__ half* dst = (__ubuf__ half*)dst.GetPhyAddr();
    __VEC_SCOPE__ {
        vector_f16 vreg;
        vci(vreg, static_cast<int64_t>(firstValue));
        for (uint16_t i = 0; i <= get_vloopn_bound_b16(ELE_CNT_B16); ++i) {
            vector_address offset;
            vector_bool preg;
            preg = vpd_b16();
            offset = vag_b16(ELE_CNT_B16);
            vst(vreg, dst, offset, NORM_B16, preg);
        }
    }

    if (count > ELE_CNT_B16) {
        CreateVecIndexCalcInner(dst + ELE_CNT_B16, static_cast<int64_t>(firstValue), count - ELE_CNT_B16,
            static_cast<int64_t>(ELE_CNT_B16));
    }
}

template <typename T = float>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<float> dst, const float firstValue, uint32_t count)
{
    __ubuf__ float* dst = (__ubuf__ float*)dst.GetPhyAddr();
    __VEC_SCOPE__ {
        vector_f32 vreg;
        vci(vreg, static_cast<int64_t>(firstValue));
        for (uint16_t i = 0; i <= get_vloopn_bound_b32(ELE_CNT_B32); ++i) {
            vector_address offset;
            vector_bool preg;
            preg = vpd_b32();
            offset = vag_b32(ELE_CNT_B32);
            vst(vreg, dst, offset, NORM_B32, preg);
        }
    }

    if (count > ELE_CNT_B32) {
        CreateVecIndexCalcInner(dst + ELE_CNT_B32, static_cast<int64_t>(firstValue), count - ELE_CNT_B32,
            static_cast<int64_t>(ELE_CNT_B32));
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
