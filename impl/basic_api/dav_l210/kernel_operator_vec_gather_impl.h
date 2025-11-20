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
 * \file kernel_operator_vec_gather_impl.h
 * \brief AscendC l210 support vector gather memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H

namespace AscendC {
// for Level 2 gather op
__aicore__ inline void CastIndexU322U16(__ubuf__ uint16_t* dst, __ubuf__ uint32_t* src, const uint32_t count)
{
    __VEC_SCOPE__ {
        for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); ++i) {
            vector_u32 input0;
            vector_u32 input1;
            vector_u32 input_even;
            vector_u32 input_odd;
            vector_u16 output;
            vector_address idx0 = vag_b32(ELE_CNT_B16);
            vector_address idx1 = vag_b16(ELE_CNT_B16);
            vector_bool preg = vpd_b16();
            vld(input0, src, idx0, NORM);
            vld(input1, src + ELE_CNT_B32, idx0, NORM);
            vdintlv(input_even, input_odd, input0, input1);
            vscvt(output, input_even, PART_EVEN);
            vscvt(output, input_odd, PART_ODD, MODE_MERGING);
            vst(output, dst, idx1, NORM_B16, preg);
        }
    }
}

/* **************************************************************************************************
 * Gather                                             *                                             *
 * **************************************************************************************************/
// Gather::Level 2
template <typename T, typename U>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ U* index,
    const uint32_t srcBaseAddr, const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

#define GATHER_OP_IMPL_B8(DATA_TYPE, IDX_TYPE, GATHER_REG, VST_REG)                                           \
template <typename T = DATA_TYPE, typename U = IDX_TYPE>                                                      \
__aicore__ inline void GatherImpl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, __ubuf__ IDX_TYPE* index, \
    const uint32_t srcBaseAddr, const uint32_t count)                                                         \
{                                                                                                             \
    __VEC_SCOPE__ {                                                                                           \
        for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); ++i) {                                         \
            vector_##GATHER_REG vreg0;                                                                        \
            vector_u8 vreg1;                                                                                  \
            vector_bool preg = vpd_b8();                                                                      \
            vector_address vgather_offset = vag_b16(ELE_CNT_B16);                                             \
            vector_address vst_offset = vag_b8(ELE_CNT_B16);                                                  \
            vgather2(vreg0, index, vgather_offset, src + srcBaseAddr / sizeof(DATA_TYPE));                    \
            vscvt(vreg1, vreg0, PART_EVEN);                                                                   \
            vst((vector_##VST_REG)vreg1, dst, vst_offset, PK_B16, preg);                                      \
        }                                                                                                     \
    }                                                                                                         \
}

#define GATHER_OP_IMPL_B16B32(DATA_TYPE, IDX_TYPE, REG_TYPE, BIT_WIDTH)                                       \
template <typename T = DATA_TYPE, typename U = IDX_TYPE>                                                      \
__aicore__ inline void GatherImpl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, __ubuf__ IDX_TYPE* index, \
    const uint32_t srcBaseAddr, const uint32_t count)                                                         \
{                                                                                                             \
    __VEC_SCOPE__ {                                                                                           \
        for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                                \
            vector_##REG_TYPE vreg;                                                                           \
            vector_bool preg = vpd_b##BIT_WIDTH();                                                            \
            vector_address offset = vag_b##BIT_WIDTH(ELE_CNT_B##BIT_WIDTH);                                   \
            vgather2(vreg, index, offset, src + srcBaseAddr / sizeof(DATA_TYPE));                             \
            vst(vreg, dst, offset, NORM_B##BIT_WIDTH, preg);                                                  \
        }                                                                                                     \
    }                                                                                                         \
}

GATHER_OP_IMPL_B8(uint8_t, uint16_t, u16, u8)
GATHER_OP_IMPL_B8(int8_t, uint16_t, s16, s8)
GATHER_OP_IMPL_B16B32(uint16_t, uint16_t, u16, 16)
GATHER_OP_IMPL_B16B32(int16_t, uint16_t, s16, 16)
GATHER_OP_IMPL_B16B32(uint32_t, uint32_t, u32, 32)
GATHER_OP_IMPL_B16B32(int32_t, uint32_t, s32, 32)
GATHER_OP_IMPL_B16B32(half, uint16_t, f16, 16)
GATHER_OP_IMPL_B16B32(float, uint32_t, f32, 32)

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
