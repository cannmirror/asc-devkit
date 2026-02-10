/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADALIGN_V2_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADALIGN_V2_IMPL_H

#include "instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_loadalign_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int32_t& dst, __ubuf__ int32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int64_t& dst, __ubuf__ int64_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint64_t& dst, __ubuf__ uint64_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_float& dst, __ubuf__ float* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B8);
    }
}

// =========BRC_B16(u16/s16/bf16/half)=========
__simd_callee__ inline void asc_loadalign_brc_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_half& dst, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B16);
    }
}

// =========BRC_B32(u32/s32/float)=========
__simd_callee__ inline void asc_loadalign_brc_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int32_t& dst, __ubuf__ int32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_float& dst, __ubuf__ float* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BRC_B32);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_half& dst, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_half& dst, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_half& dst0, vector_half& dst1, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B32);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B32);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(vector_float& dst0, vector_float& dst1, __ubuf__ float* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src_align_32b, offset, DINTLV_B32);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_half& dst, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B32);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int32_t& dst, __ubuf__ int32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B32);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_float& dst, __ubuf__ float* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK_B32);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int8_t& dst, __ubuf__ int8_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_half& dst, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int32_t& dst, __ubuf__ int32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_float& dst, __ubuf__ float* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, BLK);
    }
}

/// brc_v3(B16/B32)
__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_int16_t& dst, __ubuf__ int16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_half& dst, __ubuf__ half* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_int32_t& dst, __ubuf__ int32_t* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_float& dst, __ubuf__ float* src_align_32b, iter_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src_align_32b, offset, E2B_B32);
    }
}

#endif