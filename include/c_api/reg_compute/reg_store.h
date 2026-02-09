/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_C_API_REG_COMPUTE_REG_STORE_H
#define INCLUDE_C_API_REG_COMPUTE_REG_STORE_H

#include "instr_impl/npu_arch_3510/vector_datamove_impl.h"

// ==========偏移固定传入0，由用户自行更新src地址=========
// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign(__ubuf__  int8_t* dst_align32b, vector_int8_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  int16_t* dst_align32b, vector_int16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  int32_t* dst_align32b, vector_int32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  half* dst_align32b, vector_half src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  float* dst_align32b, vector_float src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src,
    vector_bool mask);

// ==========asc_storealign_1st(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign_1st(__ubuf__  int8_t* dst_align32b, vector_int8_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  int16_t* dst_align32b, vector_int16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  int32_t* dst_align32b, vector_int32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  half* dst_align32b, vector_half src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  float* dst_align32b, vector_float src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src,
    vector_bool mask);

// ==========asc_storealign_pack(half/u16/s16/float/u32/s32/u64/s64/bf16)=========
__simd_callee__ inline void asc_storealign_pack(__ubuf__  int16_t* dst_align32b, vector_int16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  int32_t* dst_align32b, vector_int32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  int64_t* dst_align32b, vector_int64_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  uint64_t* dst_align32b, vector_uint64_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  half* dst_align32b, vector_half src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  float* dst_align32b, vector_float src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src,
    vector_bool mask);

// ==========asc_storealign_pack_v2(float/u32/s32)=========
__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  int32_t* dst_align32b, vector_int32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  float* dst_align32b, vector_float src,
    vector_bool mask);

// ==========asc_storealign_intlv(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2)=========
__simd_callee__ inline void asc_storealign_intlv(__ubuf__  int8_t* dst_align32b, vector_int8_t src0,
    vector_int8_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src0,
    vector_uint8_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  int16_t* dst_align32b, vector_int16_t src0,
    vector_int16_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src0,
    vector_uint16_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  int32_t* dst_align32b, vector_int32_t src0,
    vector_int32_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src0,
    vector_uint32_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  half* dst_align32b, vector_half src0,
    vector_half src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src0,
    vector_bfloat16_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0,
    vector_fp8_e4m3fn_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0,
    vector_fp8_e5m2_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0,
    vector_fp8_e8m0_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0,
    vector_fp4x2_e2m1_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0,
    vector_fp4x2_e1m2_t src1);

// ==========通过地址寄存器传入偏移，用户可以选择更新偏移或者更新src地址=========
// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign(__ubuf__  int8_t* dst_align32b, vector_int8_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  int16_t* dst_align32b, vector_int16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  half* dst_align32b, vector_half src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, iter_reg offset,
    vector_bool mask);

// ==========asc_storealign_1st(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign_1st(__ubuf__  int8_t* dst_align32b, vector_int8_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  int16_t* dst_align32b, vector_int16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  half* dst_align32b, vector_half src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_1st(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, iter_reg offset,
    vector_bool mask);

// ==========asc_storealign_pack(half/u16/s16/float/u32/s32/u64/s64/bf16)=========
__simd_callee__ inline void asc_storealign_pack(__ubuf__  int16_t* dst_align32b, vector_int16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  int64_t* dst_align32b, vector_int64_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  uint64_t* dst_align32b, vector_uint64_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  half* dst_align32b, vector_half src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src, iter_reg offset,
    vector_bool mask);

// ==========asc_storealign_pack_v2(float/u32/s32)=========
__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset,
    vector_bool mask);

// ==========asc_storealign_intlv(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2/e2m1/e1m2)=========
__simd_callee__ inline void asc_storealign_intlv(__ubuf__  int8_t* dst_align32b, vector_int8_t src0,
    vector_int8_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src0,
    vector_uint8_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  int16_t* dst_align32b, vector_int16_t src0,
    vector_int16_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src0,
    vector_uint16_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  int32_t* dst_align32b, vector_int32_t src0,
    vector_int32_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src0,
    vector_uint32_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  half* dst_align32b, vector_half src0,
    vector_half src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src0,
    vector_bfloat16_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0,
    vector_fp8_e4m3fn_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0,
    vector_fp8_e5m2_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0,
    vector_fp8_e8m0_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0,
    vector_fp4x2_e2m1_t src1, iter_reg offset);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0,
    vector_fp4x2_e1m2_t src1, iter_reg offset);

// ==========偏移固定传入0，由用户自行更新src地址=========
// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int8_t*& dst, vector_store_align& src0,
    vector_int8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  uint8_t*& dst, vector_store_align& src0,
    vector_uint8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int16_t*& dst, vector_store_align& src0,
    vector_int16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  uint16_t*& dst, vector_store_align& src0,
    vector_uint16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int32_t*& dst, vector_store_align& src0,
    vector_int32_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  uint32_t*& dst, vector_store_align& src0,
    vector_uint32_t src1, uint32_t count);
    
__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int64_t*& dst, vector_store_align& src0,
    vector_int64_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  half*& dst, vector_store_align& src0,
    vector_half src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  float*& dst, vector_store_align& src0,
    vector_float src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  bfloat16_t*& dst, vector_store_align& src0,
    vector_bfloat16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp8_e4m3fn_t*& dst, vector_store_align& src0,
    vector_fp8_e4m3fn_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp8_e5m2_t*& dst, vector_store_align& src0,
    vector_fp8_e5m2_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp8_e8m0_t*& dst, vector_store_align& src0,
    vector_fp8_e8m0_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp4x2_e2m1_t*& dst, vector_store_align& src0,
    vector_fp4x2_e2m1_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp4x2_e1m2_t*& dst, vector_store_align& src0,
    vector_fp4x2_e1m2_t src1, uint32_t count);

// ==========通过地址寄存器传入偏移，用户可以选择更新偏移或者更新src地址=========
// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int8_t* dst, vector_store_align& src0,
    vector_int8_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  uint8_t* dst, vector_store_align& src0,
    vector_uint8_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int16_t* dst, vector_store_align& src0,
    vector_int16_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  uint16_t* dst, vector_store_align& src0,
    vector_uint16_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int32_t* dst, vector_store_align& src0,
    vector_int32_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  uint32_t* dst, vector_store_align& src0,
    vector_uint32_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  int64_t* dst, vector_store_align& src0,
    vector_int64_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  half* dst, vector_store_align& src0,
    vector_half src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  float* dst, vector_store_align& src0,
    vector_float src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  bfloat16_t* dst, vector_store_align& src0,
    vector_bfloat16_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp8_e4m3fn_t* dst, vector_store_align& src0,
    vector_fp8_e4m3fn_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp8_e5m2_t* dst, vector_store_align& src0,
    vector_fp8_e5m2_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp8_e8m0_t* dst, vector_store_align& src0,
    vector_fp8_e8m0_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp4x2_e2m1_t* dst, vector_store_align& src0,
    vector_fp4x2_e2m1_t src1, iter_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__  fp4x2_e1m2_t* dst, vector_store_align& src0,
    vector_fp4x2_e1m2_t src1, iter_reg& count);

#endif
