/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_C_API_REG_COMPUTE_REG_VECTOR_H
#define INCLUDE_C_API_REG_COMPUTE_REG_VECTOR_H

#include "instr_impl/npu_arch_3510/vector_compute_impl.h"

/*
*   enum class Pat {
*       ALL, // All elements are set to True
*       VL1, // The lowest element
*       VL2, // The lowest 2 element
*       VL3, // The lowest 3 element
*       VL4, // The lowest 4 element
*       VL8, // The lowest 8 element
*       VL16, // The lowest 16 element
*       VL32, // The lowest 32 element
*       VL64, // The lowest 64 element
*       VL128, // The lowest 128 element
*       M3, // Multiples of 3
*       M4, // Multiples of 4
*       H, // The lowest half elements
*       Q, // The lowest quarter elements
*       ALLF = 15 // All elements are set to False
*   };
*
*   usage example:
*       vector_bool mask = asc_create_mask_b8(Pat::VL1);
*/
#define asc_create_mask_b8 pset_b8
#define asc_create_mask_b16 pset_b16
#define asc_create_mask_b32 pset_b32

// ==========asc_create_iter_reg(b8/b16/b32)=========
__simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset);

__simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset);

__simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset);

// ==========asc_sub(uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_sub(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_bool& carry, vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_sub(vector_bool& carry, vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

// ==========asc_min(uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_min(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_min(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

// ==========asc_neg(int8_t/int16_t/half/int32_t/float)==========
__simd_callee__ inline void asc_neg(vector_int8_t& dst, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_neg(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_neg(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_neg(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_neg(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_ge(uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_ge(vector_bool& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_ge(vector_bool& dst, vector_float src0, vector_float src1, vector_bool mask);

// ==========asc_ge_scalar(uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_int8_t src, int8_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_int16_t src, int16_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_int32_t src, int32_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_uint8_t src, uint8_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_uint16_t src, uint16_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_uint32_t src, uint32_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_half src, half value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_bfloat16_t src, bfloat16_t value, vector_bool mask);

__simd_callee__ inline void asc_ge_scalar(vector_bool& dst, vector_float src, float value, vector_bool mask);

// ==========asc_reduce_max(uint16_t/int16_t/half/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_reduce_max(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_max(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_max(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_max(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_max(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_reduce_max(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_reduce_min(uint16_t/int16_t/half/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_reduce_min(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_reduce_min_datablock(uint16_t/int16_t/half/uint32_t/int32_t/float)==========
__simd_callee__ inline void asc_reduce_min_datablock_(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min_datablock_(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min_datablock_(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min_datablock_(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min_datablock_(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_reduce_min_datablock_(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_axpy(half/float)==========
__simd_callee__ inline void asc_axpy(vector_half& dst, vector_half src, half value, vector_bool mask);

__simd_callee__ inline void asc_axpy(vector_float& dst, vector_float src, float value, vector_bool mask);

// ==========asc_abs(int8_t/int16_t/int32_t/half/float)==========
__simd_callee__ inline void asc_abs(vector_int8_t& dst, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_abs(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_abs(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_abs(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_abs(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_mull(uint32_t/int32_t)==========
__simd_callee__ inline void asc_mull(vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_mull(vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

// ==========asc_le(uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float)==========
__simd_callee__ inline void asc_le(vector_bool& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_le(vector_bool& dst, vector_float src0, vector_float src1, vector_bool mask);

// ==========asc_le_scalar(uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float)==========
__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_uint8_t src, uint8_t value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_int8_t src, int8_t value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_uint16_t src, uint16_t value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_int16_t src, int16_t value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_uint32_t src, uint32_t value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_int32_t src, int32_t value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_half src, half value, vector_bool mask);

__simd_callee__ inline void asc_le_scalar(vector_bool& dst, vector_float src, float value, vector_bool mask);

// ==========asc_squeeze(uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float)==========
__simd_callee__ inline void asc_squeeze(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_int8_t& dst, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_squeeze(vector_float& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_int8_t& dst, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_squeeze_v2(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_intlv(uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float)==========
__simd_callee__ inline void asc_intlv_b8(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);

__simd_callee__ inline void asc_intlv_b16(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);

__simd_callee__ inline void asc_intlv_b32(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);

__simd_callee__ inline void asc_intlv(vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1);

__simd_callee__ inline void asc_intlv(vector_int8_t& dst0, vector_int8_t& dst1, vector_int8_t src0, vector_int8_t src1);

__simd_callee__ inline void asc_intlv(vector_uint16_t& dst0, vector_uint16_t& dst1, vector_uint16_t src0, vector_uint16_t src1);

__simd_callee__ inline void asc_intlv(vector_int16_t& dst0, vector_int16_t& dst1, vector_int16_t src0, vector_int16_t src1);

__simd_callee__ inline void asc_intlv(vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1);

__simd_callee__ inline void asc_intlv(vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1);

__simd_callee__ inline void asc_intlv(vector_half& dst0, vector_half& dst1, vector_half src0, vector_half src1);

__simd_callee__ inline void asc_intlv(vector_float& dst0, vector_float& dst1, vector_float src0, vector_float src1);

// ==========asc_unsqueeze(uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t)==========
__simd_callee__ inline void asc_unsqueeze(vector_uint8_t& dst, vector_bool mask);

__simd_callee__ inline void asc_unsqueeze(vector_int8_t& dst, vector_bool mask);

__simd_callee__ inline void asc_unsqueeze(vector_uint16_t& dst, vector_bool mask);

__simd_callee__ inline void asc_unsqueeze(vector_int16_t& dst, vector_bool mask);

__simd_callee__ inline void asc_unsqueeze(vector_uint32_t& dst, vector_bool mask);

__simd_callee__ inline void asc_unsqueeze(vector_int32_t& dst, vector_bool mask);

// ==========asc_arange(int8_t/int16_t/int32_t/half/float)==========
__simd_callee__ inline void asc_arange(vector_int8_t& dst, int8_t index);

__simd_callee__ inline void asc_arange(vector_int16_t& dst, int16_t index);

__simd_callee__ inline void asc_arange(vector_int32_t& dst, int32_t index);

__simd_callee__ inline void asc_arange(vector_half& dst, half index);

__simd_callee__ inline void asc_arange(vector_float& dst, float index);

__simd_callee__ inline void asc_arange_descend(vector_int8_t& dst, int8_t index);

__simd_callee__ inline void asc_arange_descend(vector_int16_t& dst, int16_t index);

__simd_callee__ inline void asc_arange_descend(vector_int32_t& dst, int32_t index);

__simd_callee__ inline void asc_arange_descend(vector_half& dst, half index);

__simd_callee__ inline void asc_arange_descend(vector_float& dst, float index);

// ==========asc_cumulative_histogram/asc_frequency_histogram==========
__simd_callee__ inline void asc_cumulative_histogram_bin0(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_cumulative_histogram_bin1(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_frequency_histogram_bin0(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_frequency_histogram_bin1(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask);

// ==========asc_update_mask==========
__simd_callee__ inline vector_bool asc_update_mask_b8(uint32_t& scalar);

__simd_callee__ inline vector_bool asc_update_mask_b16(uint32_t& scalar);

__simd_callee__ inline vector_bool asc_update_mask_b32(uint32_t& scalar);

// ==========asc_half2int8(rd/ru/rz/rn/rna)==========
__simd_callee__ inline void asc_half2int8_rd(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rd_sat(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rd_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rd_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_ru(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_ru_sat(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_ru_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_ru_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rz(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rz_sat(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rz_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rz_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rn(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rn_sat(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rn_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rn_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rna(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rna_sat(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rna_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int8_rna_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask);

// ==========asc_half2int8(rh/rna)==========
__simd_callee__ inline void asc_half2hif8_rh(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rh_sat(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rh_v2(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rh_sat_v2(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rna(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rna_sat(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rna_v2(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2hif8_rna_sat_v2(vector_hifloat8_t& dst, vector_half src, vector_bool mask);

// ==========asc_bfloat162float==========
__simd_callee__ inline void asc_bfloat162float(vector_float& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162float_v2(vector_float& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_int4x22bfloat16==========
__simd_callee__ inline void asc_int4x22bfloat16(vector_bfloat16_t& dst, vector_int4x2_t src, vector_bool mask);

__simd_callee__ inline void asc_int4x22bfloat16_v2(vector_bfloat16_t& dst, vector_int4x2_t src, vector_bool mask);

__simd_callee__ inline void asc_int4x22bfloat16_v3(vector_bfloat16_t& dst, vector_int4x2_t src, vector_bool mask);

__simd_callee__ inline void asc_int4x22bfloat16_v4(vector_bfloat16_t& dst, vector_int4x2_t src, vector_bool mask);

// ==========asc_int162uint32==========
__simd_callee__ inline void asc_int162uint32(vector_uint32_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_int162uint32_v2(vector_uint32_t& dst, vector_int16_t src, vector_bool mask);

// ==========asc_int322float(rd/ru/rz/rn/rna)==========
__simd_callee__ inline void asc_int322float_rd(vector_float& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322float_ru(vector_float& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322float_rz(vector_float& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322float_rn(vector_float& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322float_rna(vector_float& dst, vector_int32_t src, vector_bool mask);

// ==========asc_uint162uint8==========
__simd_callee__ inline void asc_uint162uint8(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_uint162uint8_sat(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_uint162uint8_v2(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_uint162uint8_sat_v2(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask);
// ==========asc_sqrt(half/float)==========
__simd_callee__ inline void asc_sqrt(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_sqrt(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_or(int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/float/bool)==========
__simd_callee__ inline void asc_or(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

__simd_callee__ inline void asc_or(vector_bool& dst, vector_bool src0, vector_bool src1, vector_bool mask);

// ==========asc_mul(int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
__simd_callee__ inline void asc_mul(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_mul(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_mul(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_mul(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_mul(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_mul(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_mul(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

// ==========asc_mul_scalar(int16_t/uint16_t/half/int32_t/uint32_t/float)==========
__simd_callee__ inline void asc_mul_scalar(vector_int16_t& dst, vector_int16_t src0, int16_t value, vector_bool mask);

__simd_callee__ inline void asc_mul_scalar(vector_uint16_t& dst, vector_uint16_t src0, uint16_t value, vector_bool mask);

__simd_callee__ inline void asc_mul_scalar(vector_half& dst, vector_half src0, half value, vector_bool mask);

__simd_callee__ inline void asc_mul_scalar(vector_int32_t& dst, vector_int32_t src0, int32_t value, vector_bool mask);

__simd_callee__ inline void asc_mul_scalar(vector_uint32_t& dst, vector_uint32_t src0, uint32_t value, vector_bool mask);

__simd_callee__ inline void asc_mul_scalar(vector_float& dst, vector_float src0, float value, vector_bool mask);

// ==========asc_eq(vcmp int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
__simd_callee__ inline void asc_eq(vector_bool& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_float src0, vector_float src1, vector_bool mask);

// ==========asc_eq(vcmps int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
__simd_callee__ inline void asc_eq(vector_bool& dst, vector_int8_t src0, int8_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_uint8_t src0, uint8_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_int16_t src0, int16_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_uint16_t src0, uint16_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_half src0, half value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_bfloat16_t src0, bfloat16_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_int32_t src0, int32_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_uint32_t src0, uint32_t value, vector_bool mask);

__simd_callee__ inline void asc_eq(vector_bool& dst, vector_float src0, float value, vector_bool mask);

// ==========asc_float2int32_rd/ru/rz/rn/rna)==========
__simd_callee__ inline void asc_float2int32_rd(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rd_sat(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_ru(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_ru_sat(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rz(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rz_sat(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rn(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rn_sat(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rna(vector_int32_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int32_rna_sat(vector_int32_t& dst, vector_float src, vector_bool mask);

// ==========asc_float2int16_rd/ru/rz/rn/rna)==========
__simd_callee__ inline void asc_float2int16_rd(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rd_sat(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rd_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rd_sat_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rn(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rn_sat(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rn_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rn_sat_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rna(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rna_sat(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rna_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rna_sat_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_ru(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_ru_sat(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_ru_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_ru_sat_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rz(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rz_sat(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rz_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2int16_rz_sat_v2(vector_int16_t& dst, vector_float src, vector_bool mask);

// ==========asc_bfloat162e2m1x2_rd/rn/rna/ru/rz)==========
__simd_callee__ inline void asc_bfloat162e2m1x2_rd(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_v2(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_v3(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_v4(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rn(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rn_v2(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rn_v3(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rn_v4(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rna(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rna_v2(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rna_v3(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rna_v4(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_ru(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_ru_v2(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_ru_v3(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_ru_v4(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rz(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rz_v2(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rz_v3(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162e2m1x2_rz_v4(vector_f4e2m1x2& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_float2hif8_rh/rna)==========
__simd_callee__ inline void asc_float2hif8_rh_sat(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh_sat_v2(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh_v2(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh_sat_v3(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh_v3(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh_sat_v4(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rh_v4(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_sat(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_sat_v2(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_v2(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_sat_v3(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_v3(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_sat_v4(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2hif8_rna_v4(vector_hifloat8_t& dst, vector_float src, vector_bool mask);

// ==========asc_uint82uint32==========
__simd_callee__ inline void asc_uint82uint32(vector_uint32_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_uint82uint32_v2(vector_uint32_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_uint82uint32_v3(vector_uint32_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_uint82uint32_v4(vector_uint32_t& dst, vector_uint8_t src, vector_bool mask);

// ==========asc_uint82uint32==========
__simd_callee__ inline void asc_uint322uint16_sat(vector_uint16_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint16(vector_uint16_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint16_sat_v2(vector_uint16_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint16_v2(vector_uint16_t& dst, vector_uint32_t src, vector_bool mask);

// ==========asc_ceil==========
__simd_callee__ inline void asc_ceil(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_ceil(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_ceil(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_floor==========
__simd_callee__ inline void asc_floor(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_floor(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_floor(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_rint==========
__simd_callee__ inline void asc_rint(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_rint(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_rint(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_round==========
__simd_callee__ inline void asc_round(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_round(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_round(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_trunc==========
__simd_callee__ inline void asc_trunc(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_trunc(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_trunc(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_e2m1x22bfloat==========
__simd_callee__ inline void asc_e2m1x22bfloat(vector_bfloat16_t& dst, vector_f4e2m1x2 src, vector_bool mask);

__simd_callee__ inline void asc_e2m1x22bfloat_v2(vector_bfloat16_t& dst, vector_f4e2m1x2 src, vector_bool mask);

__simd_callee__ inline void asc_e2m1x22bfloat_v3(vector_bfloat16_t& dst, vector_f4e2m1x2 src, vector_bool mask);

__simd_callee__ inline void asc_e2m1x22bfloat_v4(vector_bfloat16_t& dst, vector_f4e2m1x2 src, vector_bool mask);

// ==========asc_muls==========
__simd_callee__ inline void asc_muls(vector_half& dst, vector_float src, float value, vector_bool mask);

__simd_callee__ inline void asc_muls_v2(vector_half& dst, vector_float src, float value, vector_bool mask);

// ==========asc_add(u8/s8/u16/s18/u32/s32)=========
__simd_callee__ inline void asc_add(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_bool& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_add(vector_bool& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

// // ==========asc_addc(uint32_t/int32_t)==========
__simd_callee__ inline void asc_addc(vector_bool& dst0, vector_uint32_t& dst1,
    vector_uint32_t src0, vector_uint32_t src1, vector_bool src2, vector_bool mask);

__simd_callee__ inline void asc_addc(vector_bool& dst0, vector_int32_t& dst1,
    vector_int32_t src0, vector_int32_t src1, vector_bool src2, vector_bool mask);

// ==========asc_shiftleft(u8/s8/u16/s16/u32/s32)==========
__simd_callee__ inline void asc_shiftleft(vector_uint8_t& dst,
    vector_uint8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftleft(vector_int8_t& dst,
    vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftleft(vector_uint16_t& dst,
    vector_uint16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftleft(vector_int16_t& dst,
    vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftleft(vector_uint32_t& dst,
    vector_uint32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftleft(vector_int32_t& dst,
    vector_int32_t src0, vector_int32_t src1, vector_bool mask);

// ==========asc_shiftright(u8/s8/u16/s16/u32/s32)==========
__simd_callee__ inline void asc_shiftright(vector_uint8_t& dst,
    vector_uint8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftright(vector_int8_t& dst,
    vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftright(vector_uint16_t& dst,
    vector_uint16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftright(vector_int16_t& dst,
    vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftright(vector_uint32_t& dst,
    vector_uint32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_shiftright(vector_int32_t& dst,
    vector_int32_t src0, vector_int32_t src1, vector_bool mask);

// ==========asc_not(u8/s8/u16/s16/half/u32/s32/f32/bool)==========
__simd_callee__ inline void asc_not(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_int8_t& dst, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_int16_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_int32_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_float& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_not(vector_bool& dst, vector_bool src, vector_bool mask);

//==========asc_lt(u8/s8/half/u16/s16/float/u32/s32/bf16)==========
__simd_callee__ inline void asc_lt(vector_bool& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_float src0, vector_float src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt(vector_bool& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_uint8_t src, uint8_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_int8_t src, int8_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_bfloat16_t src, bfloat16_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_uint16_t src, uint16_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_int16_t src, int16_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_half src, half value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_uint32_t src, uint32_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_int32_t src, int32_t value, vector_bool mask);

__simd_callee__ inline void asc_lt_scalar(vector_bool& dst, vector_float src, float value, vector_bool mask);

// ==========asc_madd(half/float)==========
__simd_callee__ inline void asc_madd(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_madd(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

// ==========asc_pair_reduce_sum(half/float)==========
__simd_callee__ inline void asc_pair_reduce_sum(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_pair_reduce_sum(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_exp(half/float)==========
__simd_callee__ inline void asc_exp(vector_half& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_exp(vector_float& dst, vector_float src, vector_bool mask);

// ==========asc_add_scalar(int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
__simd_callee__ inline void asc_add_scalar(vector_int8_t& dst, vector_int8_t src, int8_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_uint8_t& dst, vector_uint8_t src, uint8_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_int16_t& dst, vector_int16_t src, int16_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_uint16_t& dst, vector_uint16_t src, uint16_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_half& dst, vector_half src, half value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_bfloat16_t& dst, vector_bfloat16_t src, bfloat16_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_int32_t& dst, vector_int32_t src, int32_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_uint32_t& dst, vector_uint32_t src, uint32_t value, vector_bool mask);

__simd_callee__ inline void asc_add_scalar(vector_float& dst, vector_float src, float value, vector_bool mask);

// ==========asc_pack(bool/uint16_t/int16_t/uint32_t/int32_t)==========
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_uint16_t src);

__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_int16_t src);

__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_uint32_t src);

__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_int32_t src);

__simd_callee__ inline void asc_pack(vector_bool& dst, vector_bool src);

__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_uint16_t src);

__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_int16_t src);

__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_uint32_t src);

__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_int32_t src);

__simd_callee__ inline void asc_pack_v2(vector_bool& dst, vector_bool src);

// ==========asc_exp_sub(half/float)==========
__simd_callee__ inline void asc_exp_sub(vector_float& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_exp_sub(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

__simd_callee__ inline void asc_exp_sub_v2(vector_float& dst, vector_half src0, vector_half src1, vector_bool mask);

__simd_callee__ inline void asc_exp_sub_v2(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);

#endif