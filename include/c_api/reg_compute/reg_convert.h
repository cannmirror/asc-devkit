/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_C_API_REG_COMPUTE_REG_CONVERT_H
#define INCLUDE_C_API_REG_COMPUTE_REG_CONVERT_H

#include "instr_impl/npu_arch_3510/vector_compute_impl.h"

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

// ==========asc_bfloat162int32_rn==========
__simd_callee__ inline void asc_bfloat162int32_rn(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rn_sat(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rn_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rn_sat_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_bfloat162int32_rna==========
__simd_callee__ inline void asc_bfloat162int32_rna(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rna_sat(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rna_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rna_sat_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_bfloat162int32_rd==========
__simd_callee__ inline void asc_bfloat162int32_rd(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rd_sat(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rd_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rd_sat_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_bfloat162int32_ru==========
__simd_callee__ inline void asc_bfloat162int32_ru(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_ru_sat(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_ru_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_ru_sat_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_bfloat162int32_rz==========
__simd_callee__ inline void asc_bfloat162int32_rz(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rz_sat(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rz_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_bfloat162int32_rz_sat_v2(vector_int32_t& dst, vector_bfloat16_t src, vector_bool mask);

// ==========asc_half2int16_rn==========
__simd_callee__ inline void asc_half2int16_rn(vector_int16_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int16_rn_sat(vector_int16_t& dst, vector_half src, vector_bool mask);

// ==========asc_half2int16_rna==========
__simd_callee__ inline void asc_half2int16_rna(vector_int16_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int16_rna_sat(vector_int16_t& dst, vector_half src, vector_bool mask);

// ==========asc_half2int16_rd==========
__simd_callee__ inline void asc_half2int16_rd(vector_int16_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int16_rd_sat(vector_int16_t& dst, vector_half src, vector_bool mask);

// ==========asc_half2int16_ru==========
__simd_callee__ inline void asc_half2int16_ru(vector_int16_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int16_ru_sat(vector_int16_t& dst, vector_half src, vector_bool mask);

// ==========asc_half2int16_rz==========
__simd_callee__ inline void asc_half2int16_rz(vector_int16_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int16_rz_sat(vector_int16_t& dst, vector_half src, vector_bool mask);

// ==========asc_int642float_rn==========
__simd_callee__ inline void asc_int642float_rn(vector_float& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642float_rn_v2(vector_float& dst, vector_int64_t src, vector_bool mask);

// ==========asc_int642float_rna==========
__simd_callee__ inline void asc_int642float_rna(vector_float& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642float_rna_v2(vector_float& dst, vector_int64_t src, vector_bool mask);

// ==========asc_int642float_rd==========
__simd_callee__ inline void asc_int642float_rd(vector_float& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642float_rd_v2(vector_float& dst, vector_int64_t src, vector_bool mask);

// ==========asc_int642float_ru==========
__simd_callee__ inline void asc_int642float_ru(vector_float& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642float_ru_v2(vector_float& dst, vector_int64_t src, vector_bool mask);

// ==========asc_int642float_rz==========
__simd_callee__ inline void asc_int642float_rz(vector_float& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642float_rz_v2(vector_float& dst, vector_int64_t src, vector_bool mask);

// ==========asc_int82half==========
__simd_callee__ inline void asc_int82half(vector_half& dst, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_int82half_v2(vector_half& dst, vector_int8_t src, vector_bool mask);

// ==========asc_int162int32==========
__simd_callee__ inline void asc_int162int32(vector_int32_t& dst, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_int162int32_v2(vector_int32_t& dst, vector_int16_t src, vector_bool mask);

// ==========asc_uint322uint8==========
__simd_callee__ inline void asc_uint322uint8(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_sat(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_v2(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_sat_v2(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_v3(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_sat_v3(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_v4(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_uint322uint8_sat_v4(vector_uint8_t& dst, vector_uint32_t src, vector_bool mask);

// ==========asc_hif82half==========
__simd_callee__ inline void asc_hif82half(vector_half& dst, vector_hifloat8_t src, vector_bool mask);

__simd_callee__ inline void asc_hif82half_v2(vector_half& dst, vector_hifloat8_t src, vector_bool mask);

// ==========asc_half2int4x2==========
// rd - sat:dis/en - v1 v2 v3 v4
__simd_callee__ inline void asc_half2int4x2_rd(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

// ru - sat:dis/en - v1 v2 v3 v4
__simd_callee__ inline void asc_half2int4x2_rd(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

// rz - sat:dis/en - v1 v2 v3 v4
__simd_callee__ inline void asc_half2int4x2_rd(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

// rn - sat:dis/en - v1 v2 v3 v4
__simd_callee__ inline void asc_half2int4x2_rd(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

// rna - sat:dis/en - v1 v2 v3 v4
__simd_callee__ inline void asc_half2int4x2_rd(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v2(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v3(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int4x2_rd_sat_v4(vector_int4x2_t& dst, vector_half src, vector_bool mask);

// ==========asc_uint82half==========
__simd_callee__ inline void asc_uint82half(vector_half& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_uint82half_v2(vector_half& dst, vector_uint8_t src, vector_bool mask);

// ==========asc_uint162uint32==========
__simd_callee__ inline void asc_uint162uint32(vector_uint32_t& dst, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_uint162uint32_v2(vector_uint32_t& dst, vector_uint16_t src, vector_bool mask);

// ==========asc_hif82float==========
__simd_callee__ inline void asc_hif82float(vector_float& dst, vector_hifloat8_t src, vector_bool mask);

__simd_callee__ inline void asc_hif82float_v2(vector_float& dst, vector_hifloat8_t src, vector_bool mask);

// ==========asc_float2bfloat16==========
__simd_callee__ inline void asc_float2bfloat16_rd(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rd_sat(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rd_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rd_sat_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rn(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rn_sat(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rn_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rn_sat_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rna(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rna_sat(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rna_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rna_sat_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_ru(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_ru_sat(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_ru_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_ru_sat_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rz(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rz_sat(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rz_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_float2bfloat16_rz_sat_v2(vector_bfloat16_t& dst, vector_float src, vector_bool mask);

// ==========asc_half2int32==========
__simd_callee__ inline void asc_half2int32_rd(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rd_v2(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rn(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rn_v2(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rna(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rna_v2(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_ru(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_ru_v2(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rz(vector_int32_t& dst, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_half2int32_rz_v2(vector_int32_t& dst, vector_half src, vector_bool mask);

// ==========asc_e5m22float==========
__simd_callee__ inline void asc_e5m22float(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask);

__simd_callee__ inline void asc_e5m22float_v2(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask);

__simd_callee__ inline void asc_e5m22float_v3(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask);

__simd_callee__ inline void asc_e5m22float_v4(vector_float& dst, vector_fp8_e5m2_t src, vector_bool mask);

// ==========asc_int322int64==========
__simd_callee__ inline void asc_int322int64(vector_int64_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322int64_v2(vector_int64_t& dst, vector_int32_t src, vector_bool mask);

// ==========asc_int322uint16==========
__simd_callee__ inline void asc_int322uint16(vector_uint16_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322uint16_sat(vector_uint16_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322uint16_v2(vector_uint16_t& dst, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_int322uint16_sat_v2(vector_uint16_t& dst, vector_int32_t src, vector_bool mask);

// ==========asc_int642int32==========
__simd_callee__ inline void asc_int642int32(vector_int32_t& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642int32_sat(vector_int32_t& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642int32_v2(vector_int32_t& dst, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_int642int32_sat_v2(vector_int32_t& dst, vector_int64_t src, vector_bool mask);

// ==========asc_uint82uint16==========
__simd_callee__ inline void asc_uint82uint16(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_uint82uint16_v2(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask);

#endif
