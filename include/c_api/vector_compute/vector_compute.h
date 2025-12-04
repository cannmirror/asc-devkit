/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_C_API_VECTOR_COMPUTE_H
#define INCLUDE_C_API_VECTOR_COMPUTE_H

#include "c_api_instr_impl/vector_compute/vector_compute_impl.h"
#include "c_api_interf_util.h"

__aicore__ inline void asc_get_cmp_mask(__ubuf__ void* dst);

__aicore__ inline void asc_get_reduce_max_min_cnt(half& val, uint32_t& index);

__aicore__ inline void asc_get_reduce_max_min_cnt(float& val, uint32_t& index);

__aicore__ inline int64_t asc_get_rsvd_count();

__aicore__ inline void asc_get_vms4_sr(uint16_t sorted_num[4]);

__aicore__ inline void asc_set_cmp_mask(__ubuf__ void* sel_mask);

__aicore__ inline void asc_set_mask_count();

__aicore__ inline void asc_set_mask_norm();

__aicore__ inline void asc_set_vector_mask(uint64_t mask1, uint64_t mask0);

// ==========asc_add(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_add(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_add_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_add_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_add_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

__aicore__ inline void asc_add(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_add_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

// ==========asc_add_scalar(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_add_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, const asc_unary_config& config);

__aicore__ inline void asc_add_scalar_sync(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, const asc_unary_config& config);

__aicore__ inline void asc_add_scalar_sync(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, const asc_unary_config& config);

__aicore__ inline void asc_add_scalar_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

__aicore__ inline void asc_add_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, const asc_unary_config& config);

__aicore__ inline void asc_add_scalar_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

// ==========asc_brcb(uint16_t/uint32_t)==========
__aicore__ inline void asc_brcb(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, const asc_brcb_config& config);

__aicore__ inline void asc_brcb_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, const asc_brcb_config& config);

__aicore__ inline void asc_brcb(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, const asc_brcb_config& config);

__aicore__ inline void asc_brcb_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, const asc_brcb_config& config);

// ==========asc_datablock_reduce(half/float)==========
__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, const asc_block_reduce_config& config);

__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_sum_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, const asc_block_reduce_config& config);

__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_sum_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_max(__ubuf__ half* dst, __ubuf__ half* src, const asc_block_reduce_config& config);

__aicore__ inline void asc_datablock_reduce_max(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_max_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_max(__ubuf__ float* dst, __ubuf__ float* src, const asc_block_reduce_config& config);

__aicore__ inline void asc_datablock_reduce_max(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_max_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, const asc_block_reduce_config& config);

__aicore__ inline void asc_datablock_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_min_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, const asc_block_reduce_config& config);

__aicore__ inline void asc_datablock_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_datablock_reduce_min_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

// ==========asc_repeat_reduce(half/float)==========
__aicore__ inline void asc_repeat_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, const asc_repeat_reduce_config& config);

__aicore__ inline void asc_repeat_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_repeat_reduce_sum_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_repeat_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, const asc_repeat_reduce_config& config);

__aicore__ inline void asc_repeat_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_repeat_reduce_sum_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_repeat_reduce_max(__ubuf__ half* dst, __ubuf__ half* src, const asc_repeat_reduce_config& config, order_t order);

__aicore__ inline void asc_repeat_reduce_max(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_max_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_max(__ubuf__ float* dst, __ubuf__ float* src, const asc_repeat_reduce_config& config, order_t order);

__aicore__ inline void asc_repeat_reduce_max(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_max_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, const asc_repeat_reduce_config& config, order_t order);

__aicore__ inline void asc_repeat_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_min_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, const asc_repeat_reduce_config& config, order_t order);

__aicore__ inline void asc_repeat_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count, order_t order);

__aicore__ inline void asc_repeat_reduce_min_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count, order_t order);

// ==========asc_bf162float==========
__aicore__ inline void asc_bf162float(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config);

__aicore__ inline void asc_bf162float(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162float_sync(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t count);

// ==========asc_bf162int32==========
__aicore__ inline void asc_bf162int32_a(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config);

__aicore__ inline void asc_bf162int32_a(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_a_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_c(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config);

__aicore__ inline void asc_bf162int32_c(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_c_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_f(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config);

__aicore__ inline void asc_bf162int32_f(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_f_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_r(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config);

__aicore__ inline void asc_bf162int32_r(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_r_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_z(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config);

__aicore__ inline void asc_bf162int32_z(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

__aicore__ inline void asc_bf162int32_z_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count);

// ==========asc_float2bf16==========
__aicore__ inline void asc_float2bf16_r(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2bf16_r(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_r_sync(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_a(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2bf16_a(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_a_sync(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_f(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2bf16_f(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_f_sync(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_c(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2bf16_c(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_c_sync(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_z(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2bf16_z(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2bf16_z_sync(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t count);

// ==========asc_float2float==========
__aicore__ inline void asc_float2float_r(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2float_r(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_r_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_f(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2float_f(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_f_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_c(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2float_c(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_c_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_a(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2float_a(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_a_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_z(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2float_z(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2float_z_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

// ==========asc_float2half==========
__aicore__ inline void asc_float2half(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_r(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half_r(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_r_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_a(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half_a(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_a_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_f(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half_f(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_f_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_c(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half_c(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_c_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_z(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half_z(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_z_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_o(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_float2half_o(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_float2half_o_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count);

// ==========asc_half2float==========
__aicore__ inline void asc_half2float(__ubuf__ float* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2float(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2float_sync(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count);

// ==========asc_half2int4==========
__aicore__ inline void asc_half2int4(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int4(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_a(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int4_a(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_a_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_c(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int4_c(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_c_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_f(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int4_f(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_f_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_r(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int4_r(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_r_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_z(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int4_z(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int4_z_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count);

// ==========asc_half2int16==========
__aicore__ inline void asc_half2int16_a(__ubuf__ int16_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int16_a(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_a_sync(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_c(__ubuf__ int16_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int16_c(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_c_sync(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_f(__ubuf__ int16_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int16_f(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_f_sync(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_r(__ubuf__ int16_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int16_r(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_r_sync(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_z(__ubuf__ int16_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int16_z(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int16_z_sync(__ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t count);

// ==========asc_half2int32==========
__aicore__ inline void asc_half2int32_a(__ubuf__ int32_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int32_a(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_a_sync(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_c(__ubuf__ int32_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int32_c(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_c_sync(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_f(__ubuf__ int32_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int32_f(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_f_sync(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_r(__ubuf__ int32_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int32_r(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_r_sync(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_z(__ubuf__ int32_t* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_half2int32_z(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_half2int32_z_sync(__ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t count);

// ==========asc_div(half/float)==========
__aicore__ inline void asc_div(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_div(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_div_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_div(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_div(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_div_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

// ==========asc_duplicate(half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t)==========
__aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ half* dst, half src, uint32_t count);

__aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ int16_t* dst, int16_t src, uint32_t count);

__aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count);

__aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count);

__aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ float* dst, float src, uint32_t count);

__aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ int32_t* dst, int32_t src, uint32_t count);

__aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, const asc_duplicate_config& config);

__aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count);

__aicore__ inline void asc_duplicate_sync(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count);

// ==========asc_exp(half/float)==========
__aicore__ inline void asc_exp(__ubuf__ half* dst, __ubuf__ half* src, const asc_unary_config& config);

__aicore__ inline void asc_exp(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_exp_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count);

__aicore__ inline void asc_exp(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config);

__aicore__ inline void asc_exp(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

__aicore__ inline void asc_exp_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count);

// ==========asc_max(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_max(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_max_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_max_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_max_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

__aicore__ inline void asc_max(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_max_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

// ==========asc_max_scalar(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_max_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, const asc_unary_config& config);

__aicore__ inline void asc_max_scalar_sync(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, const asc_unary_config& config);

__aicore__ inline void asc_max_scalar_sync(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, const asc_unary_config& config);

__aicore__ inline void asc_max_scalar_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

__aicore__ inline void asc_max_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, const asc_unary_config& config);

__aicore__ inline void asc_max_scalar_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

// ==========asc_min(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_min(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_min_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_min_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_min_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

__aicore__ inline void asc_min(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_min_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

// ==========asc_mul(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_mul(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_mul_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_mul_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_mul_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

__aicore__ inline void asc_mul(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_mul_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

// ==========asc_mul_scalar(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_mul_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, const asc_unary_config& config);

__aicore__ inline void asc_mul_scalar_sync(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, const asc_unary_config& config);

__aicore__ inline void asc_mul_scalar_sync(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, const asc_unary_config& config);

__aicore__ inline void asc_mul_scalar_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

__aicore__ inline void asc_mul_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, const asc_unary_config& config);

__aicore__ inline void asc_mul_scalar_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

// ==========asc_select(half/float)==========
__aicore__ inline void asc_select(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_select(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_select_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_select(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_select(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_select_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

// ==========asc_sub(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_sub(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config);

__aicore__ inline void asc_sub_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config);

__aicore__ inline void asc_sub_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_sub_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

__aicore__ inline void asc_sub(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, const asc_binary_config& config);

__aicore__ inline void asc_sub_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count);

// ==========asc_sub_scalar(half/float/int16_t/int32_t)==========
__aicore__ inline void asc_sub_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, const asc_unary_config& config);

__aicore__ inline void asc_sub_scalar_sync(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, const asc_unary_config& config);

__aicore__ inline void asc_sub_scalar_sync(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, const asc_unary_config& config);

__aicore__ inline void asc_sub_scalar_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

__aicore__ inline void asc_sub_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, const asc_unary_config& config);

__aicore__ inline void asc_sub_scalar_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count);

#endif