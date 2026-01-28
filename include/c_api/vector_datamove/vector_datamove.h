/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)  
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H  
#endif     

#ifndef INCLUDE_C_API_VECTOR_DATAMOVE_H
#define INCLUDE_C_API_VECTOR_DATAMOVE_H

#include "instr_impl/npu_arch_2201/vector_datamove_impl.h"

__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, uint8_t sid,
    uint16_t n_burst, uint16_t burst_len, uint16_t src_gap, uint16_t dst_gap);

__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_sync(__ubuf__ void* dst, __gm__ void* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, uint8_t sid,
    uint16_t n_burst, uint16_t burst_len, uint16_t src_gap, uint16_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_sync(__gm__ void* dst, __ubuf__ void* src, uint32_t size);

//asc_copy_gm2ub_align  int8_t / uint8_t / half / bfloat16_t / int16_t / uint16_t / float / int32_t / uint32_t
__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int8_t* dst, __gm__ int8_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ half* dst, __gm__ half* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ half* dst, __gm__ half* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ half* dst, __gm__ half* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int16_t* dst, __gm__ int16_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ float* dst, __gm__ float* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ float* dst, __gm__ float* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ float* dst, __gm__ float* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int32_t* dst, __gm__ int32_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t size);

//asc_copy_ub2gm_align
__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  int8_t* dst, __ubuf__ int8_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ half* dst, __ubuf__ half* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  half* dst, __ubuf__ half* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ half* dst, __ubuf__ half* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  int16_t* dst, __ubuf__ int16_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ float* dst, __ubuf__ float* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  float* dst, __ubuf__ float* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ float* dst, __ubuf__ float* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  int32_t* dst, __ubuf__ int32_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ double* dst, __ubuf__ double* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  double* dst, __ubuf__ double* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ double* dst, __ubuf__ double* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  int64_t* dst, __ubuf__ int64_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint8_t sid, uint16_t n_burst, uint32_t len_burst,
            uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap);

__aicore__ inline void asc_copy_ub2gm_align(__gm__  uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size);

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H)  
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H  
#endif    

