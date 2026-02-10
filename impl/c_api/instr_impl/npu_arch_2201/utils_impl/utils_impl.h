/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file utils_c_api_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)  
#warning "impl/c_api/c_api_instr_impl/utils_c_api_impl/utils_c_api_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."  
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC  
#endif    

#ifndef IMPL_C_API_C_API_INSTR_IMPL_UTILS_C_API_IMPL_UTILS_C_API_IMPL_H
#define IMPL_C_API_C_API_INSTR_IMPL_UTILS_C_API_IMPL_UTILS_C_API_IMPL_H

#include <cstdint>
#include <type_traits>
#include "utils_intf.h"
#include "utils_type.h"

constexpr uint32_t C_API_AIC_TYPE = 1;
constexpr uint32_t C_API_AIV_TYPE = 2;
constexpr uint32_t C_API_MIX_TYPE = 3;

#if defined(ASCENDC_CPU_DEBUG)
    extern uint32_t g_c_api_core_type;
    #define ASC_IS_AIV (g_c_api_core_type == C_API_AIV_TYPE)
    #define ASC_IS_AIC (g_c_api_core_type == C_API_AIC_TYPE)
    #define ASC_IS_NOT_AIV (g_c_api_core_type != C_API_AIV_TYPE)
    #define ASC_IS_NOT_AIC (g_c_api_core_type != C_API_AIC_TYPE)
#else
    #if (defined(__DAV_CUBE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3101))
        constexpr uint32_t g_c_api_core_type = C_API_AIC_TYPE;
    #elif (defined(__DAV_VEC__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3101))
        constexpr uint32_t g_c_api_core_type = C_API_AIV_TYPE;
    #else
        constexpr uint32_t g_c_api_core_type = C_API_MIX_TYPE;
    #endif
    #define ASC_IS_AIV constexpr(g_c_api_core_type == C_API_AIV_TYPE)
    #define ASC_IS_AIC constexpr(g_c_api_core_type == C_API_AIC_TYPE)
    #define ASC_IS_NOT_AIV constexpr(g_c_api_core_type != C_API_AIV_TYPE)
    #define ASC_IS_NOT_AIC constexpr(g_c_api_core_type != C_API_AIC_TYPE)
#endif

struct asc_c_api_default_repeat {
    uint8_t U8 = 1;
};

struct asc_c_api_default_block_stride {
    uint8_t U8 = 1;
    uint16_t U16 = 1;
};

struct asc_c_api_default_repeat_stride {
    uint8_t U8 = 8;
    uint16_t U16 = 8;
};

struct asc_c_api_half_default_repeat_stride {
    uint8_t U8 = 4;
    uint16_t U16 = 4;
};

struct asc_c_api_one_fourth_default_repeat_stride {
    uint8_t U8 = 2;
    uint16_t U16 = 2;
};

constexpr asc_c_api_default_repeat ASC_C_API_DEFAULT_REPEAT;
constexpr asc_c_api_default_block_stride ASC_C_API_DEFAULT_BLOCK_STRIDE;
constexpr asc_c_api_default_repeat_stride ASC_C_API_DEFAULT_REPEAT_STRIDE;
constexpr asc_c_api_half_default_repeat_stride ASC_C_API_HALF_DEFAULT_REPEAT_STRIDE;
constexpr asc_c_api_one_fourth_default_repeat_stride ASC_C_API_ONE_FOURTH_DEFAULT_REPEAT_STRIDE;

constexpr uint16_t ASC_C_API_DATABLOCK_NUM = 8;
constexpr uint8_t ASC_C_API_MRGSORT_ELEMENT_LEN = 4;
constexpr uint16_t ASC_C_API_ONE_DATABLOCK_SIZE = 32;
constexpr uint16_t ASC_C_API_REDUCE_DEFAULT_REPEAT_STRIDE = 1;
constexpr uint8_t ASC_VDEQ_SIZE = 16;

using order_t = Order_t;

union asc_sparse_addr_config {
    uint64_t config;
    struct {
        uint64_t src : 32;
        uint64_t index : 32;
    };
};

__aicore__ inline void asc_set_mask_count_begin(uint32_t count)
{
    if ASC_IS_AIV {
        set_mask_count();
        set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    }
}

__aicore__ inline void asc_set_mask_count_end()
{
    if ASC_IS_AIV {
        set_mask_norm();
    }
}

__aicore__ inline void asc_sync_post_process()
{
    pipe_barrier(pipe_t::PIPE_ALL);
}

union asc_capi_arch_version {
    uint64_t full;
    struct {
        uint64_t low_bit : 32;
        uint64_t core_ver : 12;
        uint64_t reserved : 20;
    };
};


union asc_capi_fpc_reg_config {
    uint64_t config;
    struct {
        uint64_t relu_units : 8; 
        uint64_t quant_units : 8;
        uint64_t reserved : 47;
        uint64_t unit_flag : 1;
    };
};

union asc_scalar_bitcode {
        __aicore__ asc_scalar_bitcode() {}
        half input_half;
        int16_t input_int16_t;
        uint16_t input_uint16_t;
        uint64_t output;
};

#endif // IMPL_C_API_C_API_INSTR_IMPL_UTILS_C_API_IMPL_UTILS_C_API_IMPL_H

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)  
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC  
#endif  

