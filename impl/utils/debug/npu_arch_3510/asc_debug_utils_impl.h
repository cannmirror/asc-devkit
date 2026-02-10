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
 * \file asc_debug_utils_impl.h
 * \brief
 */
#ifndef IMPL_UTILS_DEBUG_NPU_ARCH_3510_ASC_DEBUG_UTILS_H
#define IMPL_UTILS_DEBUG_NPU_ARCH_3510_ASC_DEBUG_UTILS_H

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_DEBUG_UTILS_IMPL__
#warning "asc_debug_utils_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future."
#endif

#include "impl/utils/debug/asc_utils_macros.h"
namespace __asc_aicore {
__aicore__ inline void asc_entire_dcci_impl(__gm__ uint64_t* ptr)
{
    bisheng::cce::dcci(ptr, bisheng::cce::cache_line_t::ENTIRE_DATA_CACHE,
                       bisheng::cce::dcci_dst_t::CACHELINE_OUT);
}

__aicore__ inline uint64_t asc_debug_get_system_cycle_impl()
{
    uint64_t sysCnt = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(sysCnt));
    return sysCnt;
}

template <uint64_t timeoutCycle>
__aicore__ inline void ringbuf_wait_rts_sync_impl()
{
    const uint64_t firstTimeStamp = asc_debug_get_system_cycle_impl();
    while (static_cast<uint64_t>(asc_debug_get_system_cycle_impl()) - firstTimeStamp < timeoutCycle) {
        // Wait for RTS sync
    }
}

__aicore__ inline uint32_t asc_debug_get_core_idx_impl()
{
    constexpr uint32_t dumpCoreNums = 108;
    return get_coreid() % dumpCoreNums;
}

__aicore__ inline uint64_t asc_debug_get_block_idx_impl()
{
#if defined(__DAV_VEC__)
    return get_block_idx() * bisheng::cce::get_subblockdim() + bisheng::cce::get_subblockid();
#else
    return get_block_idx();
#endif
}

__aicore__ inline void sync_all_impl()
{
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ constexpr inline DumpTensorDataType get_dump_datatype_impl()
{
    if constexpr (std::is_same<T, bool>::value) {
        return DumpTensorDataType::ACL_BOOL;
    } else if (std::is_same<T, uint8_t>::value) {
        return DumpTensorDataType::ACL_UINT8;
    } else if (std::is_same<T, int8_t>::value) {
        return DumpTensorDataType::ACL_INT8;
    } else if (std::is_same<T, int16_t>::value) {
        return DumpTensorDataType::ACL_INT16;
    } else if (std::is_same<T, uint16_t>::value) {
        return DumpTensorDataType::ACL_UINT16;
    } else if (std::is_same<T, int32_t>::value) {
        return DumpTensorDataType::ACL_INT32;
    } else if (std::is_same<T, uint32_t>::value) {
        return DumpTensorDataType::ACL_UINT32;
    } else if (std::is_same<T, uint64_t>::value) {
        return DumpTensorDataType::ACL_UINT64;
    } else if (std::is_same<T, int64_t>::value) {
        return DumpTensorDataType::ACL_INT64;
    } else if (std::is_same<T, float>::value) {
        return DumpTensorDataType::ACL_FLOAT;
    } else if (std::is_same<T, half>::value) {
        return DumpTensorDataType::ACL_FLOAT16;
    } else if (std::is_same<T, bfloat16_t>::value) {
        return DumpTensorDataType::ACL_BF16;
    } else if (std::is_same<T, hifloat8_t>::value) {
        return DumpTensorDataType::ACL_HIFLOAT8;
    } else if (std::is_same<T, fp8_e5m2_t>::value) {
        return DumpTensorDataType::ACL_FLOAT8_E5M2;
    } else if (std::is_same<T, fp8_e4m3fn_t>::value) {
        return DumpTensorDataType::ACL_FLOAT8_E4M3FN;
    } else if (std::is_same<T, fp8_e8m0_t>::value) {
        return DumpTensorDataType::ACL_FLOAT8_E8M0;
    } else if (std::is_same<T, fp4x2_e2m1_t>::value) {
        return DumpTensorDataType::ACL_FLOAT4_E2M1;
    } else if (std::is_same<T, fp4x2_e1m2_t>::value) {
        return DumpTensorDataType::ACL_FLOAT4_E1M2;
    } else {
        return DumpTensorDataType::ACL_MAX;
    }
}

}
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_DEBUG_UTILS_IMPL__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_DEBUG_UTILS_IMPL__
#endif

#endif // IMPL_UTILS_DEBUG_NPU_ARCH_3510_ASC_DEBUG_UTILS_H