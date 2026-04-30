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
 * \file asc_aicore_printf_utils.h
 * \brief
 */
#ifndef IMPL_UTILS_DEBUG_NPU_ARCH_3510_ASC_AICORE_PRINTF_UTILS_H
#define IMPL_UTILS_DEBUG_NPU_ARCH_3510_ASC_AICORE_PRINTF_UTILS_H

#include "impl/utils/debug/asc_debug_utils.h"
#include "impl/utils/debug/npu_arch_3510/asc_type_conversion_utils.h"
namespace __asc_aicore {

template <typename T>
__aicore__ inline void set_scalar_param_impl(__gm__ uint8_t* paramAddr, uint32_t paramIdx, T scalar)
{
    __gm__ uint64_t *scalarAddr = (__gm__ uint64_t *)paramAddr + paramIdx;
    *scalarAddr = 0;

    if constexpr (is_same_in_list<T, half, float>()) {
        *((__gm__ float *)scalarAddr) = static_cast<float>(scalar);
    } else if constexpr (is_same_in_list<T, double>()) {
        *((__gm__ double *)scalarAddr) = static_cast<double>(scalar);
    } else if constexpr (std::is_signed<T>::value) {
        *((__gm__ int64_t *)scalarAddr) = static_cast<int64_t>(scalar);
    } else if constexpr(std::is_unsigned<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    } else if constexpr(is_same_in_list<T, bfloat16_t, float8_e5m2_t, float8_e8m0_t, float8_e4m3_t, hifloat8_t>()) {
        *((__gm__ float *)scalarAddr) = to_float(scalar);
    } else if constexpr(std::is_pointer<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = (uintptr_t)scalar;
    } else if constexpr(std::is_enum<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    }
    asc_entire_dcci((__gm__ uint64_t*)scalarAddr);
}

} // namespace __asc_aicore

namespace __asc_simd_vf {
template <AscendC::Hardware hardware, typename T, typename U>
__simd_callee__ inline void set_dump_tlv_info_vf(U& src, __ubuf__ DumpTensorTlv* dumpTlv,
    uint32_t alignDumpLen, uint32_t desc, uint32_t dumpSize, uint16_t blockIdx)
{
    dumpTlv->type = static_cast<uint32_t>(DumpType::DUMP_TENSOR);
    dumpTlv->length = sizeof(DumpTensorTlv) - sizeof(uint32_t[2]) + alignDumpLen;
    dumpTlv->tensorAddr = 0U; // set in aicore
    dumpTlv->dataType = static_cast<uint32_t>(get_dump_datatype<T>());
    dumpTlv->desc = desc;
    dumpTlv->blockIdx = blockIdx;  // set in aicore
    dumpTlv->bufferId = static_cast<uint32_t>(0U);
    dumpTlv->position = static_cast<uint16_t>(hardware);
    dumpTlv->dim = static_cast<uint32_t>(0U);
    for (uint32_t i = 0; i < 8; ++i) {
        dumpTlv->shape[i] = static_cast<uint32_t>(0U);
    }
    dumpTlv->resv1 = static_cast<uint32_t>(0U);
    dumpTlv->dumpSize = dumpSize * sizeof(T);
}

template <AscendC::Hardware hardware, typename T, typename U>
__simd_callee__ inline void set_dump_tlv_data_vf(U& src, __ubuf__ DumpTensorTlv* dumpTlv,
    uint32_t alignDumpLen, uint32_t dumpSize)
{
    __ubuf__ T* dumpDstAddr = reinterpret_cast<__ubuf__ T*>(dumpTlv + 1);

    // Scalar copy; performance optimization deferred
    for (uint32_t i = 0; i < dumpSize; i++) {
        dumpDstAddr[i] = src[i];
    }
}

template <AscendC::Hardware hardware, typename T, typename U>
__simd_callee__ inline void set_dump_tlv_data_reg(U& src, __ubuf__ DumpTensorTlv* dumpTlv,
    uint32_t alignDumpLen, uint32_t dumpSize)
{
    __ubuf__ T* dumpDstAddr = reinterpret_cast<__ubuf__ T*>(dumpTlv + 1);

    // Store interface supports unaligned transfer; dumpDstAddr is likely unaligned
    uint32_t count = dumpTlv->dumpSize / sizeof(T);
    Store(dumpDstAddr, src, count);
}

template <AscendC::Hardware hardware, typename T, typename U>
__simd_callee__ inline void asc_dump_impl_reg(U& src, uint32_t desc, uint32_t dumpSize)
{
    __ubuf__ BlockVFBufInfo *blockInfo = get_printf_ubuf_addr(0);
    
    constexpr uint16_t dataBlockSize = 32;
    uint32_t alignDumpLen = align_up(dumpSize * sizeof(T), dataBlockSize);
    uint32_t tlvLen = sizeof(DumpTensorTlv) + alignDumpLen;

    __ubuf__ DumpTensorTlv* dumpTlv =
        (__ubuf__ DumpTensorTlv*)((__ubuf__ uint8_t*)(blockInfo->buffer) + blockInfo->writeLen);
    set_dump_tlv_info_vf<hardware, T>(src, dumpTlv, alignDumpLen, desc, dumpSize, blockInfo->blockIdx);
    set_dump_tlv_data_reg<hardware, T>(src, dumpTlv, alignDumpLen, dumpSize);

    blockInfo->magic = 0xF0A00B0F;
    blockInfo->writeLen += tlvLen;
    blockInfo->pidx += 1;
}

template <AscendC::Hardware hardware, typename T, typename U>
__simd_callee__ inline void asc_dump_impl(U& src, uint32_t desc, uint32_t dumpSize)
{
    __ubuf__ BlockVFBufInfo *blockInfo = get_printf_ubuf_addr(0);
    
    constexpr uint16_t dataBlockSize = 32;
    uint32_t alignDumpLen = align_up(dumpSize * sizeof(T), dataBlockSize);
    uint32_t tlvLen = sizeof(DumpTensorTlv) + alignDumpLen;

    __ubuf__ DumpTensorTlv* dumpTlv =
        (__ubuf__ DumpTensorTlv*)((__ubuf__ uint8_t*)(blockInfo->buffer) + blockInfo->writeLen);
    set_dump_tlv_info_vf<hardware, T>(src, dumpTlv, alignDumpLen, desc, dumpSize, blockInfo->blockIdx);
    set_dump_tlv_data_vf<hardware, T>(src, dumpTlv, alignDumpLen, dumpSize);

    blockInfo->magic = 0xF0A00B0F;
    blockInfo->writeLen += tlvLen;
    blockInfo->pidx += 1;
}

template <typename T, typename U>
__simd_callee__ inline void asc_dump_reg(U& input, uint32_t desc, uint32_t dumpSize)
{
    enable_asc_diagnostics();
    asc_dump_impl_reg<AscendC::Hardware::UB, T>(input, desc, dumpSize);
}

template <typename T>
__simd_callee__ inline void asc_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize) {
    enable_asc_diagnostics();
    asc_dump_impl<AscendC::Hardware::UB, T>(input, desc, dumpSize);
}

template <typename T>
__simd_callee__ inline void set_scalar_param_vf_impl(__ubuf__ uint8_t* paramAddr, uint32_t paramIdx, T scalar)
{
    __ubuf__ uint64_t *scalarAddr = (__ubuf__ uint64_t *)paramAddr + paramIdx;
    *scalarAddr = 0;

    if constexpr (is_same_in_list<T, half, float>()) {
        *((__ubuf__ float *)scalarAddr) = static_cast<float>(scalar);
    } else if constexpr (is_same_in_list<T, double>()) {
        *((__ubuf__ double *)scalarAddr) = static_cast<double>(scalar);
    } else if constexpr (std::is_signed<T>::value) {
        *((__ubuf__ int64_t *)scalarAddr) = static_cast<int64_t>(scalar);
    } else if constexpr(std::is_unsigned<T>::value) {
        *((__ubuf__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    } else if constexpr(is_same_in_list<T, bfloat16_t, float8_e5m2_t, float8_e8m0_t, float8_e4m3_t, hifloat8_t>()) {
        *((__ubuf__ float *)scalarAddr) = to_float(scalar);
    } else if constexpr(std::is_pointer<T>::value) {
        *((__ubuf__ uint64_t *)scalarAddr) = (uintptr_t)scalar;
    } else if constexpr(std::is_enum<T>::value) {
        *((__ubuf__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    }
}
} // namespace __asc_simd_vf

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_AICORE_PRINTF_UTILS__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_AICORE_PRINTF_UTILS__
#endif

#endif // IMPL_UTILS_DEBUG_NPU_ARCH_3510_ASC_AICORE_PRINTF_UTILS_H