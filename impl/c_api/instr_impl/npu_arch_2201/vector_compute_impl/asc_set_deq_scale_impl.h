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
#warning                                                                                                               \
    "impl/c_api/instr_impl/npu_arch_2201/vector_compute_impl/asc_set_deq_scale_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_2201_VECTOR_COMPUTE_IMPL_ASC_SET_DEQSCALE_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_2201_VECTOR_COMPUTE_IMPL_ASC_SET_DEQSCALE_IMPL_H

#include "instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

__aicore__ inline void asc_set_deq_scale_impl(float scale, int16_t offset, bool sign_mode)
{
    if ASC_IS_AIV {
        constexpr uint64_t sign_mode_bit = 46;
        constexpr uint64_t offset_mask = 0x1ff;
        constexpr uint64_t offset_bit = 37;
        uint64_t cfg = ((static_cast<uint64_t>(sign_mode) << sign_mode_bit) | ((offset & offset_mask) << offset_bit)
                        | *(reinterpret_cast<uint32_t*>(&scale)));
        set_deqscale(cfg);
    }
}

__aicore__ inline void asc_set_deq_scale_impl(half scale)
{
    if ASC_IS_AIV {
        set_deqscale(scale);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
