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
#warning "impl/c_api/instr_impl/npu_arch_2201/debug_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_H
 
#include "instr_impl/npu_arch_2201/debug_impl/dump_routing.h"
#include "instr_impl/npu_arch_2201/debug_impl/printf_impl.h"

template <typename T>
__aicore__ inline void asc_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size)
{
    using dump = dump_routing_gm<CURRENT_ARCH_VERSION>::type;
    dump{}.template run<T>(input, desc, dump_size);
}

template <typename T>
__aicore__ inline void asc_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    using dump = dump_routing_ubuf<CURRENT_ARCH_VERSION>::type;
    dump{}.template run<T>(input, desc, dump_size);
}

namespace __asc_aicore {
template <class... Args>
__aicore__ inline void printf(__gm__ const char* fmt, Args&&... args)
{
    simd_printf_impl(DumpType::DUMP_SCALAR, fmt, args...);
}
}

namespace __cce_scalar {
    using namespace __asc_aicore;
}

#define asc_assert_msg__(prompt, expr, fmt)                                                                  \
    do {                                                                                                          \
        if (!(expr)) {                                                                                            \
            simd_printf_impl(DumpType::DUMP_ASSERT, "%s[ASSERT] %s:%u: Assertion `%s' " fmt, prompt, __FILE__, __LINE__, #expr); \
            trap();                                                                                               \
        }                                                                                                         \
    } while (0)

#if defined(__ENABLE_ASCENDC_PRINTF__)
#if defined (ASCENDC_DUMP) && (ASCENDC_DUMP == 1)
#define asc_assert_impl(expr)                      \
    do {                                           \
        __gm__ const char* prompt = "";            \
        if (!(expr)) {                             \
            asc_assert_msg__(prompt, expr, "\n");  \
        }                                          \
    } while (0)
#endif
#endif

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif