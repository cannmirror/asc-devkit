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
 * \file dump_routing.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning "impl/c_api/instr_impl/npu_arch_2201/debug_impl/dump_routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_DUMP_ROUTING_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_DUMP_ROUTING_H

#include "instr_impl/npu_arch_2201/debug_impl/dump_impl.h"

class dump_ignore_gm{
public:
    template<typename T>
    __aicore__ inline void run(__gm__ uint8_t* input, uint32_t desc, uint32_t dump_size){};
};

template<uint32_t version>
struct dump_routing_gm {
    using type = dump_ignore_gm;
};

template<>
struct dump_routing_gm<arch_version::v2201> {
    using type = dump_v2201_gm;
};

class dump_ignore_ubuf{
public:
    template<typename T>
    __aicore__ inline void run(__ubuf__ uint8_t* input, uint32_t desc, uint32_t dump_size){};
};

template<uint32_t version>
struct dump_routing_ubuf {
    using type = dump_ignore_ubuf;
};

template<>
struct dump_routing_ubuf<arch_version::v2201> {
    using type = dump_v2201_ubuf;
};
#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif