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
 * \file kernel_micro_addrreg_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_ADDRREG_IMPL_H
#define ASCENDC_MODULE_MICRO_ADDRREG_IMPL_H

#include "kernel_micro_common_impl.h"
namespace AscendC {
namespace MicroAPI {
template <typename T> __simd_callee__ inline AddrReg CreateAddrRegImpl(uint16_t index0, uint16_t stride0)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
        "CreateAddrReg only support type b8/b16/b32");
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    return VagCpuSim(sizeof(T), index0, stride0);
#endif
    if constexpr (sizeof(T) == 1) {
        return vag_b8(stride0);
    } else if constexpr (sizeof(T) == 2) {
        return vag_b16(stride0);
    }
    return vag_b32(stride0);
}

template <typename T>
__simd_callee__ inline AddrReg CreateAddrRegImpl(uint16_t index0, uint16_t stride0, uint16_t index1, uint16_t stride1)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
        "CreateAddrReg only support type b8/b16/b32");
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    return VagCpuSim(sizeof(T), index0, stride0, index1, stride1);
#endif
    if constexpr (sizeof(T) == 1) {
        return vag_b8(stride0, stride1);
    } else if constexpr (sizeof(T) == 2) {
        return vag_b16(stride0, stride1);
    }
    return vag_b32(stride0, stride1);
}

template <typename T>
__simd_callee__ inline AddrReg CreateAddrRegImpl(uint16_t index0, uint16_t stride0, uint16_t index1, uint16_t stride1,
    uint16_t index2, uint16_t stride2)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
        "CreateAddrReg only support type b8/b16/b32");
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    return VagCpuSim(sizeof(T), index0, stride0, index1, stride1, index2, stride2);
#endif
    if constexpr (sizeof(T) == 1) {
        return vag_b8(stride0, stride1, stride2);
    } else if constexpr (sizeof(T) == 2) {
        return vag_b16(stride0, stride1, stride2);
    }
    return vag_b32(stride0, stride1, stride2);
}

template <typename T>
__simd_callee__ inline AddrReg CreateAddrRegImpl(uint16_t index0, uint16_t stride0, uint16_t index1, uint16_t stride1,
    uint16_t index2, uint16_t stride2, uint16_t index3, uint16_t stride3)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
        "CreateAddrReg only support type b8/b16/b32");
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    return VagCpuSim(sizeof(T), index0, stride0, index1, stride1, index2, stride2, index3, stride3);
#endif
    if constexpr (sizeof(T) == 1) {
        return vag_b8(stride0, stride1, stride2, stride3);
    } else if constexpr (sizeof(T) == 2) {
        return vag_b16(stride0, stride1, stride2, stride3);
    }
    return vag_b32(stride0, stride1, stride2, stride3);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_ADDRREG_IMPL_H