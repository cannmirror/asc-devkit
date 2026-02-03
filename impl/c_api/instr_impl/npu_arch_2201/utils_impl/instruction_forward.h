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
#warning "impl/c_api/instr_impl/utils_impl/instruction_forward.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_UTILS_IMPL_INSTRUCTION_FORWARD_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_UTILS_IMPL_INSTRUCTION_FORWARD_H

extern  __gm__ uint8_t* g_sysPrintFifoSpace;

class instruction_forward 
{
private:
    __aicore__ static inline void sync_impl() 
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        asc_sync();
#endif
    }

    __aicore__ static inline auto get_core_id_impl() 
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        return asc_get_core_id();
#else   
        return 0;
#endif
    }

    __aicore__ static inline auto get_system_cycle_impl() 
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        return asc_get_system_cycle();
#else   
        return 0;
#endif
    }

    __aicore__ static inline void set_atomic_none_impl() 
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        set_atomic_none();
#endif
    }

    __aicore__ static inline auto get_ctrl_impl() 
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        return asc_get_ctrl();
#else   
        return 0;
#endif
    }

    template<typename T>
    __aicore__ static inline void set_ctrl_impl(T ctrl) 
    {
        if constexpr (CURRENT_ARCH_VERSION == arch_version::v2201) {
            asc_set_ctrl(ctrl);
       }
    }
    
    template <typename T>
    __aicore__ static inline void copy_ub2gm(__gm__ T* dst, __ubuf__ T* src,
        uint16_t n_burst, uint16_t burst_len, uint16_t src_gap, uint16_t dst_gap)
    {
        if constexpr (CURRENT_ARCH_VERSION == arch_version::v2201) {
            asc_copy_ub2gm(reinterpret_cast<__gm__ void*>(dst), reinterpret_cast<__ubuf__ void*>(src),
                n_burst, burst_len, src_gap, dst_gap);
        }
    }

    __aicore__ static inline auto get_g_sys_print_fifo_space() {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2201
        g_sysPrintFifoSpace = nullptr;
#endif  
        return g_sysPrintFifoSpace;
    }
public:
    __aicore__ static inline void call_sync()
    {
        sync_impl();
    }

    __aicore__ static inline auto call_get_core_id()
    {
        return get_core_id_impl();
    }

    __aicore__ static inline auto call_get_system_cycle()
    {
        return get_system_cycle_impl();
    }

    __aicore__ static inline void call_set_atomic_none()
    {
        set_atomic_none_impl();
    }

    __aicore__ static inline decltype(auto) call_get_ctrl()
    {
        return get_ctrl_impl();
    }
    
    template<typename T>
    __aicore__ static inline void call_set_ctrl(T ctrl)
    {
        set_ctrl_impl(ctrl);
    }
    
    template<typename T>
    __aicore__ static inline void call_copy_ub2gm(__gm__ T* dst, __ubuf__ T* src,
        uint16_t n_burst, uint16_t burst_len, uint16_t src_gap, uint16_t dst_gap)
    {
        copy_ub2gm(dst, src, n_burst, burst_len, src_gap, dst_gap);
    }

    __aicore__ static inline auto call_get_g_sys_print_fifo_space()
    {
        return get_g_sys_print_fifo_space();
    }
};

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif