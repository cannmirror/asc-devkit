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
 * \file printf_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning "impl/c_api/instr_impl/npu_arch_2201/debug_impl/printf_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_PRINTF_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_PRINTF_IMPL_H

#include "instr_impl/npu_arch_2201/utils_impl/debug_utils.h"

__aicore__ __gm__ inline uint8_t* get_g_sysPrintFifoSpace()
{
    return g_sysPrintFifoSpace;
}

__aicore__ __gm__ inline BlockRingBufInfo* get_block_ring_buf_info()
{
    uint32_t block_idx = asc_get_core_id() % ASC_DUMP_CORE_COUNT;
    uint32_t block_length = reinterpret_cast<__gm__ BlockRingBufInfo*>(get_g_sysPrintFifoSpace())->length;
    __gm__ BlockRingBufInfo* ring_buf_info =
        reinterpret_cast<__gm__ BlockRingBufInfo*>(get_g_sysPrintFifoSpace() + block_length * block_idx);
    return ring_buf_info->magic == ASC_MAGIC_NUM_CHECK ? ring_buf_info : nullptr;
}

__aicore__ inline uint32_t get_string_length(__gm__ const char* s)
{
    uint32_t i = 0;
    while (*(s + i) != '\0') {
        i++;
    }
    return i + 1;
}

__aicore__ inline uint32_t get_print_args_len(uint32_t& args_num)
{
    return 0;
}

template <typename... Args>
__aicore__ inline uint32_t get_print_args_len(uint32_t& args_num, Args&&... args);

template <typename... Args>
__aicore__ inline uint32_t get_print_args_len_impl(uint32_t& args_num, __gm__ const char* s, Args&&... args)
{
    constexpr uint32_t param_size = sizeof(uint64_t);
    const uint32_t str_len = get_string_length(s);
    args_num += 1;
    return param_size + str_len + get_print_args_len(args_num, args...);
}

template <typename T, typename... Args>
__aicore__ inline uint32_t get_print_args_len_impl(uint32_t& args_num, T scalar, Args&&... args)
{
    constexpr uint32_t param_size = sizeof(uint64_t);
    args_num += 1;
    return param_size + get_print_args_len(args_num, args...);
}

template <typename... Args>
__aicore__ inline uint32_t get_print_args_len(uint32_t& args_num, Args&&... args)
{
    return get_print_args_len_impl(args_num, args...);
}

__aicore__ constexpr uint32_t align_tlv_len(uint32_t data_len)
{
    constexpr uint32_t num = 7;
    return ((data_len + num) & ~num) + num + 1;
}

template <typename... Args>
__aicore__ inline uint32_t get_print_tlv_len(uint32_t& args_num, __gm__ const char* fmt, Args&&... args)
{
    constexpr uint32_t print_info_len = sizeof(PrintTlvInfoHead);
    const uint32_t fmt_len = get_string_length(fmt);
    const uint32_t args_len = get_print_args_len(args_num, args...);
    return align_tlv_len(print_info_len + args_len + fmt_len); // gm need 8 byte align
}

__aicore__ __gm__ inline RingBufReadInfo* get_ring_buf_read_info(__gm__ BlockRingBufInfo* block_ring_buf_info)
{
    __gm__ uint8_t* block_head = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info);

    return reinterpret_cast<__gm__ RingBufReadInfo*>(block_head + sizeof(BlockRingBufInfo));
}

__aicore__ __gm__ inline RingBufWriteInfo* get_ring_buf_write_info(__gm__ BlockRingBufInfo* block_ring_buf_info)
{
    __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);

    return reinterpret_cast<__gm__ RingBufWriteInfo*>(ring_buf_addr + block_ring_buf_info->ringBufLen);
}

__aicore__ inline void skip_ring_buf_directly(__gm__ RingBufWriteInfo* write_info)
{
    write_info->bufOffset = 0;
    asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(write_info));
    return;
}

__aicore__ inline void skip_ring_buf_with_info(
    __gm__ RingBufWriteInfo* write_info, __gm__ uint8_t* ring_buf_addr, uint32_t ring_buf_len)
{
    __gm__ SkipTlvInfo* skip_info = reinterpret_cast<__gm__ SkipTlvInfo*>(ring_buf_addr + write_info->bufOffset);
    skip_info->type = static_cast<uint32_t>(DumpType::DUMP_SKIP);
    skip_info->length = ring_buf_len - write_info->bufOffset - sizeof(SkipTlvInfo);
    write_info->bufOffset = 0;
    write_info->packIdx += 1;
    asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(skip_info));
    asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(write_info));
    return;
}

__aicore__ inline bool ring_buffer_wait(
    __gm__ RingBufReadInfo* read_info, __gm__ RingBufWriteInfo* write_info, uint32_t tlv_len)
{
    const uint64_t first_time_stamp = static_cast<uint64_t>(asc_get_system_cycle());
    while (write_info->bufOffset < read_info->bufOffset && write_info->bufOffset + tlv_len >= read_info->bufOffset) {
        uint64_t spend_time = static_cast<uint64_t>(asc_get_system_cycle()) - first_time_stamp;
        if (spend_time > ASC_TIMEOUT_CYCLE) {
            return false;
        }
        asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(read_info));
    }
    return true;
}

__aicore__ inline bool wait_ring_buf_begin_read(__gm__ RingBufReadInfo* read_info)
{
    const uint64_t first_time_stamp = static_cast<uint64_t>(asc_get_system_cycle());
    while (read_info->bufOffset == 0) {
        uint64_t spend_time = static_cast<uint64_t>(asc_get_system_cycle()) - first_time_stamp;
        if (spend_time > ASC_TIMEOUT_CYCLE) {
            return false;
        }
        asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(read_info));
    }
    return true;
}

__aicore__ inline bool check_and_wait_ring_buf_space(__gm__ BlockRingBufInfo* block_ring_buf_info, uint32_t tlv_len)
{
    constexpr uint32_t min_tlv_len = sizeof(SkipTlvInfo);

    __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);
    uint32_t ring_buf_len = block_ring_buf_info->ringBufLen;

    __gm__ RingBufReadInfo* read_info = get_ring_buf_read_info(block_ring_buf_info);
    __gm__ RingBufWriteInfo* write_info = get_ring_buf_write_info(block_ring_buf_info);

    if (min_tlv_len >= ring_buf_len || tlv_len > ring_buf_len) {
        return false;
    }
    if (write_info->bufOffset + min_tlv_len >= ring_buf_len) {
        if (!wait_ring_buf_begin_read(read_info)) { // check if reading begins
            return false;
        }
        skip_ring_buf_directly(write_info);
    } else if (write_info->bufOffset + tlv_len > ring_buf_len) {
        if (!wait_ring_buf_begin_read(read_info)) { // check if reading begins
            return false;
        }
        skip_ring_buf_with_info(write_info, ring_buf_addr, ring_buf_len);
    }
    if (write_info->packIdx > 0 &&
        write_info->bufOffset < read_info->bufOffset &&
        write_info->bufOffset + tlv_len >= read_info->bufOffset) {
        return ring_buffer_wait(read_info, write_info, tlv_len);
    }
    return true;
}

__aicore__ __gm__ inline uint8_t* get_ring_buf_tlv(__gm__ BlockRingBufInfo* block_ring_buf_info)
{
    __gm__ RingBufWriteInfo* write_info = get_ring_buf_write_info(block_ring_buf_info);
    __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);
    return ring_buf_addr + write_info->bufOffset;
}

__aicore__ inline void write_ring_buf_tlv_head(
    DumpType print_type, __gm__ PrintTlvInfoHead* print_tlv, uint32_t tlv_len, uint32_t args_num)
{
    print_tlv->type = static_cast<uint32_t>(print_type);
    print_tlv->length = tlv_len - sizeof(uint32_t[2]); // exclude type and length
    print_tlv->resvMem[0] = static_cast<uint32_t>(0U);
    print_tlv->resvMem[1] = static_cast<uint32_t>(0U);
    print_tlv->fmtOffset = (args_num + 1) * sizeof(uint64_t); // include fmt offset
    asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(print_tlv));
}

__aicore__ inline void mem_copy_gm_to_gm(__gm__ uint8_t* dst, __gm__ const uint8_t* src, uint32_t len)
{
    if (dst == nullptr || src == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < len; i++) {
        *(dst + i) = *(src + i);
    }
    asc_dcci_entire_out((__gm__ uint64_t*)(dst));
}

__aicore__ inline void write_string(__gm__ uint8_t* param_addr, uint32_t param_idx, __gm__ const char* s, uint32_t& offset)
{
    __gm__ uint64_t *string_addr = reinterpret_cast<__gm__ uint64_t *>(param_addr) + param_idx;
    __gm__ uint64_t *dst_str_addr = reinterpret_cast<__gm__ uint64_t *>(param_addr + offset);

    // write string value offset
    *((__gm__ uint64_t *)string_addr) = static_cast<uint64_t>(offset - ASC_ONE_PARAM_SIZE * param_idx);
    asc_dcci_entire_out((__gm__ uint64_t*)string_addr);

    // write string content
    __gm__ char *d = (__gm__ char *)(dst_str_addr);
    uint32_t str_len = get_string_length(s);

    for (uint32_t i = 0; i < str_len; i++) {
        *(d + i) = *(s + i);
        asc_dcci_entire_out((__gm__ uint64_t*)d);
    }
    offset += str_len;
}

template <typename T>
__aicore__ inline void write_scalar(__gm__ uint8_t* param_addr, uint32_t param_idx, T scalar)
{
    __gm__ uint64_t *scalar_addr = (__gm__ uint64_t *)param_addr + param_idx;
    *scalar_addr = 0;

    static_assert(!AscendC::Std::is_same_v<T, double>, "printf unsupport double type");

    if constexpr (AscendC::Std::is_same_v<T, half> || AscendC::Std::is_same_v<T, float>) {
        *((__gm__ float *)scalar_addr) = static_cast<float>(scalar);
    } else if constexpr (AscendC::Std::is_same_v<T, bfloat16_t>) {
        *((__gm__ float *)scalar_addr) = ToFloat(scalar);
    } else if constexpr (std::is_signed<T>::value) {
        *((__gm__ int64_t *)scalar_addr) = static_cast<int64_t>(scalar);
    } else if constexpr(std::is_unsigned<T>::value) {
        *((__gm__ uint64_t *)scalar_addr) = static_cast<uint64_t>(scalar);
    } else if constexpr(std::is_pointer<T>::value) {
        *((__gm__ uint64_t *)scalar_addr) = (uintptr_t)scalar;
    } else if constexpr(std::is_enum<T>::value) {
        *((__gm__ uint64_t *)scalar_addr) = static_cast<uint64_t>(scalar);
    }

    asc_dcci_entire_out((__gm__ uint64_t*)scalar_addr);
}

__aicore__ inline void set_param(__gm__ uint8_t* param_addr, uint32_t param_idx, uint32_t& offset)
{
    return;
}

template <typename... Args>
__aicore__ inline void set_param(__gm__ uint8_t* param_addr, uint32_t param_idx, uint32_t& offset, Args&&... args);

template <typename... Args>
__aicore__ inline void set_param_impl(__gm__ uint8_t *param_addr, uint32_t param_idx, uint32_t &offset,
                                    __gm__ const char *s, Args&&... args)
{
    write_string(param_addr, param_idx, s, offset);
    set_param(param_addr, param_idx + 1, offset, args...);
}

template <typename T, typename... Args>
__aicore__ inline void set_param_impl(__gm__ uint8_t* param_addr, uint32_t param_idx, uint32_t& offset, T scalar,
                                    Args&&... args)
{
    write_scalar(param_addr, param_idx, scalar);
    set_param(param_addr, param_idx + 1, offset, args...);
}

template <typename... Args>
__aicore__ inline void set_param(__gm__ uint8_t* param_addr, uint32_t param_idx, uint32_t& offset, Args&&... args)
{
    set_param_impl(param_addr, param_idx, offset, args...);
}

template <typename... Args>
__aicore__ inline void write_ring_buf_tlv_data(__gm__ PrintTlvInfoHead* print_tlv, __gm__ const char* fmt, Args&&... args)
{
    const uint32_t str_len = get_string_length(fmt);
    __gm__ uint8_t* param_addr =
        reinterpret_cast<__gm__ uint8_t*>(print_tlv + 1);
    __gm__ uint8_t* fmt_addr = param_addr + print_tlv->fmtOffset - sizeof(uint64_t);
    mem_copy_gm_to_gm(fmt_addr, reinterpret_cast<__gm__ const uint8_t*>(fmt), str_len);
    uint32_t str_param_offset = print_tlv->fmtOffset + str_len;
    set_param(param_addr, 0, str_param_offset, args...);
}

__aicore__ inline void update_write_info(__gm__ RingBufWriteInfo* write_info, uint32_t tlv_len)
{
    write_info->bufOffset += tlv_len;
    write_info->packIdx += 1;
    asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(write_info));
}

template <class... Args>
__aicore__ inline void print_ring_buf_impl(DumpType print_type, __gm__ const char* fmt, Args&&... args)
{
    __gm__ BlockRingBufInfo* block_ring_buf_info = get_block_ring_buf_info();
    if (block_ring_buf_info == nullptr) {
        return;
    }
    uint32_t args_num = 0;
    const uint32_t tlv_len = get_print_tlv_len(args_num, fmt, args...);
    if (!check_and_wait_ring_buf_space(block_ring_buf_info, tlv_len)) {
        return;
    }

    __gm__ PrintTlvInfoHead* print_tlv = reinterpret_cast<__gm__ PrintTlvInfoHead*>(get_ring_buf_tlv(block_ring_buf_info));

    write_ring_buf_tlv_head(print_type, print_tlv, tlv_len, args_num);
    write_ring_buf_tlv_data(print_tlv, fmt, args...);

    __gm__ RingBufWriteInfo* write_info = get_ring_buf_write_info(block_ring_buf_info);

    update_write_info(write_info, tlv_len);
}

__aicore__ inline uint64_t preprocess_printf()
{
    enable_printf();
    uint64_t ctrl_value = asc_get_ctrl();
    set_atomic_none();
    return ctrl_value;
}

__aicore__ inline void postprocess_printf(uint64_t ctrl_value)
{
    asc_set_ctrl(ctrl_value);
}

template <class... Args>
__aicore__ inline void simd_printf_compute(DumpType print_type, __gm__ const char* fmt, Args&&... args)
{
    uint64_t ctrl_value = preprocess_printf();

    if (get_g_sysPrintFifoSpace() != nullptr) {
        print_ring_buf_impl(print_type, fmt, args...);
    }

    postprocess_printf(ctrl_value);
}

template <class... Args>
static __attribute__((noinline)) __aicore__ void simd_printf_impl(DumpType print_type, __gm__ const char* fmt, Args&&... args)
{
#if defined (ASCENDC_DUMP) && (ASCENDC_DUMP == 1)
    simd_printf_compute(print_type, fmt, args...);
#endif
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif