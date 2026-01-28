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
 * \file dump_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning "impl/c_api/instr_impl/npu_arch_2201/debug_impl/dump_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_DUMP_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_DEBUG_IMPL_DUMP_IMPL_H

#include "instr_impl/npu_arch_2201/utils_impl/debug_utils.h"
#include "instr_impl/npu_arch_2201/utils_impl/instruction_forward.h"
class print_ring_buf_head_base {
public:
    __aicore__ __gm__ inline BlockRingBufInfo* call_get_block_ring_buf_info()
    {
        return get_block_ring_buf_info();
    }
private:
    __aicore__ __gm__ inline BlockRingBufInfo* get_block_ring_buf_info()
    {
        uint32_t block_idx = instruction_forward::call_get_core_id() % ASC_DUMP_CORE_COUNT;
        uint32_t block_length = reinterpret_cast<__gm__ BlockRingBufInfo*>(instruction_forward::call_get_g_sys_print_fifo_space())->length;
        __gm__ BlockRingBufInfo* ring_buf_info =
            reinterpret_cast<__gm__ BlockRingBufInfo*>(instruction_forward::call_get_g_sys_print_fifo_space() + block_length * block_idx);
        return ring_buf_info->magic == ASC_MAGIC_NUM_CHECK ? ring_buf_info : nullptr;
    }
};

class check_print_ring_buf_base {
public:
    __aicore__ inline bool call_check_and_wait_ring_buf_space(__gm__ BlockRingBufInfo* block_ring_buf_info, const uint32_t tlv_len)
    {
        return check_and_wait_ring_buf_space(block_ring_buf_info, tlv_len);
    }
private:
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

    __aicore__ inline bool wait_ring_buf_begin_read(__gm__ RingBufReadInfo* read_info)
    {
        const uint64_t& first_time_stamp = static_cast<uint64_t>(instruction_forward::call_get_system_cycle());
        while (read_info->bufOffset == 0) {
            uint64_t spend_time = static_cast<uint64_t>(instruction_forward::call_get_system_cycle()) - first_time_stamp;
            if (spend_time > ASC_TIMEOUT_CYCLE) {
                return false;
            }
            asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(read_info));
        }
        return true;
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
        __gm__ SkipTlvInfo* skipInfo = reinterpret_cast<__gm__ SkipTlvInfo*>(ring_buf_addr + write_info->bufOffset);
        skipInfo->type = static_cast<uint32_t>(DumpType::DUMP_SKIP);
        skipInfo->length = ring_buf_len - write_info->bufOffset - sizeof(SkipTlvInfo);
        write_info->bufOffset = 0;
        write_info->packIdx += 1;
        asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(skipInfo));
        asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(write_info));
        return;
    }

    __aicore__ inline bool ring_buffer_wait(__gm__ RingBufReadInfo* read_info, __gm__ RingBufWriteInfo* write_info,
                                      uint32_t tlv_len)
    {
        const uint64_t& first_time_stamp = static_cast<uint64_t>(instruction_forward::call_get_system_cycle());
        while (write_info->bufOffset < read_info->bufOffset && write_info->bufOffset + tlv_len >= read_info->bufOffset) {
            uint64_t spend_time = static_cast<uint64_t>(instruction_forward::call_get_system_cycle()) - first_time_stamp;
            if (spend_time > ASC_TIMEOUT_CYCLE) {
                return false;
            }
            asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(read_info));
        }
        return true;
    }

    __aicore__ inline bool check_and_wait_ring_buf_space(__gm__ BlockRingBufInfo* block_ring_buf_info, const uint32_t tlv_len)
    {
        __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);
        uint32_t ring_buf_len = block_ring_buf_info->ringBufLen;

        if (MIN_TLV_LEN >= ring_buf_len || tlv_len > ring_buf_len) {
            return false;
        }

        __gm__ RingBufReadInfo* read_info = get_ring_buf_read_info(block_ring_buf_info);
        __gm__ RingBufWriteInfo* write_info = get_ring_buf_write_info(block_ring_buf_info);

        if (write_info->bufOffset + MIN_TLV_LEN >= ring_buf_len) {
            if (!wait_ring_buf_begin_read(read_info)) { // check read is begin
                return false;
            }
            skip_ring_buf_directly(write_info);
        } else if (write_info->bufOffset + tlv_len > ring_buf_len) {
            if (!wait_ring_buf_begin_read(read_info)) { // check read is begin
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
};

class tlv_base {
public:
    __aicore__ __gm__ inline uint8_t* call_get_ring_buf_tlv(__gm__ BlockRingBufInfo* block_ring_buf_info)
    {
        return get_ring_buf_tlv(block_ring_buf_info);
    }

    template <Hardware hardware, typename T, typename U>
    __aicore__ inline void call_write_ring_buf_tlv_head(U src, __gm__ DumpTensorTlvInfoHead* dump_tensor_tlv,
        uint32_t align_dump_len, uint32_t desc, uint32_t dump_size)
    {
        write_ring_buf_tlv_head<hardware, T, U>(src, dump_tensor_tlv, align_dump_len, desc, dump_size);
    }

    template <Hardware hardware, typename T, typename U>
    __aicore__ inline void call_write_ring_buf_tlv_data(U src, __gm__ DumpTensorTlvInfoHead* dump_tensor_tlv,
        uint32_t align_dump_len, uint32_t dump_size)
    {
        write_ring_buf_tlv_data<hardware, T, U>(src, dump_tensor_tlv, align_dump_len, dump_size);
    }

    __aicore__ inline void call_update_write_info(__gm__ RingBufWriteInfo* write_info, uint32_t tlv_len)
    {
        update_write_info(write_info, tlv_len);
    }
private:
    __aicore__ __gm__ inline RingBufWriteInfo* get_ring_buf_write_info(__gm__ BlockRingBufInfo* block_ring_buf_info)
    {
        __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);
        return reinterpret_cast<__gm__ RingBufWriteInfo*>(ring_buf_addr + block_ring_buf_info->ringBufLen);
    }

    __aicore__ __gm__ inline uint8_t* get_ring_buf_tlv(__gm__ BlockRingBufInfo* block_ring_buf_info)
    {
        __gm__ RingBufWriteInfo* write_info = get_ring_buf_write_info(block_ring_buf_info);
        __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);
        return ring_buf_addr + write_info->bufOffset;
    }

    template <Hardware hardware, typename T, typename U>
    __aicore__ inline void write_ring_buf_tlv_head(U src, __gm__ DumpTensorTlvInfoHead* dump_tensor_tlv,
        uint32_t align_dump_len, uint32_t desc, uint32_t dump_size)
    {
        dump_tensor_tlv->type = static_cast<uint32_t>(DumpType::DUMP_TENSOR);
        dump_tensor_tlv->length = sizeof(DumpTensorTlvInfoHead) - sizeof(uint32_t[2]) + align_dump_len;
        dump_tensor_tlv->tensorAddr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src));
        dump_tensor_tlv->dataType = static_cast<uint32_t>(get_dump_datatype<T>());
        dump_tensor_tlv->desc = desc;
        dump_tensor_tlv->bufferId = static_cast<uint32_t>(0U);
        dump_tensor_tlv->position = static_cast<uint16_t>(hardware);
        dump_tensor_tlv->resv0 = static_cast<uint16_t>(0U);
        dump_tensor_tlv->dim = static_cast<uint32_t>(0U);
        for (uint32_t i = 0; i < K_MAX_SHAPE_DIM; ++i) {
            dump_tensor_tlv->shape[i] = static_cast<uint32_t>(0U);
        }
        dump_tensor_tlv->resv1 = static_cast<uint32_t>(0U);
        dump_tensor_tlv->dumpSize = dump_size * sizeof(T);
        asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(dump_tensor_tlv));
    }

    template <Hardware hardware, typename T, typename U>
    __aicore__ inline void write_ring_buf_tlv_data(U src, __gm__ DumpTensorTlvInfoHead* dump_tensor_tlv,
        uint32_t align_dump_len, uint32_t dump_size)
    {
        if constexpr (hardware == Hardware::GM) {
            __gm__ T* dst = reinterpret_cast<__gm__ T*>(dump_tensor_tlv + 1);
            if (dst == nullptr || src == nullptr) {
                return;
            }
            uint32_t dump_len = dump_size * sizeof(T);
            for (uint32_t i = 0; i < dump_len; i++) {
                *(dst + i) = *(src + i);
            }
            asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(dst));
        } else if constexpr (hardware == Hardware::UB) {
            __gm__ T* dump_addr = reinterpret_cast<__gm__ T*>(dump_tensor_tlv + 1);
            uint16_t burst_len = static_cast<uint16_t>(align_dump_len / ASC_C_API_ONE_DATABLOCK_SIZE);
            instruction_forward::call_sync();
            if (dump_tensor_tlv->position == static_cast<uint16_t>(Hardware::UB)) {
                instruction_forward::call_copy_ub2gm(dump_addr, src, 0, 1, burst_len, 0, 0);
            }
            instruction_forward::call_sync();
            asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(dump_addr));
        }
    }
    __aicore__ inline void update_write_info(__gm__ RingBufWriteInfo* write_info, uint32_t tlv_len)
    {
        write_info->bufOffset += tlv_len;
        write_info->packIdx += 1;
        asc_dcci_entire_out(reinterpret_cast<__gm__ uint64_t*>(write_info));
    }
};

class dump_base : public print_ring_buf_head_base, public check_print_ring_buf_base, public tlv_base
{
public:
    template <Hardware hardware, typename T, typename U>
    __aicore__ inline void call_asc_dump_ring_impl(U src, uint32_t desc, uint32_t dump_size)
    {
        auto ctrl_value = asc_dump_pre_process();
        if (instruction_forward::call_get_g_sys_print_fifo_space() != nullptr) {
            asc_dump_ring_impl<hardware, T>(src, desc, dump_size);
        }
        asc_dump_post_process(ctrl_value);
    }
private:
    __aicore__ __gm__ inline RingBufWriteInfo* get_ring_buf_write_info(__gm__ BlockRingBufInfo* block_ring_buf_info)
    {
        __gm__ uint8_t* ring_buf_addr = reinterpret_cast<__gm__ uint8_t*>(block_ring_buf_info->ringBufAddr);
        return reinterpret_cast<__gm__ RingBufWriteInfo*>(ring_buf_addr + block_ring_buf_info->ringBufLen);
    }

    template <Hardware hardware, typename T, typename U>
    __aicore__ inline void asc_dump_ring_impl(U src, uint32_t desc, uint32_t dump_size) 
    {
        static_assert((get_dump_datatype<T>() != DumpTensorDataType::ACL_MAX), "dump tensor not support this data type");
        if (dump_size == 0) {
            return;
        }

        __gm__ BlockRingBufInfo* block_ring_buf_info = call_get_block_ring_buf_info();
        if (block_ring_buf_info == nullptr) {
            return;
        }
        uint32_t align_dump_len = align_up(dump_size * sizeof(T), ASC_C_API_ONE_DATABLOCK_SIZE);
        uint32_t tlv_len = sizeof(DumpTensorTlvInfoHead) + align_dump_len;
        if (!call_check_and_wait_ring_buf_space(block_ring_buf_info, tlv_len)) {
            return;
        }

        __gm__ DumpTensorTlvInfoHead* dump_tensor_tlv =
            reinterpret_cast<__gm__ DumpTensorTlvInfoHead*>(call_get_ring_buf_tlv(block_ring_buf_info));

        call_write_ring_buf_tlv_head<hardware, T>(src, dump_tensor_tlv, align_dump_len, desc, dump_size);
        call_write_ring_buf_tlv_data<hardware, T>(src, dump_tensor_tlv, align_dump_len, dump_size);

        __gm__ RingBufWriteInfo* write_info = get_ring_buf_write_info(block_ring_buf_info);

        call_update_write_info(write_info, tlv_len);
    }

    __aicore__ inline uint64_t asc_dump_pre_process() 
    {
        enable_printf();
        uint64_t ret = instruction_forward::call_get_ctrl();
        instruction_forward::call_set_atomic_none();
        return ret;
    }

    template <typename T>
    __aicore__ inline void asc_dump_post_process(T ctrl_value) 
    {
        instruction_forward::call_set_ctrl(ctrl_value);
    }
};

class dump_v2201_gm : public dump_base
{
public:
    template<typename T>
    __attribute__((noinline)) __aicore__ void run(__gm__ T* input, uint32_t desc, uint32_t dump_size)
    {
        call_asc_dump_ring_impl<Hardware::GM, T>(input, desc, dump_size);
    }
};

class dump_v2201_ubuf : public dump_base
{
public:
    template<typename T>
    __attribute__((noinline)) __aicore__ void run(__ubuf__ T* input, uint32_t desc, uint32_t dump_size)
    {
        call_asc_dump_ring_impl<Hardware::UB, T>(input, desc, dump_size);
    }
};
#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif