/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file half2int16_impl.h
 * \brief
 */
#ifndef IMPL_INSTR_VECTOR_COMPUTE_VCONV_HALF2INT16_ASC_2201_HALF2INT16_IMPL_H
#define IMPL_INSTR_VECTOR_COMPUTE_VCONV_HALF2INT16_ASC_2201_HALF2INT16_IMPL_H

#include "c_api/c_api_interf_util.h"

namespace CApiInternal {

template<typename T, typename U>
__aicore__ inline void half2int16_a_impl(__ubuf__ T* dst, __ubuf__ U* src, const asc_unary_config& config)
{
    vconv_f162s16a(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_block_stride),
                   static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.dst_repeat_stride),
                   static_cast<uint16_t>(config.src_repeat_stride));
}

template<typename T, typename U>
__aicore__ inline void half2int16_a_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    half2int16_a_impl<T, U>(dst, src, CAPI_DEFAULT_UNARY_CFG);
    set_mask_norm();
}

template<typename T, typename U>
__aicore__ inline void half2int16_a_sync_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    half2int16_a_impl<T, U>(dst, src, count);
    pipe_barrier(pipe_t::PIPE_ALL);
}

template<typename T, typename U>
__aicore__ inline void half2int16_c_impl(__ubuf__ T* dst, __ubuf__ U* src, const asc_unary_config& config)
{
    vconv_f162s16c(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_block_stride),
                   static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.dst_repeat_stride),
                   static_cast<uint16_t>(config.src_repeat_stride));
}

template<typename T, typename U>
__aicore__ inline void half2int16_c_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    half2int16_c_impl<T, U>(dst, src, CAPI_DEFAULT_UNARY_CFG);
    set_mask_norm();
}

template<typename T, typename U>
__aicore__ inline void half2int16_c_sync_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    half2int16_c_impl<T, U>(dst, src, count);
    pipe_barrier(pipe_t::PIPE_ALL);
}

template<typename T, typename U>
__aicore__ inline void half2int16_f_impl(__ubuf__ T* dst, __ubuf__ U* src, const asc_unary_config& config)
{
    vconv_f162s16f(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_block_stride),
                   static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.dst_repeat_stride),
                   static_cast<uint16_t>(config.src_repeat_stride));
}

template<typename T, typename U>
__aicore__ inline void half2int16_f_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    half2int16_f_impl<T, U>(dst, src, CAPI_DEFAULT_UNARY_CFG);
    set_mask_norm();
}

template<typename T, typename U>
__aicore__ inline void half2int16_f_sync_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    half2int16_f_impl<T, U>(dst, src, count);
    pipe_barrier(pipe_t::PIPE_ALL);
}

template<typename T, typename U>
__aicore__ inline void half2int16_r_impl(__ubuf__ T* dst, __ubuf__ U* src, const asc_unary_config& config)
{
    vconv_f162s16r(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_block_stride),
                   static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.dst_repeat_stride),
                   static_cast<uint16_t>(config.src_repeat_stride));
}

template<typename T, typename U>
__aicore__ inline void half2int16_r_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    half2int16_r_impl<T, U>(dst, src, CAPI_DEFAULT_UNARY_CFG);
    set_mask_norm();
}

template<typename T, typename U>
__aicore__ inline void half2int16_r_sync_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    half2int16_r_impl<T, U>(dst, src, count);
    pipe_barrier(pipe_t::PIPE_ALL);
}

template<typename T, typename U>
__aicore__ inline void half2int16_z_impl(__ubuf__ T* dst, __ubuf__ U* src, const asc_unary_config& config)
{
    vconv_f162s16z(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_block_stride),
                   static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.dst_repeat_stride),
                   static_cast<uint16_t>(config.src_repeat_stride));
}

template<typename T, typename U>
__aicore__ inline void half2int16_z_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    half2int16_z_impl<T, U>(dst, src, CAPI_DEFAULT_UNARY_CFG);
    set_mask_norm();
}

template<typename T, typename U>
__aicore__ inline void half2int16_z_sync_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    half2int16_z_impl<T, U>(dst, src, count);
    pipe_barrier(pipe_t::PIPE_ALL);
}

}

#endif