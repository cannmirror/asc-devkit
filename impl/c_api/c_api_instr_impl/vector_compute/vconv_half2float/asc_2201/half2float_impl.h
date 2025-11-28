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
 * \file half2float_impl.h
 * \brief
 */
#ifndef IMPL_INSTR_VECTOR_COMPUTE_VCONV_HALF2FLOAT_ASC_2201_HALF2FLOAT_IMPL_H
#define IMPL_INSTR_VECTOR_COMPUTE_VCONV_HALF2FLOAT_ASC_2201_HALF2FLOAT_IMPL_H

#include "c_api_interf_util.h"

namespace CApiInternal {

template<typename T, typename U>
__aicore__ inline void half2float_impl(__ubuf__ T* dst, __ubuf__ U* src, const asc_unary_config& config)
{
    vconv_f162f32(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_block_stride),
                   static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.dst_repeat_stride),
                   static_cast<uint16_t>(config.src_repeat_stride));
}

template<typename T, typename U>
__aicore__ inline void half2float_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    asc_unary_config config{};
    config.dst_block_stride = 1;
    config.src_block_stride = 1;
    config.dst_repeat_stride = 8;
    config.src_repeat_stride = 4;
    config.repeat = 1;
    half2float_impl<T, U>(dst, src, config);
    set_mask_norm();
}

template<typename T, typename U>
__aicore__ inline void half2float_sync_impl(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count)
{
    half2float_impl<T, U>(dst, src, count);
    pipe_barrier(pipe_t::PIPE_ALL);
}

}

#endif