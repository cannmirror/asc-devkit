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
 * \file repeat_reduce_max_impl.h
 * \brief
 */
#ifndef IMPL_INSTR_VECTOR_COMPUTE_VCMAX_ASC_2201_REPEAT_REDUCE_MAX_IMPL_H
#define IMPL_INSTR_VECTOR_COMPUTE_VCMAX_ASC_2201_REPEAT_REDUCE_MAX_IMPL_H

#include "c_api/c_api_interf_util.h"

namespace CApiInternal {

template <typename T>
__aicore__ void repeat_reduce_max_impl(__ubuf__ T* dst, __ubuf__ T* src, const asc_repeat_reduce_config& config, order_t order)
{
    vcmax(dst, src, static_cast<uint8_t>(config.repeat), static_cast<uint16_t>(config.dst_repeat_stride),
           static_cast<uint16_t>(config.src_block_stride), static_cast<uint16_t>(config.src_repeat_stride), order);
}

template <typename T>
__aicore__ void repeat_reduce_max_impl(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count, order_t order)
{
    set_mask_count();
    set_vector_mask(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
    repeat_reduce_max_impl<T>(dst, src, CAPI_REPEAT_DEFAULT_REDUCE_CFG, order);
    set_mask_norm();
}

template <typename T>
__aicore__ void repeat_reduce_max_sync_impl(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count, order_t order)
{
    repeat_reduce_max_impl<T>(dst, src, count, order);
    pipe_barrier(pipe_t::PIPE_ALL);
}

}

#endif