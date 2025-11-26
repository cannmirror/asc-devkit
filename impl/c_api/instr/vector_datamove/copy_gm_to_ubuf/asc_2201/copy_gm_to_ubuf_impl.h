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
 * \file copy_gm_to_ubuf_impl.h
 * \brief
 */
#ifndef IMPL_INSTR_VECTOR_DMAMOVE_COPY_GM_TO_UBUF_ASC_2201_COPY_GM_TO_UBUF_IMPL_H
#define IMPL_INSTR_VECTOR_DMAMOVE_COPY_GM_TO_UBUF_ASC_2201_COPY_GM_TO_UBUF_IMPL_H

#include "c_api/c_api_interf_util.h"
#include "c_api/utils/c_api_constants.h"

namespace CApiInternal {

__aicore__ inline void copy_gm2ub_impl(__ubuf__ void* dst, __gm__ void* src, uint32_t size)
{
    copy_gm_to_ubuf(dst, src, 0, 1, size / CAPI_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void copy_gm2ub_impl(__ubuf__ void* dst, __gm__ void* src, const asc_copy_config& config)
{
    copy_gm_to_ubuf(dst, src, config.config);
}

__aicore__ inline void copy_gm2ub_sync_impl(__ubuf__ void* dst, __gm__ void* src, uint32_t size)
{
    copy_gm2ub_impl(dst, src, size);
    pipe_barrier(pipe_t::PIPE_ALL);
}

}

#endif