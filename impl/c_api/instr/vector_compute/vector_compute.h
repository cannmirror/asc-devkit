/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef C_API_INSTR_VECTOR_COMPUTE_H
#define C_API_INSTR_VECTOR_COMPUTE_H

#include "get_acc_val/asc_2201/get_acc_val_impl.h"
#include "get_cmp_mask/asc_2201/get_cmp_mask_impl.h"
#include "get_max_min_cnt/asc_2201/get_max_min_cnt_impl.h"
#include "get_rsvd_cnt/asc_2201/get_rsvd_cnt_impl.h"
#include "get_vms4_sr/asc_2201/get_vms4_sr_impl.h"
#include "set_cmp_mask/asc_2201/set_cmp_mask_impl.h"
#include "set_deqscale/asc_2201/set_deqscale_impl.h"
#include "set_mask_count/asc_2201/set_mask_count_impl.h"
#include "set_mask_norm/asc_2201/set_mask_norm_impl.h"
#include "set_vector_mask/asc_2201/set_vector_mask_impl.h"

__aicore__ inline int64_t asc_GetAccVal()
{
    return CApiInternal::asc_GetAccVal();
}

__aicore__ inline void asc_GetCmpMask(__ubuf__ void* dst)
{
    CApiInternal::asc_GetCmpMask(dst);
}

__aicore__ inline void asc_GetReduceMaxMinCnt(half& val, uint32_t& index)
{
    CApiInternal::asc_GetReduceMaxMinCnt(val, index);
}

__aicore__ inline void asc_GetReduceMaxMinCnt(float& val, uint32_t& index)
{
    CApiInternal::asc_GetReduceMaxMinCnt(val, index);
}

__aicore__ inline int64_t asc_GetRsvdCount()
{
    return CApiInternal::asc_GetRsvdCount();
}

__aicore__ inline void asc_GetVms4Sr(uint16_t sortedNum[4])
{
    CApiInternal::asc_GetVms4Sr(sortedNum);
}

__aicore__ inline void asc_SetCmpMask(__ubuf__ void *src)
{
    CApiInternal::asc_SetCmpMask(src);
}

__aicore__ inline void asc_SetDeqScale(half scaleValue)
{
    CApiInternal::asc_SetDeqScale(scaleValue);
}

__aicore__ inline void asc_SetDeqScale(const DeqScaleConfig config)
{
    CApiInternal::asc_SetDeqScale(config);
}

__aicore__ inline void asc_SetDeqScale(__ubuf__ uint64_t* config)
{
    CApiInternal::asc_SetDeqScale(config);
}

__aicore__ inline void asc_SetMaskCount()
{
    CApiInternal::asc_SetMaskCount();
}

__aicore__ inline void asc_SetMaskNorm()
{
    CApiInternal::asc_SetMaskNorm();
}

__aicore__ inline void asc_SetVectorMask(uint64_t mask1, uint64_t mask0)
{
    CApiInternal::asc_SetVectorMask(mask1, mask0);
}

#endif