/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef C_API_INSTR_SIMD_ATOMIC_H
#define C_API_INSTR_SIMD_ATOMIC_H

#include "dcci/asc_2201/dcci_impl.h"
#include "set_atomic_add/asc_2201/set_atomic_add_impl.h"
#include "set_atomic_bf16/asc_2201/set_atomic_bf16_impl.h"
#include "set_atomic_f16/asc_2201/set_atomic_f16_impl.h"
#include "set_atomic_f32/asc_2201/set_atomic_f32_impl.h"
#include "set_atomic_max/asc_2201/set_atomic_max_impl.h"
#include "set_atomic_min/asc_2201/set_atomic_min_impl.h"
#include "set_atomic_none/asc_2201/set_atomic_none_impl.h"
#include "set_atomic_s8/asc_2201/set_atomic_s8_impl.h"
#include "set_atomic_s16/asc_2201/set_atomic_s16_impl.h"
#include "set_atomic_s32/asc_2201/set_atomic_s32_impl.h"

__aicore__ inline void asc_DataCacheCleanAndInvalid(__gm__ void *dst, uint64_t entire)
{
    CApiInternal::asc_DataCacheCleanAndInvalid(dst, entire);
}

__aicore__ inline void asc_SetAtomicAdd()
{
    CApiInternal::asc_SetAtomicAdd();
}

__aicore__ inline void asc_SetAtomicBfloat16()
{
    CApiInternal::asc_SetAtomicBfloat16();
}

__aicore__ inline void asc_SetAtomicHalf()
{
    CApiInternal::asc_SetAtomicHalf();
}

__aicore__ inline void asc_SetAtomicFloat()
{
    CApiInternal::asc_SetAtomicFloat();
}

__aicore__ inline void asc_SetAtomicMax()
{
    CApiInternal::asc_SetAtomicMax();
}

__aicore__ inline void asc_SetAtomicMin()
{
    CApiInternal::asc_SetAtomicMin();
}

__aicore__ inline void asc_SetAtomicNone()
{
    CApiInternal::asc_SetAtomicNone();
}

__aicore__ inline void asc_SetAtomicS8()
{
    CApiInternal::asc_SetAtomicS8();
}

__aicore__ inline void asc_SetAtomicS16()
{
    CApiInternal::asc_SetAtomicS16();
}

__aicore__ inline void asc_SetAtomicS32()
{
    CApiInternal::asc_SetAtomicS32();
}

#endif