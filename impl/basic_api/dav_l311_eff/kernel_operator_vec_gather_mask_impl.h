/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_vec_gather_mask_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
#include "kernel_struct_gather.h"

namespace AscendC {
template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint16_t* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Gather is not supported on current device!"); });
}

template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Gather is not supported on current device!"); });
}

__aicore__ inline int64_t GetGatherMaskRemainCountImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetGatherMaskRemainCount on current device");
    return 0;
}

#define REGISTER_GATHER_MASK_B16(data_type, vector_type)                                                               \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, __ubuf__ uint16_t* src1,   \
    const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)             \
    {                                                                                                                  \
        ASCENDC_ASSERT((false), {                                                                                      \
                KERNEL_LOG(KERNEL_ERROR,                                                                               \
                    "vector calculate is not support on current device!");                                                   \
            });                                                                                                        \
    }                                                                                                                  \

REGISTER_GATHER_MASK_B16(half, f16)
REGISTER_GATHER_MASK_B16(uint16_t, u16)
REGISTER_GATHER_MASK_B16(int16_t, s16)

#define REGISTER_GATHER_MASK_B32(data_type, vector_type)                                                               \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, __ubuf__ uint32_t* src1,   \
        const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)         \
    {                                                                                                                  \
        ASCENDC_ASSERT((false), {                                                                                      \
                KERNEL_LOG(KERNEL_ERROR,                                                                               \
                    "vector calculate is not support on current device!");                                                   \
            });                                                                                                        \
    }                                                                                                                  \

REGISTER_GATHER_MASK_B32(float, f32)
REGISTER_GATHER_MASK_B32(uint32_t, u32)
REGISTER_GATHER_MASK_B32(int32_t, s32)


template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t src1Pattern,
    const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Gather is not supported on current device!"); });
}

#define REGISTER_GATHER_MASK_SOLID_B16(data_type, vector_type)                                                         \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, const uint8_t src1Pattern, \
        const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)         \
    {                                                                                                       \
            ASCENDC_ASSERT((false), {                                                                       \
                KERNEL_LOG(KERNEL_ERROR,                                                                    \
                    " vector calculate is not support on current device!");                   					\
            });                                                                                             \
    }																									    \

REGISTER_GATHER_MASK_SOLID_B16(half, f16)
REGISTER_GATHER_MASK_SOLID_B16(uint16_t, u16)
REGISTER_GATHER_MASK_SOLID_B16(int16_t, s16)

#define REGISTER_GATHER_MASK_SOLID_B32(data_type, vector_type)                                                         \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, const uint8_t src1Pattern, \
        const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)         \
    {                                                                                                       \
            ASCENDC_ASSERT((false), {                                                                       \
                KERNEL_LOG(KERNEL_ERROR,                                                                    \
                    " vector calculate is not support on current device!");                   					\
            });                                                                                             \
    }		                                                                                                 \

REGISTER_GATHER_MASK_SOLID_B32(float, f32)
REGISTER_GATHER_MASK_SOLID_B32(uint32_t, u32)
REGISTER_GATHER_MASK_SOLID_B32(int32_t, s32)
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_REDUCEV2_IMPL_H
