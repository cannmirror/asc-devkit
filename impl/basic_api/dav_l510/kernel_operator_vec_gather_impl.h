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
 * \file kernel_operator_vec_gather_impl.h
 * \brief AscendC l510 support vector gather api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H

namespace AscendC {

/* **************************************************************************************************
 * Gather                                                                                           *
 * **************************************************************************************************/
// gatherb::Level 0
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* offset,
    const uint32_t srcLength, uint8_t repeatTime, const GatherRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b16 or b32"); });
}

// for gather op
#define GATHER_OP_B8_MASK_COUNT_MODE(T, U)                                    \
    {                                                                         \
        ASCENDC_ASSERT((false), {                                             \
                        KERNEL_LOG(KERNEL_ERROR,                              \
                            "vector calculate  not supported on current device !"); \
            });                                                               \
    }

#define GATHER_OP_B16B32_MASK_COUNT_MODE(T, U)                                \
    {                                                                         \
        ASCENDC_ASSERT((false), {                                             \
                        KERNEL_LOG(KERNEL_ERROR,                              \
                            "vector calculate  not supported on current device !"); \
            });                                                               \
    }

// gather::Level 0 - mask count mode
template <typename T, typename U = uint32_t>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ U* srcOffset,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint16_t dstRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b16 or b32"); });
}

#define GATHER_OP_B16_MASK_BIT_MODE(T, U)                                     \
    {                                                                         \
        ASCENDC_ASSERT((false), {                                             \
                        KERNEL_LOG(KERNEL_ERROR,                              \
                            "vector calculate  not supported on current device !"); \
            });                                                               \
    }

#define GATHER_OP_B32_MASK_BIT_MODE(T, U)                                     \
    {                                                                         \
        ASCENDC_ASSERT((false), {                                             \
                        KERNEL_LOG(KERNEL_ERROR,                              \
                            "vector calculate  not supported on current device !"); \
            });                                                               \
    }
// gather::Level 0 - mask bit mode
template <typename T, typename U = uint32_t>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ U* srcOffset,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask[2], const uint8_t repeatTime,
    const uint16_t dstRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b16 or b32"); });
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
