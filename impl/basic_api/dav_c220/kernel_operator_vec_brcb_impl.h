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
 * \file kernel_operator_vec_brcb_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H
#include "kernel_struct_brcb.h"

namespace AscendC {
/* **************************************************************************************************
 * Brcb                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void BrcbImpl(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t repeatTime,
    const BrcbRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        ResetMask();
        if constexpr(sizeof(T) == B16_BYTE_SIZE) {
            vbrcb((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src0, repeatParams.dstBlkStride,
                repeatParams.dstRepStride, repeatTime);
        } else if constexpr(sizeof(T) == B32_BYTE_SIZE) {
            vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, repeatParams.dstBlkStride,
                repeatParams.dstRepStride, repeatTime);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Brcb, current api support dtype "
                "combination is src and dst both: half / bfloat16_t / int16_t / uint16_t / float / int32_t / "
                "uint32_t.");});
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H