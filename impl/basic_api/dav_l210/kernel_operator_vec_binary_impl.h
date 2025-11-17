/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_vec_binary_impl.h
 * \brief AscendC l210 support vector binary memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_binary_continuous_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const int32_t &count)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)count;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu on current device"); });
}

// AddDeqRelu::Level 0
template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTime;
    (void)repeatParams;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu on current device"); });
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTime;
    (void)repeatParams;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu on current device"); });
}
}

#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
