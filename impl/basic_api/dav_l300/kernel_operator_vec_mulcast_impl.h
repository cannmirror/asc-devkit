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

/* !
 * \file kernel_operator_vec_mulcast_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false,
    { KERNEL_LOG(KERNEL_ERROR, "unsupported MulCast"); });
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask[2], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false,
    { KERNEL_LOG(KERNEL_ERROR, "unsupported MulCast"); });
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint32_t calCount)
{
    ASCENDC_ASSERT(false,
    { KERNEL_LOG(KERNEL_ERROR, "unsupported MulCast"); });
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H