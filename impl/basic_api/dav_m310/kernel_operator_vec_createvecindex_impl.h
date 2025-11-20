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
 * \file kernel_operator_vec_createvecindex_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dst, const T &firstValue, uint64_t mask,
    uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device don't support CreateVecIndex"); });
}

template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dst, const T &firstValue, uint64_t mask[],
    uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device don't support CreateVecIndex"); });
}

template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> dst, const T &firstValue, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device don't support CreateVecIndex"); });
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H