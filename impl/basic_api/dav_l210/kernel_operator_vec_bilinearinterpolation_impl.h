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
 * \file kernel_operator_vec_bilinearinterpolation_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPOLATION_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPOLATION_IMPL_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
    const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask, uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    ASCENDC_ASSERT(false,
    { KERNEL_LOG(KERNEL_ERROR, "unsupported BilinearInterpolation"); });
}

template <typename T>
__aicore__ inline void BilinearInterpolationCalc(LocalTensor<T> &dst, LocalTensor<T> &src0,
    LocalTensor<uint32_t> &src0Offset, LocalTensor<T> &src1, uint64_t mask[2], uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    ASCENDC_ASSERT(false,
    { KERNEL_LOG(KERNEL_ERROR, "unsupported BilinearInterpolation"); });
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPOLATION_IMPL_H