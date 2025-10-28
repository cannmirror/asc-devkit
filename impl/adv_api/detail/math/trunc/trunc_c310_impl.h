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

/*!
 * \file trunc_c310_impl.h
 * \brief
 */
#ifndef DETAIL_MATH_TRUNC_TRUNC_C310_IMPL_H
#define DETAIL_MATH_TRUNC_TRUNC_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"
 
namespace AscendC {
template <typename T, bool isReuseSource = false>
__aicore__ inline void TruncImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Trunc only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Trunc");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Trunc");
    Truncate<T, RoundMode::CAST_TRUNC>(dstTensor, srcTensor, calCount);
}
template <typename T, bool isReuseSource = false>
__aicore__ inline void TruncImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    TruncImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}
}   // namespace AscendC
#endif  //DETAIL_MATH_TRUNC_TRUNC_C310_IMPL_H
