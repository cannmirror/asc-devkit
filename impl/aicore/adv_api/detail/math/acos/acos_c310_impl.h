/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file acos_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_ACOS_ACOS_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_ACOS_ACOS_C310_IMPL_H
#include "kernel_tensor.h"
#include "../asin/asin_c310_impl.h"
#include "../../common/check.h"

namespace AscendC {
// Compute acos values according to formula: arccos(x) = PI*0.5 - arcsin(x).
template <typename T>
__aicore__ inline void AcosCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t calSize)
{
    constexpr bool convertToCos = true;
    AsinCompute<T, convertToCos>(dst, src, calSize);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcosImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Acos only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Acos");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Acos");
    AcosCompute(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcosImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    AcosImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_ACOS_ACOS_C310_IMPL_H
