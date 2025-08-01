/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file floor_C310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_FLOOR_FLOOR_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_FLOOR_FLOOR_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

namespace AscendC {

template <typename T, bool isReuseSource = false>
__aicore__ inline void FloorImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Floor");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Floor");
    Truncate<T, RoundMode::CAST_FLOOR>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void FloorImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    FloorImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_FLOOR_FLOOR_C310_IMPL_H