/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file clamp_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_CLAMP_CLAMP_COMMON_IMPL_H
#define IMPL_MATH_CLAMP_CLAMP_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount, CLAMPMODE selMode)
{
    if (selMode == CLAMPMODE::CLAMP_MIN) {
        CHECK_FUNC_HIGHLEVEL_API(ClampMin, (T, isReuseSource), (dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount));
        Maxs(dstTensor, srcTensor, scalar, calCount);
    } else if (selMode == CLAMPMODE::CLAMP_MAX) {
        CHECK_FUNC_HIGHLEVEL_API(ClampMax, (T, isReuseSource), (dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount));
        Mins(dstTensor, srcTensor, scalar, calCount);
    }
}
/* **************************************************************************************************
 * ClampMax                                           *
 * ************************************************************************************************* */
#pragma begin_pipe(V)
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMaxImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    ClampCompute<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount, CLAMPMODE::CLAMP_MAX);
}

/* **************************************************************************************************
 * ClampMin                                           *
 * ************************************************************************************************* */

template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    ClampCompute<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount, CLAMPMODE::CLAMP_MIN);
}
} // namespace AscendC
#endif // IMPL_MATH_CLAMP_CLAMP_COMMON_IMPL_H