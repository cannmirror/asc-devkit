/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file floor.h
 * \brief
 */

#ifndef LIB_MATH_FLOOR_H
#define LIB_MATH_FLOOR_H

#include "kernel_tensor.h"
#include "../../../impl/adv_api/detail/math/floor/floor_common_impl.h"

#if (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220 || __CCE_AICORE__ == 300)) || \
    defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102)

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \brief the floor function is the function that takes as input a real number x,
 * and gives as output the greatest integer less than or equal to x. (e.g. floor(2.4) is 2, floor(-2.4) is -3).
 * For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.floor.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Floor(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
#if __CCE_AICORE__ == 220
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
#endif
    static_assert((std::is_same<T, float>::value || std::is_same<T, half>::value),
        "Failed to check the data types, current api support data types are half/float.");
    FloorImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/* !
 * \ingroup Floor
 * \note support data type: half and float
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Floor(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t calCount)
{
#if __CCE_AICORE__ == 220
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
#endif
    static_assert((std::is_same<T, float>::value || std::is_same<T, half>::value),
        "Failed to check the data types, current api support data types are half/float.");
    FloorImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_FLOOR_H
