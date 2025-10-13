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
 * \file geglu.h
 * \brief
 */
#ifndef LIB_ACTIVATION_GEGLU_H
#define LIB_ACTIVATION_GEGLU_H

#include "kernel_tensor.h"
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "../../../impl/adv_api/detail/activation/geglu/geglu_c310_impl.h"
#else
#include "../../../impl/adv_api/detail/activation/geglu/geglu_common_impl.h"
#endif
namespace AscendC {
#pragma begin_pipe(V)
/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
// GeGLU(x1,x2) = x1*GeLU(x2), x1 is src0, x2 is src1
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, const LocalTensor<uint8_t> &sharedTmpBuffer, uint32_t calCount)
{
    GeGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
}

/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, uint32_t calCount)
{
    // Only for AI Vector Core.
    if (g_coreType == AIC) {
        return;
    }

    GeGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, calCount);
}

/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    GeGLU<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, srcTensor0.GetSize());
}
/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1)
{
    GeGLU<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, srcTensor0.GetSize());
}
#pragma end_pipe
}
#endif
