/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layernormgradbeta.h
 * \brief
 */
#ifndef AICORE_ADV_API_NORMALIZATION_LAYERNORMGRADBETA_H
#define AICORE_ADV_API_NORMALIZATION_LAYERNORMGRADBETA_H

#include "kernel_tensor.h"
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "detail/normalization/layernormgrad/layernormgradbeta_c310_impl.h"
#else
#include "detail/normalization/layernormgrad/layernormgradbeta_common_impl.h"
#endif
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \brief layernormgradbeta intf function:
 * reduce_axis = (0, 1)
 * pd_gamma = sum(inputDy*res_for_gamma, reduce_axis, keepdims=True)
 * pd_beta  = sum(inputDy, reduce_axis, keepdims=True)
 *
 * \note support data type: half and float
 *
 * \param [out] outputPdGamma, output LocalTensor, shape is [H]
 * \param [out] outputPdBeta, output LocalTensor, shape is [H]
 * \param [in] resForGamma, input LocalTensor, shape is [B, S, H]
 * \param [in] inputDy, input LocalTensor, shape is [B, S, H]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] tiling, layernormgradbeta tiling
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGradBeta(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta,
    const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LayerNormGradBetaTiling& tiling)
{
    LayerNormGradBetaImpl<T, isReuseSource>(outputPdGamma, outputPdBeta, resForGamma, inputDy, sharedTmpBuffer, tiling);
}

/*!
 * \brief layernormgradbeta intf function
 *
 * \note support data type: half and float
 *
 * \param [out] outputPdGamma, output LocalTensor, shape is [H]
 * \param [out] outputPdBeta, output LocalTensor, shape is [H]
 * \param [in] resForGamma, input LocalTensor, shape is [B, S, H]
 * \param [in] inputDy, input LocalTensor, shape is [B, S, H]
 * \param [in] tiling, layernormgradbeta tiling
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGradBeta(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta,
    const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy, LayerNormGradBetaTiling& tiling)
{
    LayerNormGradBetaImpl<T, isReuseSource>(outputPdGamma, outputPdBeta, resForGamma, inputDy, tiling);
}
#pragma end_pipe
} // namespace AscendC
#endif // AICORE_ADV_API_NORMALIZATION_LAYERNORMGRADBETA_H
