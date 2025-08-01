/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/* !
 * \file welfordfinalize.h
 * \brief
 */
#ifndef AICORE_ADV_API_NORMALIZATION_WELFORDFINALIZE_H
#define AICORE_ADV_API_NORMALIZATION_WELFORDFINALIZE_H
#include "normalization/welfordfinalize_utils.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
#include "detail/normalization/welfordfinalize/welfordfinalize_common_impl.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "detail/normalization/welfordfinalize/welfordfinalize_c310_impl.h"
#endif
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
namespace AscendC {
#pragma begin_pipe(V)
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance,
    const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
}
/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] counts, input LocalTensor, shape is [abLength]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
    const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource>(
        outputMean, outputVariance, inputMean, inputVariance, counts, sharedTmpBuffer, para);
}

/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, para);
}

/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] counts, input LocalTensor, shape is [abLength]
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
    WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, counts, para);
}
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance,
    const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource, config>(
        outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
}
/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] counts, input LocalTensor, shape is [abLength]
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
    const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource, config>(
        outputMean, outputVariance, inputMean, inputVariance, counts, sharedTmpBuffer, para);
}

/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource, config>(outputMean, outputVariance, inputMean, inputVariance, para);
}

/*!
 * \brief Calculate the final mean and variance using the Welford algorithm.
 *
 * \note support data type: float
 *
 * \param [out] outputMean, output LocalTensor, shape is [8]
 * \param [out] outputVariance, output LocalTensor, shape is [8]
 * \param [in] inputMean, input LocalTensor, shape is [abLength]
 * \param [in] inputVariance, input LocalTensor, shape is [abLength]
 * \param [in] counts, input LocalTensor, shape is [abLength]
 * \param [in] para, para detailed information about the original data shape
 */
template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
    WelfordFinalizePara& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    WelfordFinalizeImpl<isReuseSource, config>(outputMean, outputVariance, inputMean, inputVariance, counts, para);
}
#endif
#pragma end_pipe
} // namespace AscendC
#endif
#endif // AICORE_ADV_API_NORMALIZATION_WELFORDFINALIZE_H
