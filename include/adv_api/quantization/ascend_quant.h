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
 * \file ascend_quant.h
 * \brief AscendQuant is an Quantization function which refers to
 *        techniques for performing computations and storing tensors at lower bitwidths than floating point precision.
 * Mathematical formulas: AscendQuant(x, scale, offset) = scale * x + offset
 * For per tensor interface:
 * quantization parameters(scale and offset) are scalars which are used in whole input(x).
 * For per channel interface:
 * input x shape: [m, n]
 * quantization parameter scale or offset shape: [n]
 * quantization parameter contains tensor which is recycled every n elements.
 */
#ifndef LIB_ASCEND_QUANT_ASCEND_QUANT_H
#define LIB_ASCEND_QUANT_ASCEND_QUANT_H
#include "include/adv_api/quantization/ascend_quant_utils.h"
#if __CCE_AICORE__ == 100 || __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200 || \
    defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "../../../impl/adv_api/detail/quantization/quant/ascend_quant_common_impl.h"
namespace AscendC {
#pragma begin_pipe(V)

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource>(dstTensor, srcTensor, scale, offset, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
{
    AscendQuant<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const float scale, const float offset)
{
    AscendQuant<T, isReuseSource>(dstTensor, srcTensor, scale, offset, srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource>(dstTensor, srcTensor, scaleTensor, offset, scaleCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset)
{
    AscendQuant<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset)
{
    AscendQuant<T, isReuseSource>(dstTensor, srcTensor, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] offsetCount: amount of per channel offsets to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] offsetCount: amount of per channel offsets to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount,
    const uint32_t offsetCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource>(dstTensor, srcTensor, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor)
{
    AscendQuant<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
{
    AscendQuant<T, isReuseSource>(dstTensor, srcTensor, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, scale, offset, calCount);
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const float scale, const float offset)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, scale, offset, srcTensor.GetSize());
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset,
        srcTensor.GetSize());
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, calCount);
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor,
    const LocalTensor<srcT>& offsetTensor)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor,
    const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleTensor.GetSize(),
        offsetTensor.GetSize(), srcTensor.GetSize());
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount,
    const uint32_t offsetCount, const uint32_t calCount)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleCount, offsetCount,
        calCount);
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const srcT offset,
    const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, scaleCount,
        calCount);
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const srcT offset)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, scaleTensor, offset, scaleCount, calCount);
}
template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const srcT offset)
{
    AscendQuantImpl<dstT, srcT, isReuseSource>(dstTensor, srcTensor, scaleTensor, offset, scaleTensor.GetSize(),
        srcTensor.GetSize());
}
template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                   const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                   const AscendQuantParam& para)
{
    AscendQuantImpl<dstT, srcT, scaleT, isReuseSource, config, policy>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                   const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                   const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    AscendQuantImpl<dstT, srcT, scaleT, isReuseSource, config, policy>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                   const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
                                   const AscendQuantParam& para)
{
    AscendQuantImpl<dstT, srcT, scaleT, isReuseSource, config, policy>(dstTensor, srcTensor, scaleTensor, offset, para);
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                   const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                   const scaleT offset, const AscendQuantParam& para)
{
    AscendQuantImpl<dstT, srcT, scaleT, isReuseSource, config, policy>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
}
#else
/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, scale, offset, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const float scale, const float offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, scale, offset, srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offset, scaleCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] offsetCount: amount of per channel offsets to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] offsetCount: amount of per channel offsets to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount,
    const uint32_t offsetCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}
#endif
#pragma end_pipe
}  // namespace AscendC
#endif
#endif  // LIB_ASCEND_QUANT_ASCEND_QUANT_H
