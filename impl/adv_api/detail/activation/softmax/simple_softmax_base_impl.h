/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file simple_softmax_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102)
#include "regbase/c310/simple_softmax_impl.h"
#elif __CCE_AICORE__ == 300
#include "regbase/v300/simple_softmax_impl.h"
#include "softmax_common/softmax_common_simple.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/simple_softmax_impl.h"
#include "softmax_common/softmax_common_simple.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/simple_softmax_impl.h"
#include "softmax_common/softmax_common_simple.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SimpleSoftMax, (T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config),
        (dst, inSumTensor, inMaxTensor, src, workLocal, tiling, softmaxShapeInfo));
    SimpleSoftMaxBaseImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dst, inSumTensor,
        inMaxTensor, src, workLocal, tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SimpleSoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dst, inSumTensor, inMaxTensor, src, workLocal,
        tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102)
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
#endif
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SimpleSoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dst, inSumTensor, inMaxTensor, src, workLocal,
        tiling, softmaxShapeInfo);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H