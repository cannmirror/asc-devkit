/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_QUANTIZATION_DEQUANT_DEQUANT_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_QUANTIZATION_DEQUANT_DEQUANT_CHECK_H

#include "quantization/ascend_dequant_utils.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "dequant_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "dequant_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename dstT, typename scaleT, bool isPureDqParams, DeQuantMode mode>
__aicore__ inline void CheckFuncAscendDequant(__gm__ const char* apiName, const LocalTensor<dstT>& dstTensor,
    const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale,
    const LocalTensor<uint8_t>& sharedTmpBuffer, DequantParams& params, uint32_t calCount)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassAscendDequant<dstT, scaleT, isPureDqParams, mode> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, deqScale, sharedTmpBuffer, params, calCount);
#endif
}

template <typename dstT, typename scaleT, bool isPureDqParams, DeQuantMode mode>
__aicore__ inline void CheckFuncAscendDequant(__gm__ const char* apiName, const LocalTensor<dstT>& dstTensor,
    const LocalTensor<int32_t>& srcTensor, const scaleT deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer,
    DequantParams& params)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassAscendDequantScalar<dstT, scaleT, isPureDqParams, mode> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, deqScale, sharedTmpBuffer, params);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_QUANTIZATION_DEQUANT_DEQUANT_CHECK_H
