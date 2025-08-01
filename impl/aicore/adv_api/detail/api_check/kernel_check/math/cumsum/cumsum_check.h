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
 * \file cumsum_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_CUMSUM_CUMSUM_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_CUMSUM_CUMSUM_CHECK_H

#include "math/cumsum_utils.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "cumsum_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "cumsum_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, const CumSumConfig& config>
__aicore__ inline void CheckFuncCumSum(__gm__ const char* apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& lastRowTensor, const LocalTensor<T>& srcTensor, LocalTensor<uint8_t>& sharedTmpBuffer,
    const CumSumInfo& cumSumInfo)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassCumSum<T, config> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_MATH_CUMSUM_CUMSUM_CHECK_H
