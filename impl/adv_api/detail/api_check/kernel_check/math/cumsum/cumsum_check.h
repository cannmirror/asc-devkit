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
 * \file cumsum_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_CUMSUM_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_CUMSUM_CHECK_H_

#include "include/adv_api/math/cumsum_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "cumsum_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "cumsum_check_c310.h"
#else
#include "cumsum_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, const CumSumConfig &config>
__aicore__ inline void CheckFuncCumSum(__gm__ const char *apiName, const LocalTensor<T> &dstTensor, const LocalTensor<T> &lastRowTensor,
    const LocalTensor<T> &srcTensor, LocalTensor<uint8_t> &sharedTmpBuffer, const CumSumInfo &cumSumInfo)
{
    CheckFuncClassCumSum<T, config> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_CUMSUM_CHECK_H_
