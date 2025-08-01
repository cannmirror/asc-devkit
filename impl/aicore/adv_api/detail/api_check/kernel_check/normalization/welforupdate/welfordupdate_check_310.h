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
 * \file welfordupdate_check_310.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORUPDATE_WELFORDUPDATE_CHECK_310_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORUPDATE_WELFORDUPDATE_CHECK_310_H

#include "normalization/layernorm_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, typename U, bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
class CheckFuncClassWelfordUpdate {
public:
    __aicore__ inline CheckFuncClassWelfordUpdate(){};
    __aicore__ inline CheckFuncClassWelfordUpdate(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance,
        const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const WelfordUpdateParam& para){};
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORUPDATE_WELFORDUPDATE_CHECK_310_H
