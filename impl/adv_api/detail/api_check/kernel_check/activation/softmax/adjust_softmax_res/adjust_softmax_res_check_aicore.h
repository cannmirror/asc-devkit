/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adjust_softmax_res_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_AICORE_H_

#include "include/adv_api/activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
class CheckFuncClassAdjustSoftMaxRes {
public:
    __aicore__ inline CheckFuncClassAdjustSoftMaxRes() {};
    __aicore__ inline CheckFuncClassAdjustSoftMaxRes(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
        const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_AICORE_H_
