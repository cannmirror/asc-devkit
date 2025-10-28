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
 * \file adjust_softmax_res_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_H_

#include "include/adv_api/activation/softmax_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "adjust_softmax_res_check_common.h"
#else
#include "adjust_softmax_res_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline void CheckFuncAdjustSoftMaxRes(__gm__ const char *apiName, const LocalTensor<T1>& softMaxRes,
    const LocalTensor<T2>& maxTensor, const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CheckFuncClassAdjustSoftMaxRes<T1, T2, isDataFormatNZ, stepSizeMode> checkFun(apiName);
    checkFun.VerifyingParameters(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_ADJUST_SOFTMAX_RES_ADJUST_SOFTMAX_RES_CHECK_H_
