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
 * \file confusion_transpose_check_310.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_CHECK_310_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_CHECK_310_H

#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T>
class CheckFuncClassConfusionTranspose {
public:
    __aicore__ inline CheckFuncClassConfusionTranspose(){};
    __aicore__ inline CheckFuncClassConfusionTranspose(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, TransposeType transposeType, ConfusionTransposeTiling& tiling){};
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_CHECK_310_H
