/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file pad_common_impl.h
 * \brief
 */
#ifndef IMPL_PAD_PAD_PAD_COMMON_IMPL_H
#define IMPL_PAD_PAD_PAD_COMMON_IMPL_H

#include "../../api_check/kernel_api_check.h"
#if (__CCE_AICORE__ <= 200) && (__NPU_ARCH__ != 5102)
#include "pad_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "pad_v220_impl.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "pad_c310_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void PadImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, PadParams &padParams,
    const LocalTensor<uint8_t> &sharedTmpBuffer, PadTiling &tiling)
{
    CHECK_FUNC_HIGHLEVEL_API(Pad, (T), (dstTensor, srcTensor, padParams, sharedTmpBuffer, tiling));
    PadCompute<T>(dstTensor, srcTensor, padParams, sharedTmpBuffer, tiling);
}

template <typename T>
__aicore__ inline void UnPadImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    UnPadParams &unPadParams, LocalTensor<uint8_t> &sharedTmpBuffer, UnPadTiling &tiling)
{
    CHECK_FUNC_HIGHLEVEL_API(UnPad, (T), (dstTensor, srcTensor, unPadParams, sharedTmpBuffer, tiling));

    UnPadCompute<T>(dstTensor, srcTensor, unPadParams, sharedTmpBuffer, tiling);
}
} // namespace AscendC
#endif // IMPL_PAD_PAD_PAD_COMMON_IMPL_H
