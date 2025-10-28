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
 * \file sign.h
 * \brief
 */
#ifndef LIB_MATH_SIGN_H
#define LIB_MATH_SIGN_H
#include <type_traits>
#include "kernel_log.h"
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
#include "../../../impl/adv_api/detail/math/sign/sign_common_impl.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102)
#include "../../../impl/adv_api/detail/math/sign/sign_c310_impl.h"
#endif


#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102)
#pragma begin_pipe(V)
namespace AscendC {
/*!
 * \ingroup Sign
 * \brief compute the Sign operation by element. (e.g. sign(0.1) = 1, sign(-0.1) is -1), sign(0) is 0)
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer: input local temporary Tensor
 * \param [in] calCount: the number of elements to be processed.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sign(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    SignCompute<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/*!
 * \ingroup Sign
 * \brief compute the Sign operation by element. (e.g. sign(0.1) = 1, sign(-0.1) is -1), sign(0) is 0)
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer: input local temporary Tensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sign(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Sign<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

/*!
 * \ingroup Sign
 * \brief compute the Sign operation by element. (e.g. sign(0.1) = 1, sign(-0.1) is -1), sign(0) is 0)
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] calCount: the number of elements to be processed.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sign(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
    SignCompute<T, isReuseSource>(dstTensor, srcTensor, calCount);
#else
    // Using the Stack Space to Allocate sharedTmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    Sign<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
#endif
}

/*!
 * \ingroup Sign
 * \brief compute the Sign operation by element. (e.g. sign(0.1) = 1, sign(-0.1) is -1), sign(0) is 0)
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sign(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor)
{
    Sign<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}
}  // namespace AscendC
#pragma end_pipe
#endif

#endif  // LIB_MATH_SIGN_H
