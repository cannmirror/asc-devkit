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
 * \file xor_common_impl.h
 * \brief
 */
#ifndef LIB_XOR_XOR_COMMON_IMPL_H
#define LIB_XOR_XOR_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220 || __CCE_AICORE__ == 200)
#include "xor_membase_impl.h"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 300)
#include "xor_v300_impl.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "xor_c310_impl.h"
#endif

namespace AscendC {
#if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)
template <typename T>
__aicore__ inline void IsXorParamValid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    uint32_t tmpBufferSize = bufferSize / sizeof(T);
    uint32_t stackSize = tmpBufferSize / ONE_BLK_SIZE * ONE_BLK_SIZE; // integer multiple of 32 bytes

    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);
    CheckTmpBufferSize(stackSize, 0, bufferSize);
}
#endif
}  // namespace AscendC

#endif  // LIB_XOR_XOR_COMMON_IMPL_H
