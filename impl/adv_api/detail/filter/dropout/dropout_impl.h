/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file dropout_impl.h
 * \brief
 */
#ifndef IMPL_FILTER_DROPOUT_DROPOUT_IMPL_H
#define IMPL_FILTER_DROPOUT_DROPOUT_IMPL_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002)
#include "dropout_m200_impl.h"
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "dropout_c220_impl.h"
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3002
#include "dropout_m300_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
#pragma begin_pipe(V)
template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOutOpt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const float keepProb,
    const DropOutShapeInfo& info)
{
    float divValue = 1.0;
    divValue = divValue / keepProb;

    const uint32_t dataSize = info.firstAxis * info.srcLastAxis;
    T actualVal;
    actualVal = static_cast<T>(divValue);
    if constexpr (dropOutMode == DROPOUT_MODE_BYTE_MISALIGN) {
        DropOutByteMode(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, actualVal, info);
    } else if constexpr (dropOutMode == DROPOUT_MODE_BYTE_ALIGN) {
        DropOutByteMode(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, actualVal, dataSize);
    } else if constexpr (dropOutMode == DROPOUT_MODE_BIT_ALIGN) {
        DropOutBitMode<T, isInitBitMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, actualVal,
            dataSize);
    } else if constexpr (dropOutMode == DROPOUT_MODE_BIT_MISALIGN) {
        DropOutBitMode<T, isInitBitMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, actualVal,
            info);
    }
}

template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOutImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const float keepProb,
    const DropOutShapeInfo& info)
{
    CHECK_FUNC_HIGHLEVEL_API(DropOut, (T, isInitBitMode, dropOutMode),
        (dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb, info));
    TRACE_START(TraceId::DropOut);

    if constexpr (dropOutMode != 0) {
        DropOutOpt<T, isInitBitMode, dropOutMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb, info);
    } else if (info.srcLastAxis < info.maskLastAxis) {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BYTE_MISALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer,
            keepProb, info);
    } else if (info.srcLastAxis == info.maskLastAxis) {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BYTE_ALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb,
            info);
    } else if (info.srcLastAxis == (info.maskLastAxis * ONE_BYTE_BIT_SIZE)) {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BIT_ALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb,
            info);
    } else {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BIT_MISALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer,
            keepProb, info);
    }
    TRACE_STOP(TraceId::DropOut);
}

template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOutImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const float keepProb, const DropOutShapeInfo& info)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    DropOutImpl<T, isInitBitMode, dropOutMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb, info);
}
#pragma end_pipe
} // namespace AscendC
#endif // IMPL_FILTER_DROPOUT_DROPOUT_IMPL_H
