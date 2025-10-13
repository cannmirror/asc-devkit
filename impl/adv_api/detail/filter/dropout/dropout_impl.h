/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

#if (__CCE_AICORE__ <= 200) && (__NPU_ARCH__ != 5102)
#include "dropout_m200_impl.h"
#elif __CCE_AICORE__ == 220
#include "dropout_c220_impl.h"
#elif __CCE_AICORE__ == 300
#include "dropout_m300_impl.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "dropout_c310_impl.h"
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
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        actualVal = ToBfloat16(divValue);
    } else {
        actualVal = static_cast<T>(divValue);
    }
#else
    actualVal = static_cast<T>(divValue);
#endif
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
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
    CheckTensorPos<T>(dstLocal, Hardware::UB, "dstLocal", "VECIN / VECCALC / VECOUT", "DropOut");
    CheckTensorPos<T>(srcLocal, Hardware::UB, "srcLocal", "VECIN / VECCALC / VECOUT", "DropOut");
    CheckTensorPos<uint8_t>(maskLocal, Hardware::UB, "maskLocal", "VECIN / VECCALC / VECOUT", "DropOut");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "DropOut");
#endif
    TRACE_START(TraceId::DropOut);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
    static_assert((dropOutMode == 0 || dropOutMode == 1 || dropOutMode == 2 || dropOutMode == 3 || dropOutMode == 4),
        "dropOutMode should be 0 / 1 / 2 / 3 / 4");
    ASCENDC_ASSERT((info.firstAxis > 0), { KERNEL_LOG(KERNEL_ERROR, "info.firstAxis must > 0!"); });
    constexpr float keepProbMin = 0;
    constexpr float keepProbMax = 1;
    ASCENDC_ASSERT(((keepProb > keepProbMin) && (keepProb < keepProbMax)),
        { KERNEL_LOG(KERNEL_ERROR,"keepProb should be larger than 0 and smaller than 1, current keepProb is %f!",
            keepProb); });
    if constexpr(dropOutMode == 1 || dropOutMode == 4) {
        ASCENDC_ASSERT((info.maskLastAxis % 2 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "maskLastAxis should be multiples of 2 when dropOutMode is 1 or 4!"); });
    }
    static_assert(SupportType<T, half, float, bfloat16_t>(),
        "Dropout Only Supportes half, float, bfloat16_t on current device.");
#endif

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
