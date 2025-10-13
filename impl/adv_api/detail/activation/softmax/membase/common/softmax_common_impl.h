/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_MEMBASE_COMMON_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_MEMBASE_COMMON_SOFTMAX_COMMON_IMPL_H

namespace AscendC {
template <typename T1, typename T2, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResNZImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    uint32_t floatStepSize = ONE_BLK_FLOAT_NUM;
    uint32_t halfStepSize = ONE_BLK_HALF_NUM;

    bool isUpdateNeedCheck = false;
    const uint32_t splitNZBlockCount = softmaxShapeInfo.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    SetVectorMask<float>(SOFTMAX_SHAPE_NZ_BASIC_COUNT);
    for (uint32_t j = 0; j < softmaxShapeInfo.srcM; j++) {
        uint32_t offset = j * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        uint32_t splitCount = softmaxShapeInfo.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if constexpr (sizeof(T2) == sizeof(float)) {
            T2 maxValue = maxTensor.GetValue(j * floatStepSize);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == from) {
                for (uint32_t k = 0; k < splitNZBlockCount; k++) {
                    Duplicate<T1, false>(softMaxRes[offset + splitCount * k], to, MASK_PLACEHOLDER, 1, 1,
                        DEFAULT_REPEAT_STRIDE);
                }
                isUpdateNeedCheck = true;
            }
        } else {
            T2 maxValue = maxTensor.GetValue(j * halfStepSize);
            uint16_t checkValue = *reinterpret_cast<uint16_t*>(&maxValue);
            if (checkValue == (uint16_t)from) {
                for (uint32_t k = 0; k < splitNZBlockCount; k++) {
                    Duplicate<T1, false>(softMaxRes[offset + splitCount * k], to, MASK_PLACEHOLDER, 1, 1,
                        DEFAULT_REPEAT_STRIDE);
                }
                isUpdateNeedCheck = true;
            }
        }
    }
    ResetMask();
    return isUpdateNeedCheck;
}

template <typename T1, typename T2, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResNDImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    uint32_t floatStepSize = ONE_BLK_FLOAT_NUM;
    uint32_t halfStepSize = ONE_BLK_HALF_NUM;
    if constexpr (stepSizeMode) {
        floatStepSize = 1;
        halfStepSize = 1;
    }

    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t j = 0; j < softmaxShapeInfo.srcM; j++) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            T2 maxValue = maxTensor.GetValue(j * floatStepSize);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == from) {
                Duplicate<T1, false>(softMaxRes[j * softmaxShapeInfo.srcK], to, MASK_PLACEHOLDER, 1, 1,
                    DEFAULT_REPEAT_STRIDE);
                isUpdateNeedCheck = true;
            }
        } else {
            T2 maxValue = maxTensor.GetValue(j * halfStepSize);
            uint16_t checkValue = *reinterpret_cast<uint16_t*>(&maxValue);
            if (checkValue == (uint16_t)from) {
                Duplicate<T1, false>(softMaxRes[j * softmaxShapeInfo.srcK], to, MASK_PLACEHOLDER, 1, 1,
                    DEFAULT_REPEAT_STRIDE);
                isUpdateNeedCheck = true;
            }
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResBaseImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
                                                const uint32_t from, const T1 to,
                                                const SoftMaxShapeInfo& softmaxShapeInfo)
{
    SetMaskNorm();
    ResetMask();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    if constexpr (isDataFormatNZ) {
        isUpdateNeedCheck =
            AdjustSoftMaxResNZImpl<T1, T2, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    } else {
        isUpdateNeedCheck =
            AdjustSoftMaxResNDImpl<T1, T2, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    }
    return isUpdateNeedCheck;
}
}

#endif // IMPL_ACTIVATION_SOFTMAX_MEMBASE_COMMON_SOFTMAX_COMMON_IMPL_H