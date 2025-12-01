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
 * \file kernel_operator_vec_template_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TEMPLATE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TEMPLATE_IMPL_H

#include "kernel_utils.h"
#include "kernel_struct_binary.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
namespace Internal {
__aicore__ inline bool IsCounterMode()
{
    // CTRL[56] as 1,for counter mask
    constexpr uint32_t CTRL_COUNTER = 56;
    return ((get_ctrl() >> CTRL_COUNTER) & 0x1) == 0x1;
}

template <bool isSetMask = true, bool isNormalMode = true, bool isMaskBitMode = true>
__aicore__ inline uint32_t VecMicroGetCount(const uint64_t maskArray[], const uint64_t maskCount,
    __ubuf__ uint64_t *maskBuf)
{
    if constexpr (isNormalMode && !isMaskBitMode && !isSetMask) { // no count, return 0
        return 0;
    }
    if constexpr (isNormalMode && isMaskBitMode) { // no count, return 0
        return 0;
    }
    if constexpr (isNormalMode && !isMaskBitMode) { // count from maskCount
        return static_cast<uint32_t>(maskCount);
    }
    // Counter mode
    uint32_t count = 0;
    if constexpr (!isSetMask) {
        // get SPR.MASK in VF
        MicroAPI::MaskReg sprLoadMaskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(maskBuf, sprLoadMaskReg);
        // insert membar(vec store operation) before load maskBuf[0](scalar load operation)
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        count = static_cast<uint32_t>(maskBuf[0]);
    } else if constexpr (isMaskBitMode) {
        count = static_cast<uint32_t>(maskArray[0]);
    } else {
        count = static_cast<uint32_t>(maskCount);
    }
    return count;
}

template <typename T, bool isNormalMode = true>
__aicore__ inline uint16_t VecMicroGetRepeatTimes(uint32_t count, const uint8_t repeatTime)
{
    if constexpr (isNormalMode) {
        return repeatTime;
    }
    return CeilDivision(count, GetVecLen() / sizeof(T));
}

template <typename T, bool isSetMask = true, bool isNormalMode = true, bool isMaskBitMode = true>
__aicore__ inline MicroAPI::MaskReg VecMicroGetMaskReg(__ubuf__ uint64_t *maskBuf, uint32_t &count)
{
    MicroAPI::MaskReg maskReg;
    if constexpr (isNormalMode && !isMaskBitMode && !isSetMask) {
        if constexpr (SupportBytes<T, 2, 4>()) {
            maskReg = MicroAPI::MoveMask<T>();
        } else {
            ASCENDC_ASSERT(false, "unsupported dtype on current device!");
        }
    } else if constexpr (isNormalMode && isMaskBitMode) {
        if constexpr (SupportBytes<T, 1>()) {
            MicroAPI::DataCopy(maskReg, (__ubuf__ uint32_t *)maskBuf);
        } else {
            maskReg = MicroAPI::MoveMask<T>();
        }
    } else {
        maskReg = MicroAPI::UpdateMask<T>(count);
    }
    return maskReg;
}

enum class BinaryFuncMode {
    NORMAL,        // Add, Sub, Mul, Div, Max, Min, And, Or etc..
    DST_SRC_INPUT, // FusedMulAdd, FusedMulAddRelu, MulAddDst
};
/*
 * T: data type
 * func: MicroAPI input/output function
 * isSetMask: basic api whether to set mask
 * isNormalMode: true: NormalMode, false: CounterMode
 * isMaskBitMode: true: mask bit mode, false: mask count mode
 */
template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode,
    BinaryFuncMode funcMode = BinaryFuncMode::NORMAL, typename T, typename U>
__aicore__ inline void VecBinaryVFImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, const BinaryRepeatParams repeatParams,
    __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = 0;
    constexpr bool TUCompare = sizeof(T) > sizeof(U);
    using TT = typename Conditional<TUCompare, T, U>::type;
    newRepeatTimes = VecMicroGetRepeatTimes<TT, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg maskRegDst;
    MicroAPI::MaskReg maskRegSrc;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<TT, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
        maskRegSrc = maskReg;
        maskRegDst = maskReg;
        if constexpr (sizeof(U) == 2 * sizeof(T)) {
            MicroAPI::MaskPack(maskRegDst, maskReg);
        } else if constexpr (sizeof(T) == 2 * sizeof(U)) {
            MicroAPI::MaskPack(maskRegSrc, maskReg);
        }
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint8_t ElePerBlkU = GetDataBlockSizeInBytes() / sizeof(U);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<TT, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
            maskRegSrc = maskReg;
            maskRegDst = maskReg;
            if constexpr (sizeof(U) == 2 * sizeof(T)) {
                MicroAPI::MaskPack(maskRegDst, maskReg);
            } else if constexpr (sizeof(T) == 2 * sizeof(U)) {
                MicroAPI::MaskPack(maskRegSrc, maskReg);
            }
        }
        MicroAPI::RegTensor<T> dstVreg;
        MicroAPI::RegTensor<U> srcVreg0, srcVreg1;
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<U, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcVreg0,
            src0 + index * repeatParams.src0RepStride * ElePerBlkU, repeatParams.src0BlkStride, maskRegSrc);
        MicroAPI::DataCopy<U, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcVreg1,
            src1 + index * repeatParams.src1RepStride * ElePerBlkU, repeatParams.src1BlkStride, maskRegSrc);
        if constexpr (funcMode == BinaryFuncMode::DST_SRC_INPUT) {
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dstVreg,
                dst + index * repeatParams.dstRepStride * ElePerBlkT, repeatParams.dstBlkStride, maskRegDst);
        }
        func(dstVreg, srcVreg0, srcVreg1, maskReg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, dstVreg, repeatParams.dstBlkStride, maskRegDst);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, BinaryFuncMode funcMode = BinaryFuncMode::NORMAL, typename T,
    typename U>
__aicore__ inline void VecBinaryImplTemplate(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const BinaryRepeatParams &repeatParams)
{
    constexpr bool TUCompare = sizeof(T) > sizeof(U);
    using TT = typename Conditional<TUCompare, T, U>::type;
    if constexpr (isMaskBitMode) {
        ASCENDC_ASSERT(maskCount == 0, "maskCount must be 0 when isMaskBitMode is true.");
    } else {
        ASCENDC_ASSERT(maskArray == nullptr, "maskArray must be nullptr when isMaskBitMode is false.");
    }
    __ubuf__ uint64_t *maskBuf = nullptr;

    if (Internal::IsCounterMode()) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2); // maskReg 256bit PK-> 128bit
        }
        VF_CALL<VecBinaryVFImpl<func, isSetMask, isMaskBitMode, false, funcMode, T, U>>(dst, src0, src1, maskArray,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    } else {
        if constexpr (isMaskBitMode) {
            if constexpr (SupportBytes<TT, 1>()) {
                ASCENDC_ASSERT(isSetMask, "mask must be set when sizeof(T) is 1.");
                auto eventIDV2S = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(eventIDV2S);
                WaitFlag<HardEvent::V_S>(eventIDV2S);
                maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 4);
                maskBuf[0] = maskArray[0];
                maskBuf[1] = maskArray[1];
                maskBuf[2] = maskArray[2];
                maskBuf[3] = maskArray[3];
                auto eventIDS2V = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(eventIDS2V);
                WaitFlag<HardEvent::S_V>(eventIDS2V);
            } else if constexpr (isSetMask) {
                SetVectorMask<TT>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
            }
        }
        // when isSetMask is false, normal mode, maskBuf = nullptr, not support B8
        VF_CALL<VecBinaryVFImpl<func, isSetMask, isMaskBitMode, true, funcMode, T, U>>(dst, src0, src1, maskArray,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (isMaskBitMode && SupportBytes<TT, 1>()) {
            AscendC::AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    }
}
} // namespace Internal
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TEMPLATE_IMPL_H
