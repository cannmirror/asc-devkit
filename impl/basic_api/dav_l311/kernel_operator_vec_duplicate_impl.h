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
 * \file kernel_operator_vec_duplicate_impl.h
 * \brief AscendC l311 support vector duplicate api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#include <type_traits>
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
        std::is_same<T, half>::value || std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value,
        "Duplicate instr only support uint8_t/int8_t/half/int16_t/uint16_t/int32_t/uint32_t/float type");
}

template <typename T> constexpr __aicore__ inline void CheckDuplicateL0SupportedType()
{
    static_assert(SupportType<T, half, int16_t, uint16_t,
        int32_t, uint32_t, float>(),
        "Duplicate instr only support half/int16_t/uint16_t/int32_t/"
        "uint32_t/float type on current device");
}

namespace Internal {
template <bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T>
__aicore__ inline void VecDupLevel0VFImpl(__ubuf__ T *dst, const T& scalarValue, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride,
    __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = 0;
    newRepeatTimes = VecMicroGetRepeatTimes<T, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T> dstVreg;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);

    MicroAPI::Duplicate(dstVreg, scalarValue, maskFull);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
        }
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + index * dstRepeatStride * ElePerBlkT, dstVreg, dstBlockStride, maskReg);
    }
}

template <bool isSetMask, bool isMaskBitMode, typename T>
__aicore__ inline void VecDupLevel0Template(__ubuf__ T *dst, const T& scalarValue, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
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
        VF_CALL<VecDupLevel0VFImpl<isSetMask, isMaskBitMode, false, T>>(dst, scalarValue, maskArray, maskCount,
            repeatTime, dstBlockStride, dstRepeatStride, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    } else {
        if constexpr (isMaskBitMode) {
            if constexpr (SupportBytes<T, 1>()) {
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
                SetVectorMask<T>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
            }
        }
        // when isSetMask is false, normal mode, maskBuf = nullptr, not support B8
        VF_CALL<VecDupLevel0VFImpl<isSetMask, isMaskBitMode, true, T>>(dst, scalarValue, maskArray, maskCount,
            repeatTime, dstBlockStride, dstRepeatStride, maskBuf);
        if constexpr (isMaskBitMode && SupportBytes<T, 1>()) {
            AscendC::AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    }
}
}

// level 2
template <typename T>
typename std::enable_if_t<
!std::is_same<T, uint8_t>::value &&
!std::is_same<T, int8_t>::value &&
!std::is_same<T, uint16_t>::value &&
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, uint32_t>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value ||
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value ||
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, const int32_t& count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)count;
        MaskReg preg;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = CreatePredicate<T>(sreg);
            Duplicate(vDst, scalarValue, preg);
            DataCopy(dst, vDst, i * sregLower, preg);
        }
    }
}

// level 0, continuous mode
template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dst, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    CheckDuplicateL0SupportedType<T>();
    Internal::VecDupLevel0Template<isSetMask, false>(dst, scalarValue, nullptr, mask,
                                    repeatTime, dstBlockStride, dstRepeatStride);
}

// level 0, mask bit mode
template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dst, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    CheckDuplicateL0SupportedType<T>();
    Internal::VecDupLevel0Template<isSetMask, true>(dst, scalarValue, mask, 0,
                                        repeatTime, dstBlockStride, dstRepeatStride);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
