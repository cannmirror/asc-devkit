/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_template_impl.h"

namespace AscendC {
namespace Internal {
template <auto func, typename T, typename RegType>
__aicore__ inline void VecBinaryScalarLevel2VFImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint32_t calCount)
{
    RegType srcReg;
    RegType dstReg;
    uint32_t count = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg mask;
    constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() / sizeof(T) * RegType::trait.REG_NUM);
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, RegType::trait>(count);
        MicroAPI::LoadAlign(srcReg, src + i * repeatStride);
        func(dstReg, srcReg, scalarValue, mask);
        MicroAPI::StoreAlign(dst + i * repeatStride, dstReg, mask);
    }
}

template <auto func, typename T>
__aicore__ inline void VecBinaryScalarLevel2ImplTemplate(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint32_t calCount)
{
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        VF_CALL<VecBinaryScalarLevel2VFImpl<func, T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src,
            scalarValue, calCount);
    } else {
        VF_CALL<VecBinaryScalarLevel2VFImpl<func, T, MicroAPI::RegTensor<T>>>(dst, src, scalarValue, calCount);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T>
__aicore__ inline void VecBinaryScalarLevel0VFImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams, __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = 0;
    newRepeatTimes = VecMicroGetRepeatTimes<T, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
        }
        MicroAPI::RegTensor<T> dstVreg;
        MicroAPI::RegTensor<T> srcVreg;
#ifndef NO_OVERLAP_IN_MULTI_REPEAT
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
#endif
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcVreg,
            src + index * repeatParams.srcRepStride * ElePerBlkT, repeatParams.srcBlkStride, maskReg);
        func(dstVreg, srcVreg, scalarValue, maskReg);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, dstVreg, repeatParams.dstBlkStride, maskReg);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, typename T>
__aicore__ inline void VecBinaryScalarLevel0Template(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
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
        VF_CALL<VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, false, T>>(dst, src, scalarValue, maskArray,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        };
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
        VF_CALL<VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, true, T>>(dst, src, scalarValue, maskArray,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (isMaskBitMode && SupportBytes<T, 1>()) {
            AscendC::AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T, MicroAPI::LoadDist pattern, uint8_t scalarIdx>
__aicore__ inline void VecBinaryScalarLevel0VFImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams, __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = 0;
    newRepeatTimes = VecMicroGetRepeatTimes<T, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::RegTensor<T> vDstReg0;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
        }
#ifndef NO_OVERLAP_IN_MULTI_REPEAT
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
#endif
        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign<T, pattern>(vSrcReg0, src0);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vSrcReg1,
                src1 + index * repeatParams.srcRepStride * ElePerBlkT, repeatParams.srcBlkStride, maskReg);
        } else if constexpr (scalarIdx == 1) {
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vSrcReg0,
                src0 + index * repeatParams.srcRepStride * ElePerBlkT, repeatParams.srcBlkStride, maskReg);
            MicroAPI::LoadAlign<T, pattern>(vSrcReg1, src1);
        }
        func(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, vDstReg0, repeatParams.dstBlkStride, maskReg);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, typename T, MicroAPI::LoadDist pattern, uint8_t scalarIdx>
__aicore__ inline void VecBinaryScalarLevel0Template(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
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
        VF_CALL<VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, false, T, pattern, scalarIdx>>(dst, src0, src1, maskArray,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        };
    } else {
        if constexpr (isMaskBitMode && isSetMask) {
            SetVectorMask<T>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
        }
        // when isSetMask is false, normal mode, maskBuf = nullptr, not support B8
        VF_CALL<VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, true, T, pattern, scalarIdx>>(dst, src0, src1, maskArray,
            maskCount, repeatTime, repeatParams, maskBuf);
    }
}
} // namespace Internal
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
// Adds::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Adds not support current datatype!");
    constexpr auto func = MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Adds not support current datatype!");
    constexpr auto func = MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Adds::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t,
        complex32, complex64>()),
        "Adds not support current datatype!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        constexpr auto func =
            MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
// Muls::Level 0
namespace MicroAPIMuls {
template <typename T, typename RegT>
__aicore__ inline void Muls(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    if constexpr (SupportType<T, bfloat16_t>()) {
        MicroAPI::Duplicate(dstReg, scalarValue, mask);
        MicroAPI::Mul(dstReg, srcReg, dstReg, mask);
    } else {
        MicroAPI::Muls(dstReg, srcReg, scalarValue, mask);
    }
}
} // namespace MicroAPIMuls
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Muls not support current datatype!");
    constexpr auto func = MicroAPIMuls::Muls<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Muls not support current datatype!");
    constexpr auto func = MicroAPIMuls::Muls<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Muls::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src, const T scalarValue, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t, complex32, complex64>()),
        "Muls not support current datatype!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        constexpr auto func = MicroAPIMuls::Muls<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIMuls::Muls<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
// Maxs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    constexpr auto func = MicroAPI::Maxs<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    constexpr auto func = MicroAPI::Maxs<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Maxs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Maxs not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func =
            MicroAPI::Maxs<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPI::Maxs<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
// Mins::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Mins not support current datatype!");
    constexpr auto func = MicroAPI::Mins<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Mins not support current datatype!");
    constexpr auto func = MicroAPI::Mins<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Mins::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Mins not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func =
            MicroAPI::Mins<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPI::Mins<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
// ShiftLeft::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftLeft not support current datatype!");
    constexpr auto func = MicroAPI::ShiftLefts<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftLeft not support current datatype!");
    constexpr auto func = MicroAPI::ShiftLefts<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>()),
        "ShiftLeft not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPI::ShiftLefts<T, int16_t, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPI::ShiftLefts<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
// ShiftRight::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams, bool roundEn = false)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftRight not support current datatype!");
    constexpr auto func = MicroAPI::ShiftRights<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams, bool roundEn = false)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftRight not support current datatype!");
    constexpr auto func = MicroAPI::ShiftRights<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// ShiftRight::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T *dst, __ubuf__ T *src, const T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>()),
        "ShiftRight not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPI::ShiftRights<T, int16_t, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPI::ShiftRights<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
namespace MicroAPILeakyRelu {
template <typename T, typename RegT>
__aicore__ inline void LeakyRelu(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    vlrelu(dstReg, srcReg, scalarValue, mask, MODE_ZEROING);
}
}

// LeakyRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "LeakyRelu not support current datatype!");
    constexpr auto func = MicroAPILeakyRelu::LeakyRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "LeakyRelu not support current datatype!");
    constexpr auto func = MicroAPILeakyRelu::LeakyRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// LeakyRelu::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float>()), "LeakyRelu not support current datatype!");
    constexpr auto func = MicroAPILeakyRelu::LeakyRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
}

/* **************************************************************************************************
 * Subs: LocalTensor - Scalar                                             *
 * ************************************************************************************************* */
// Subs::Level 0
namespace MicroAPISubs {
template <typename T, typename RegT>
__aicore__ inline void Subs(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Sub(dstReg, srcReg, dstReg, mask);
}
} // namespace MicroAPISubs
template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Subs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t, complex32, complex64>()),
        "Subs not support current datatype!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Subs: Scalar - LocalTensor                                             *
 * ************************************************************************************************* */
// Subs::Level 0
namespace MicroAPISubs {
template <typename T, typename RegT>
__aicore__ inline void Subs2(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Sub(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPISubs

template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Subs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t, complex32, complex64>()),
        "Subs not support current datatype!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Divs                                             *
 * ************************************************************************************************* */
namespace MicroAPIDivs {
template <typename T, typename RegT>
__aicore__ inline void Divs(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Div(dstReg, srcReg, dstReg, mask);
}
} // namespace MicroAPIDivs
// Divs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Divs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t, complex32, complex64>()),
        "Divs not support current datatype!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Divs   Scalar / LocalTensor                                         *
 * ************************************************************************************************* */
namespace MicroAPIDivs {
template <typename T, typename RegT>
__aicore__ inline void Divs2(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Div(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPIDivs
// Div::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Divs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t, complex32, complex64>()),
        "Divs not support current datatype!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Ands                                             *
 * ************************************************************************************************* */
namespace MicroAPIAnds {
template <typename T, typename RegT>
__aicore__ inline void Ands(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::And(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPIAnds
// Ands::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Ands::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ands not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Ors                                             *
 * ************************************************************************************************* */
namespace MicroAPIOrs {
template <typename T, typename RegT>
__aicore__ inline void Ors(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Or(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPIOrs
// Ors::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Ors::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ors not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * FusedMulsCast                                            *
 * ************************************************************************************************* */
// FusedMulsCast::Level 2

template <typename T, typename U>
__aicore__ inline void FusedMulsCastImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, float>>(), "Failed to check dtype in "
        "FusedMulsCast, current api support dtype combination is src : float, dst: half, scalar: float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    __VEC_SCOPE__
    {
        U scalar = src1[0];
        MicroAPI::RegTensor<T> vDstReg;
        MicroAPI::RegTensor<U> vSrcReg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<U>(sreg);
            MicroAPI::LoadAlign(vSrcReg, src0 + i * sregLower);
            MicroAPI::FusedMulsCast(vDstReg, vSrcReg, scalar, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, vDstReg, mask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void FusedMulsCastImpl(__ubuf__ T *dst, __ubuf__ U *src, U scalar, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, float>>(), "Failed to check dtype in "
        "FusedMulsCast, current api support dtype combination is src : float, dst: half, scalar: float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vDstReg;
        MicroAPI::RegTensor<U> vSrcReg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<U>(sreg);
            MicroAPI::LoadAlign(vSrcReg, src + i * sregLower);
            MicroAPI::FusedMulsCast(vDstReg, vSrcReg, scalar, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, vDstReg, mask);
        }
    }
}

template <typename T, BinaryScalarOp op, uint8_t scalarIdx>
__aicore__ inline void BinaryScalarOpTemplateCntB64(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &calCount)
{
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> dstReg, srcReg0, srcReg1, dupReg;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> preReg;
    MicroAPI::MaskReg mask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::RegTensor<uint32_t> zeroReg;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    uint32_t repeatStride = (uint32_t)B64_DATA_NUM_PER_REPEAT * 2;
    uint16_t repeatTime = CeilDivision(calCount, repeatStride);
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0, maskFull);

    if constexpr (scalarIdx == 0) {
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ T *)src0);
        MicroAPI::LoadUnAlign(preReg, ureg, (__ubuf__ T *)src0);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1], (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(srcReg0, dupReg, maskFull);
    } else if (scalarIdx == 1) {
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ T *)src1);
        MicroAPI::LoadUnAlign(preReg, ureg, (__ubuf__ T *)src1);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1], (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(srcReg1, dupReg, maskFull);
    }

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign<T>(srcReg1, src1 + i * repeatStride);
        } else if (scalarIdx == 1) {
            MicroAPI::LoadAlign<T>(srcReg0, src0 + i * repeatStride);
        }
        if constexpr (op == BinaryScalarOp::ADDS) {
            MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::MULS) {
            MicroAPI::Mul(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::MAXS) {
            MicroAPI::Max(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::MINS) {
            MicroAPI::Min(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::SUBS) {
            MicroAPI::Sub(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::DIVS) {
            MicroAPI::Div(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::ANDS) {
            MicroAPI::And(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::ORS) {
            MicroAPI::Or(dstReg, srcReg0, srcReg1, mask);
        }
        MicroAPI::StoreAlign<T>(dst + i * repeatStride, dstReg, mask);
    }
}

template <typename T, MicroAPI::LoadDist pattern, BinaryScalarOp op, uint8_t scalarIdx>
__aicore__ inline void BinaryScalarOpTemplateCnt(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &calCount)
{
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::RegTensor<T> vDstReg0;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg preg;

    uint32_t sregLower = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        preg = MicroAPI::UpdateMask<T>(sreg);

        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign<T, pattern>(vSrcReg0, src0);
            MicroAPI::LoadAlign(vSrcReg1, src1 + i * sregLower);
        } else if constexpr (scalarIdx == 1) {
            MicroAPI::LoadAlign(vSrcReg0, src0 + i * sregLower);
            MicroAPI::LoadAlign<T, pattern>(vSrcReg1, src1);
        }

        if constexpr (op == BinaryScalarOp::ADDS) {
            MicroAPI::Add(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::MULS) {
            MicroAPI::Mul(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::MAXS) {
            MicroAPI::Max(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::MINS) {
            MicroAPI::Min(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::SUBS) {
            MicroAPI::Sub(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::DIVS) {
            MicroAPI::Div(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::ANDS) {
            MicroAPI::And(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::ORS) {
            MicroAPI::Or(vDstReg0, vSrcReg0, vSrcReg1, preg);
        }

        MicroAPI::StoreAlign(dst + i * sregLower, vDstReg0, preg);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Adds not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Adds not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t,
        complex32, complex64>()),
        "Adds not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 1) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B8, BinaryScalarOp::ADDS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::ADDS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::ADDS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::ADDS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Maxs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 1) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B8, BinaryScalarOp::MAXS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::MAXS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::MAXS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::MAXS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Mins not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Mins not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Mins not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 1) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B8, BinaryScalarOp::MINS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::MINS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::MINS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::MINS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Muls not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Muls not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t, complex32, complex64>()),
        "Muls not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::MULS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::MULS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::MULS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t, complex32, complex64>()),
        "Subs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::SUBS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::SUBS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::SUBS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t, complex32, complex64>()),
        "Divs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::DIVS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::DIVS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::DIVS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ands not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::ANDS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::ANDS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::ANDS, scalarIdx>>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ors not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::ORS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::ORS, scalarIdx>>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VF_CALL<BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::ORS, scalarIdx>>(dst, src0, src1, calCount);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
