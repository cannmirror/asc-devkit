/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file kernel_operator_vec_ternary_scalar_impl.h
 * \brief AscendC c310 support vaxpy level 0/2 api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_operator_common_impl.h"
#include "kernel_utils.h"

namespace AscendC {
namespace Internal {
template <auto func, typename T, typename U, typename RegT, typename RegU>
__simd_vf__ inline void VecAxpyLevel2VFImpl(__ubuf__ T *dst, __ubuf__ U *src, U scalarValue, const uint32_t calCount)
{
    RegU srcReg;
    RegT dstReg;
    uint32_t count = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg mask;
    constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() / sizeof(T) * RegT::trait.REG_NUM);
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, RegT::trait>(count);
        MicroAPI::DataCopy(srcReg, src + i * repeatStride);
        MicroAPI::DataCopy(dstReg, dst + i * repeatStride);
        func(dstReg, srcReg, scalarValue, mask);
        MicroAPI::DataCopy(dst + i * repeatStride, dstReg, mask);
    }
}

template <auto func, typename T, typename U>
__aicore__ inline void VecAxpyLevel2ImplTemplate(__ubuf__ T *dst, __ubuf__ U *src, U scalarValue,
    const uint32_t calCount)
{
    if constexpr (SupportBytes<T, 8>()) {
        VecAxpyLevel2VFImpl<func, T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>>(dst, src, scalarValue, calCount);
    } else {
        VecAxpyLevel2VFImpl<func, T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>>(dst, src, scalarValue,
            calCount);
    }
}

/*
 * T: data type
 * func: MicroAPI input/output function
 * isSetMask: basic api whether to set mask
 * isNormalMode: true: NormalMode, false: CounterMode
 * isMaskBitMode: true: mask bit mode, false: mask count mode
 */
template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T, typename U>
__simd_vf__ inline void VecAxpyVFImpl(__ubuf__ T *dst, __ubuf__ U *src, U scalarValue, const maskStruct maskArrayStruct,
    const uint64_t maskCount, const uint8_t repeatTime, const UnaryRepeatParams repeatParams,
    __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArrayStruct.maskArray, maskCount, maskBuf);
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
        MicroAPI::RegTensor<U> srcVreg;
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<U, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcVreg,
            src + index * repeatParams.srcRepStride * ElePerBlkU, repeatParams.srcBlkStride, maskRegSrc);

        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dstVreg,
            dst + index * repeatParams.dstRepStride * ElePerBlkT, repeatParams.dstBlkStride, maskRegDst);
        func(dstVreg, srcVreg, scalarValue, maskReg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, dstVreg, repeatParams.dstBlkStride, maskRegDst);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, typename T, typename U>
__aicore__ inline void VecAxpyImplTemplate(__ubuf__ T *dst, __ubuf__ U *src, U scalarValue, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    constexpr bool TUCompare = sizeof(T) > sizeof(U);
    using TT = typename Conditional<TUCompare, T, U>::type;
    if constexpr (isMaskBitMode) {
        ASCENDC_ASSERT(maskCount == 0, "maskCount must be 0 when isMaskBitMode is true.");
    } else {
        ASCENDC_ASSERT(maskArray == nullptr, "maskArray must be nullptr when isMaskBitMode is false.");
    }
    __ubuf__ uint64_t *maskBuf = nullptr;

    uint16_t maskArraySize = (maskArray == nullptr) ? 0 : MASK_ARRAY_SIZE;
    maskStruct maskArrayStruct;
    for (uint16_t i = 0; i < maskArraySize; i++) {
        maskArrayStruct.maskArray[i] = maskArray[i];
    }
    
    if (Internal::IsCounterMode()) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2); // maskReg 256bit PK-> 128bit
        }
        VecAxpyVFImpl<func, isSetMask, isMaskBitMode, false, T, U>(dst, src, scalarValue, maskArrayStruct, maskCount,
            repeatTime, repeatParams, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    } else {
        if constexpr (isMaskBitMode && isSetMask) {
            SetVectorMask<TT>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
        }
        VecAxpyVFImpl<func, isSetMask, isMaskBitMode, true, T, U>(dst, src, scalarValue, maskArrayStruct, maskCount,
            repeatTime, repeatParams, maskBuf);
    }
}
} // namespace Internal

namespace MicroAPIAxpy {
namespace CastParam {
constexpr MicroAPI::CastTrait half2floatTrait = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };
}
template <typename T, typename U, typename RegT, typename RegU>
__simd_callee__ inline void Axpy(RegT &dstReg, RegU &srcReg, U scalarValue, MicroAPI::MaskReg &mask)
{
    if constexpr (SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<uint64_t, uint64_t>,
        Tuple<int64_t, int64_t>>()) {
        MicroAPI::Axpy(dstReg, srcReg, scalarValue, mask);
    } else if constexpr (SupportType<Tuple<T, U>, Tuple<bfloat16_t, bfloat16_t>>()) {
        RegT tmpReg;
        MicroAPI::Duplicate(tmpReg, scalarValue, mask);
        MicroAPI::Mul(tmpReg, srcReg, tmpReg, mask);
        MicroAPI::Add(dstReg, tmpReg, dstReg, mask);
    } else if constexpr (SupportType<Tuple<T, U>, Tuple<float, half>>()) {
        RegU tmpReg;
        RegT cvtReg;
        MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint32_t> &)tmpReg, (MicroAPI::RegTensor<uint16_t> &)srcReg);
        MicroAPI::Cast<float, half, CastParam::half2floatTrait>(cvtReg, tmpReg, mask);
        MicroAPI::Muls(cvtReg, cvtReg, static_cast<T>(scalarValue), mask);
        MicroAPI::Add(dstReg, cvtReg, dstReg, mask);
    }
}
} // namespace MicroAPIAxpy

// Axpy::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T *dst, __ubuf__ U *src, const U &scalarValue, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<bfloat16_t, bfloat16_t>,
        Tuple<float, half>>(),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPIAxpy::Axpy<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>>;
    Internal::VecAxpyImplTemplate<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime, repeatParams);
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T *dst, __ubuf__ U *src, const U &scalarValue, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<bfloat16_t, bfloat16_t>,
        Tuple<float, half>>(),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPIAxpy::Axpy<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>>;
    Internal::VecAxpyImplTemplate<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Axpy::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T *dst, __ubuf__ U *src, const U &scalarValue, const int32_t &calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<bfloat16_t, bfloat16_t>,
        Tuple<float, half>, Tuple<uint64_t, uint64_t>, Tuple<int64_t, int64_t>>(),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIAxpy::Axpy<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>>;
        Internal::VecAxpyLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIAxpy::Axpy<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>>;
        Internal::VecAxpyLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
