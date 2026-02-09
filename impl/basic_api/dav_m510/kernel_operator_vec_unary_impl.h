/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_vec_template_impl.h"
#include "micro_api/kernel_micro_intf.h"

namespace AscendC {
namespace Internal {
template <auto func, typename T, typename RegType>
__aicore__ inline void VecUnaryLevel2VFImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    RegType srcReg;
    RegType dstReg;
    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::MaskReg mask;
    constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() / sizeof(T) * RegType::trait.REG_NUM);
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(sreg, repeatStride));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, RegType::trait>(sreg);
        MicroAPI::LoadAlign(srcReg, src + i * repeatStride);
        func(dstReg, srcReg, mask);
        MicroAPI::StoreAlign(dst + i * repeatStride, dstReg, mask);
    }
}

template <auto func, typename T>
__aicore__ inline void VecUnaryLevel2ImplTemplate(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    if constexpr (SupportBytes<T, 8>()) {
        VF_CALL<VecUnaryLevel2VFImpl<func, T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src, count);
    } else {
        VF_CALL<VecUnaryLevel2VFImpl<func, T, MicroAPI::RegTensor<T>>>(dst, src, count);
    }
}

template <typename T>
__simd_vf__ inline void VecUnaryLevel2ImplFloat(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count) {
    static_assert(SupportType<T, float>(), "Failed to check dtype in Rsqrt FAST_INVERSE mode, "
        "current api only supports float. ");
    constexpr uint32_t sregLower = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    constexpr uint32_t posZero = 0x00000000u;
    constexpr uint32_t negZero = 0x80000000u;
    constexpr uint32_t posInf = 0x7f800000u;
    constexpr float subnormalBound = 1.1754944e-38;
    constexpr float halfFactor = 0.5f;
    constexpr float negHalfFactor = -0.5f;
    constexpr float oneHalf = 1.5f;
    constexpr float negOne = -1.0f;
    constexpr float multiplyFactor0 = 16777216.0f;
    constexpr float multiplyFactor1 = 4096.0f;
    uint32_t sreg = static_cast<uint32_t>(count);
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, sregLower));
    NotNumUnion notNum0;
    notNum0.i = F32_INF;
    NotNumUnion notNum1;
    notNum1.i = F32_NEG_INF;
    /*
    * Improves Basic Api with high precision mode by using fast_inverse approach with following formula.
    * if x == float("inf"):
    * return 0.0f
    * if x == 0.0f:
    * return float("inf")
    * float r, r1, y, s, t, e;
    * bool p;
    * p = (x < 1.1754944e-38);
    * if (p)
    * x = x*16777216.0f;
    * r = errdiv(1.0, x); // div 指令
    * y = errsqrt(r); // sqrt 指令
    * y = y*(1.5 - 0.5*x*y*y);
    * s = 1 - x*r;
    * t = r - y*y;
    * e = s + x*t;
    * y = y + y*e*0.5;
    * if (p)
    * y = y*4096.0f; // y = y*2**12, 返回input是subnorma的结果
    * return y;
    */
    MicroAPI::RegTensor<T> regZero;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> regOne;
    MicroAPI::RegTensor<T> regOneHalf;
    MicroAPI::RegTensor<T> tmpReg;
    MicroAPI::RegTensor<T> divReg;
    MicroAPI::RegTensor<T> mulReg;
    MicroAPI::RegTensor<T> resReg;
    MicroAPI::RegTensor<T> negInfReg;
    MicroAPI::RegTensor<T> posInfReg;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg isInfMask;
    MicroAPI::MaskReg isPosZeroMask;
    MicroAPI::MaskReg isNegZeroMask;
    MicroAPI::MaskReg cmpMask;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T>();
    MicroAPI::Duplicate(regZero, 0.0f, maskFull);
    MicroAPI::Duplicate(posInfReg, notNum0.f, maskFull);
    MicroAPI::Duplicate(negInfReg, notNum1.f, maskFull);
    for (uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::LoadAlign(srcReg, src + i * sregLower);

        MicroAPI::Duplicate(regOne, 1.0f, maskFull);
        MicroAPI::Duplicate(regOneHalf, oneHalf, maskFull);

        MicroAPI::CompareScalar<T, CMPMODE::LT>(cmpMask, srcReg, subnormalBound, mask);
        MicroAPI::Muls(tmpReg, srcReg, multiplyFactor0, mask);
        MicroAPI::Select(srcReg, tmpReg, srcReg, cmpMask);

        MicroAPI::Div(divReg, regOne, srcReg, mask);           // r = errdiv(1.0, x);
        MicroAPI::Sqrt(resReg, divReg, mask);                  // y = errsqrt(r);
        MicroAPI::Muls(tmpReg, srcReg, negHalfFactor, mask);   // -0.5x
        MicroAPI::Mul(mulReg, tmpReg, resReg, mask);           // -0.5xy
        MicroAPI::MulAddDst(regOneHalf, mulReg, resReg, mask); // 1.5 - 0.5xy*y
        MicroAPI::Mul(resReg, regOneHalf, resReg, mask);       // y = y * (1.5 + (-0.5*x*y) * y)

        MicroAPI::Muls(tmpReg, srcReg, negOne, mask);      // -x
        MicroAPI::MulAddDst(regOne, tmpReg, divReg, mask); // s = 1 - x*r
        MicroAPI::Muls(tmpReg, resReg, negOne, mask);      // -y
        MicroAPI::MulAddDst(divReg, tmpReg, resReg, mask); // t = r + (-y) * y
        // e = s + x * t => s = s + x * t
        MicroAPI::MulAddDst(regOne, srcReg, divReg, mask);
        // y = y + y * e * 0.5
        MicroAPI::Muls(mulReg, resReg, halfFactor, mask);  // 0.5*y
        MicroAPI::MulAddDst(resReg, mulReg, regOne, mask); // y = y + s*0.5y

        MicroAPI::Muls(tmpReg, resReg, multiplyFactor1, mask);
        MicroAPI::Select(dstReg, tmpReg, resReg, cmpMask);

        MicroAPI::CompareScalar(isInfMask, (MicroAPI::RegTensor<uint32_t> &)srcReg, posInf, mask);
        MicroAPI::Select(dstReg, regZero, dstReg, isInfMask);
        MicroAPI::CompareScalar(isPosZeroMask, (MicroAPI::RegTensor<uint32_t> &)srcReg, posZero, mask);
        MicroAPI::Select(dstReg, posInfReg, dstReg, isPosZeroMask);
        MicroAPI::CompareScalar(isNegZeroMask, (MicroAPI::RegTensor<uint32_t> &)srcReg, negZero, mask);
        MicroAPI::Select(dstReg, negInfReg, dstReg, isNegZeroMask);

        MicroAPI::StoreAlign(dst + i * sregLower, dstReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void VecUnaryLevel2ImplB64(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count) {
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg0;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg0;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vRegOne, vRegNegOne, vRegZero, vRegF;
    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::MaskReg preg, maskZero, maskOne, maskNegOne;
    uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH_2XVL / sizeof(T));
    uint16_t repeatTime = CeilDivision(count, sregLower);
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        MicroAPI::Duplicate(vRegOne, T(1), preg);
        MicroAPI::Duplicate(vRegZero, T(0), preg);
        MicroAPI::Duplicate(vRegF, static_cast<T>(0xffffffffffffffff), preg);
        MicroAPI::LoadAlign(vSrcReg0, src + i * sregLower);
        MicroAPI::CompareScalar(maskZero, vSrcReg0, T(0), preg);
        MicroAPI::Select(vDstReg0, vRegF, vRegZero, maskZero);
        MicroAPI::CompareScalar(maskOne, vSrcReg0, T(1), preg);
        MicroAPI::Select(vDstReg0, vRegOne, vDstReg0, maskOne);
        if constexpr (IsSameType<T, int64_t>::value) {
            MicroAPI::Duplicate(vRegNegOne, T(-1), preg);
            MicroAPI::CompareScalar(maskNegOne, vSrcReg0, T(-1), preg);
            MicroAPI::Select(vDstReg0, vRegNegOne, vDstReg0, maskNegOne);
        }
        MicroAPI::StoreAlign(dst + i * sregLower, vDstReg0, preg);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T>
__aicore__ inline void VecUnaryLevel0VFImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, const UnaryRepeatParams &repeatParams,
    __ubuf__ uint64_t *maskBuf)
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
        func(dstVreg, srcVreg, maskReg);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, dstVreg, repeatParams.dstBlkStride, maskReg);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, typename T>
__aicore__ inline void VecUnaryLevel0Template(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
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
        VF_CALL<VecUnaryLevel0VFImpl<func, isSetMask, isMaskBitMode, false, T>>(dst, src, maskArray, maskCount,
            repeatTime, repeatParams, maskBuf);
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
        VF_CALL<VecUnaryLevel0VFImpl<func, isSetMask, isMaskBitMode, true, T>>(dst, src, maskArray, maskCount,
            repeatTime, repeatParams, maskBuf);
        if constexpr (isMaskBitMode && SupportBytes<T, 1>()) {
            AscendC::AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    }
}
} // namespace Internal

template <typename T, bool isSetMask = true, const ExpConfig& config>
__aicore__ inline void ExpImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == ExpAlgo::INTRINSIC || config.algo == ExpAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == ExpAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::ExpSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, ExpAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Exp<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const ExpConfig& config>
__aicore__ inline void ExpImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == ExpAlgo::INTRINSIC || config.algo == ExpAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == ExpAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::ExpSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, ExpAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Exp<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const LnConfig& config>
__aicore__ inline void LnImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == LnAlgo::INTRINSIC || config.algo == LnAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Ln<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == LnAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::LnSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, LnAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Ln<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const LnConfig& config>
__aicore__ inline void LnImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == LnAlgo::INTRINSIC || config.algo == LnAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Ln<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == LnAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::LnSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, LnAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Ln<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AbsImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, int16_t, float, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Abs<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AbsImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, int16_t, float, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Abs<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
}

namespace MicroAPIReciprocal {
template <typename T, typename RegT, bool precisionMode = false>
__aicore__ inline void Reciprocal(RegT &dstReg, RegT &srcReg, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, 1.0f, mask);
    if constexpr (!precisionMode) {
        MicroAPI::Div(dstReg, dstReg, srcReg, mask);
    } else {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = 
                                        {MicroAPI::MaskMergeMode::ZEROING, true, DivAlgo::PRECISION_1ULP_FTZ_FALSE};
        MicroAPI::Div<T, &mode>(dstReg, dstReg, srcReg, mask);
    }
}
} // namespace MicroAPIReciprocal
template <typename T, bool isSetMask = true, const ReciprocalConfig& config>
__aicore__ inline void ReciprocalImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == ReciprocalAlgo::INTRINSIC || config.algo == ReciprocalAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPIReciprocal::Reciprocal<T, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE) {
        constexpr auto func = MicroAPIReciprocal::Reciprocal<T, MicroAPI::RegTensor<T>, true>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const ReciprocalConfig& config>
__aicore__ inline void ReciprocalImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == ReciprocalAlgo::INTRINSIC || config.algo == ReciprocalAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPIReciprocal::Reciprocal<T, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE) {
        constexpr auto func = MicroAPIReciprocal::Reciprocal<T, MicroAPI::RegTensor<T>, true>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const SqrtConfig& config>
__aicore__ inline void SqrtImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == SqrtAlgo::INTRINSIC || config.algo == SqrtAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Sqrt<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == SqrtAlgo::FAST_INVERSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, true, SqrtAlgo::FAST_INVERSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == SqrtAlgo::PRECISION_0ULP_FTZ_FALSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_0ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == SqrtAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const SqrtConfig& config>
__aicore__ inline void SqrtImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == SqrtAlgo::INTRINSIC || config.algo == SqrtAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Sqrt<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == SqrtAlgo::FAST_INVERSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, true, SqrtAlgo::FAST_INVERSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == SqrtAlgo::PRECISION_0ULP_FTZ_FALSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_0ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == SqrtAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    }
}

namespace MicroAPIRsqrt {
template <typename T, typename RegT, bool precisionMode = false> __aicore__ inline void Rsqrt(RegT &dstReg, RegT &srcReg, MicroAPI::MaskReg &mask)
{
    MicroAPI::MaskReg cmpMask;
    MicroAPI::Duplicate(dstReg, static_cast<T>(1.0f), mask);
    MicroAPI::CompareScalar<T, CMPMODE::LT>(cmpMask, srcReg, static_cast<T>(0.0f), mask);
    if constexpr (!precisionMode) {
        MicroAPI::Sqrt(srcReg, srcReg, mask);
        MicroAPI::Div(dstReg, dstReg, srcReg, mask);
        MicroAPI::Select(dstReg, srcReg, dstReg, cmpMask);
    } else {
        if constexpr (SupportType<T, half>()) {
            static constexpr AscendC::MicroAPI::SqrtSpecificMode SqrtMode = 
                                    {MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_1ULP_FTZ_FALSE};
            MicroAPI::Sqrt<T, &SqrtMode>(srcReg, srcReg, mask);
            static constexpr AscendC::MicroAPI::DivSpecificMode divMode = 
                                    {MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_1ULP_FTZ_FALSE};
            MicroAPI::Div<T, &divMode>(dstReg, dstReg, srcReg, mask);
        } else {
            static constexpr AscendC::MicroAPI::SqrtSpecificMode SqrtMode = 
                                    {MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
            MicroAPI::Sqrt<T, &SqrtMode>(srcReg, srcReg, mask);
            static constexpr AscendC::MicroAPI::DivSpecificMode divMode = 
                                    {MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_0ULP_FTZ_FALSE};
            MicroAPI::Div<T, &divMode>(dstReg, dstReg, srcReg, mask);
        }
    }
}
} // namespace MicroAPIRsqrt
template <typename T, bool isSetMask = true, const RsqrtConfig& config>
__aicore__ inline void RsqrtImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == RsqrtAlgo::INTRINSIC || config.algo == RsqrtAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPIRsqrt::Rsqrt<T, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == RsqrtAlgo::FAST_INVERSE || config.algo == RsqrtAlgo::PRECISION_0ULP_FTZ_FALSE || 
                        config.algo == RsqrtAlgo::PRECISION_1ULP_FTZ_FALSE) {
        constexpr auto func = MicroAPIRsqrt::Rsqrt<T, MicroAPI::RegTensor<T>, true>;
        Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, const RsqrtConfig& config>
__aicore__ inline void RsqrtImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == RsqrtAlgo::INTRINSIC || config.algo == RsqrtAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPIRsqrt::Rsqrt<T, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == RsqrtAlgo::FAST_INVERSE || config.algo == RsqrtAlgo::PRECISION_0ULP_FTZ_FALSE || 
                        config.algo == RsqrtAlgo::PRECISION_1ULP_FTZ_FALSE) {
        constexpr auto func = MicroAPIRsqrt::Rsqrt<T, MicroAPI::RegTensor<T>, true>;
        Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void NotImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, uint16_t, int16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Not<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void NotImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, uint16_t, int16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Not<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReluImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int32_t>()), "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Relu<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecUnaryLevel0Template<func, isSetMask, true>(dst, src, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReluImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int32_t>()), "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Relu<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecUnaryLevel0Template<func, isSetMask, false>(dst, src, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, const ExpConfig& config> __aicore__ inline void ExpImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == ExpAlgo::INTRINSIC || config.algo == ExpAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else if constexpr (config.algo == ExpAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::ExpSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, ExpAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Exp<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}

template <typename T, const LnConfig& config> __aicore__ inline void LnImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == LnAlgo::INTRINSIC || config.algo == LnAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Ln<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else if constexpr (config.algo == LnAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::LnSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, LnAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Ln<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}

template <typename T> __aicore__ inline void AbsImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, int8_t, half, int16_t, float, int32_t, int64_t>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func =
            MicroAPI::Abs<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else {
        constexpr auto func = MicroAPI::Abs<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}

template <typename T, typename U, typename std::enable_if<!IsSameType<T, U>::value, bool>::type = true>
__aicore__ inline void AbsImpl(__ubuf__ T *dst, __ubuf__ U *src, const uint32_t count)
{
    static_assert(SupportType<T, half, float>() && SupportType<U, complex32, complex64>(),
        "current data type is not supported on current device!");
    static_assert(std::is_same_v<T, typename U::EleType>, "dst type do not match with src complex elements' type");
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vDstReg0;
        uint32_t sreg = (uint32_t)count;
        MicroAPI::MaskReg preg;
        static constexpr uint32_t repeatStride =
            static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U) * MicroAPI::RegTraitNumTwo.REG_NUM);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, repeatStride));
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = MicroAPI::UpdateMask<U, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::LoadAlign(vSrcReg0, src + i * repeatStride);
            MicroAPI::Abs<T, U, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne>,
                MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>>(vDstReg0, vSrcReg0, preg);
            MicroAPI::StoreAlign(dst + i * repeatStride, vDstReg0, preg);
        }
    }
}

template <typename T, const ReciprocalConfig& config> __aicore__ inline void ReciprocalImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t>()),
        "current data type is not supported on current device!");

    if constexpr (SupportType<T, half, float>()) {
        if constexpr (config.algo == ReciprocalAlgo::INTRINSIC || config.algo == ReciprocalAlgo::PRECISION_1ULP_FTZ_TRUE) {
            constexpr auto func = MicroAPIReciprocal::Reciprocal<T, MicroAPI::RegTensor<T>>;
            Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
        } else if constexpr (config.algo == ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE) {
            constexpr auto func = MicroAPIReciprocal::Reciprocal<T, MicroAPI::RegTensor<T>, true>;
            Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
        }
    } else {
        Internal::VecUnaryLevel2ImplB64<T>(dst, src, count);
    }
}

template <typename T, const SqrtConfig& config> __aicore__ inline void SqrtImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == SqrtAlgo::INTRINSIC || config.algo == SqrtAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Sqrt<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else if constexpr (config.algo == SqrtAlgo::FAST_INVERSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, true, SqrtAlgo::FAST_INVERSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else if constexpr (config.algo == SqrtAlgo::PRECISION_0ULP_FTZ_FALSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_0ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else if constexpr (config.algo == SqrtAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::SqrtSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_1ULP_FTZ_FALSE };
        constexpr auto func = MicroAPI::Sqrt<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}

template <typename T>
__aicore__ inline void RsqrtPrecisionModeImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t &count)
{
    if constexpr (IsSameType<T, half>::value) {
        constexpr auto func = MicroAPIRsqrt::Rsqrt<T, MicroAPI::RegTensor<T>, true>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else {
        Internal::VecUnaryLevel2ImplFloat<T>(dst, src, count);
    }
}

template <typename T, const RsqrtConfig& config> __aicore__ inline void RsqrtImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
    if constexpr (config.algo == RsqrtAlgo::INTRINSIC || config.algo == RsqrtAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPIRsqrt::Rsqrt<T, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else {
        RsqrtPrecisionModeImpl(dst, src, count);
    }
}

template <typename T> __aicore__ inline void NotImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert(
        (SupportType<T, int8_t, uint8_t, half, float, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func =
            MicroAPI::Not<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else {
        constexpr auto func = MicroAPI::Not<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}

template <typename T> __aicore__ inline void ReluImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, half, float, int32_t, int64_t>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func =
            MicroAPI::Relu<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else {
        constexpr auto func = MicroAPI::Relu<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}

/* **************************************************************************************************
 * Neg                                            *
 * ************************************************************************************************* */
// Neg::Level 2
template <typename T> __aicore__ inline void NegImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t count)
{
    static_assert((SupportType<T, int8_t, int16_t, int32_t, half, float, int64_t>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func =
            MicroAPI::Neg<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    } else {
        constexpr auto func = MicroAPI::Neg<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecUnaryLevel2ImplTemplate<func, T>(dst, src, count);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H