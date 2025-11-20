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
 * \file kernel_micro_vec_unary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H

namespace AscendC {
namespace MicroAPI {
namespace Internal {
__aicore__ inline constexpr SqrtSpecificMode GetSqrtSpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .precisionMode = false,
        .algo = SqrtAlgo::INTRINSIC
    };
}

__aicore__ inline constexpr SqrtSpecificMode GetSqrtSpecificMode(const SqrtSpecificMode *sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .precisionMode = sprMode->precisionMode,
        .algo = sprMode->algo
    };
}

__aicore__ inline constexpr ExpSpecificMode GetExpSpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .algo = ExpAlgo::INTRINSIC
    };
}

__aicore__ inline constexpr ExpSpecificMode GetExpSpecificMode(const ExpSpecificMode *sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .algo = sprMode->algo
    };
}

__aicore__ inline constexpr LnSpecificMode GetLnSpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .algo = LnAlgo::INTRINSIC
    };
}

__aicore__ inline constexpr LnSpecificMode GetLnSpecificMode(const LnSpecificMode *sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .algo = sprMode->algo
    };
}

__aicore__ inline constexpr LogSpecificMode GetLogSpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .algo = LogAlgo::INTRINSIC
    };
}

__aicore__ inline constexpr LogSpecificMode GetLogSpecificMode(const LogSpecificMode *sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .algo = sprMode->algo
    };
}

__aicore__ inline constexpr Log2SpecificMode GetLog2SpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .algo = Log2Algo::INTRINSIC
    };
}

__aicore__ inline constexpr Log2SpecificMode GetLog2SpecificMode(const Log2SpecificMode *sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .algo = sprMode->algo
    };
}

__aicore__ inline constexpr Log10SpecificMode GetLog10SpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .algo = Log10Algo::INTRINSIC
    };
}

__aicore__ inline constexpr Log10SpecificMode GetLog10SpecificMode(const Log10SpecificMode *sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .algo = sprMode->algo
    };
}
} // namespace Internal
template <MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void AbsB64Impl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(sizeof(ActualT) == 8, "AbsB64Impl data type should be B64");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "AbsB64Impl RegT should be RegTraitNumTwo");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    RegTensor<int32_t, RegTraitNumOne> zeroReg, lowReg, highReg;
    MaskReg carryMask, carryLow, carryHigh;
    vbr(zeroReg, 0);
    vcmp_lt(carryMask, (RegTensor<int32_t> &)srcReg.reg[1], zeroReg, mask);
    SubCarryOut(carryLow, lowReg, zeroReg, (RegTensor<int32_t> &)srcReg.reg[0], carryMask);
    SubCarryOuts(carryHigh, highReg, zeroReg, (RegTensor<int32_t> &)srcReg.reg[1], carryLow, carryMask);
    vsel((RegTensor<int32_t> &)dstReg.reg[0], lowReg, (RegTensor<int32_t> &)srcReg.reg[0], carryMask);
    vsel((RegTensor<int32_t> &)dstReg.reg[1], highReg, (RegTensor<int32_t> &)srcReg.reg[1], carryMask);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void AbsImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t, half, float, int64_t>(),
        "current data type is not supported on current device!");
    static_assert(
        SupportEnum<mode, MaskMergeMode::ZEROING>(), "current Abs api only supported Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (sizeof(ActualT) != 8) {
        vabs(dstReg, srcReg, mask, modeValue);
    } else {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            B64TraitOneToTaitTwo(traitTwoSrcReg, srcReg);
            AbsB64Impl<mode>(traitTwoDstReg, traitTwoSrcReg, maskTrait2);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            RegT dstTemp;
            AbsB64Impl<mode>(dstTemp, srcReg, mask);
            dstReg = dstTemp;
        }
    }
}

template <typename T = DefaultType, typename U = DefaultType, auto mode = MaskMergeMode::ZEROING,
    typename RegT, typename RegU>
__simd_callee__ inline void ComplexAbsKernel(RegT& dstReg, RegU& srcReg, MaskReg& mask)
{
    using ActualU = typename RegU::ActualT;
    static_assert(CheckRegTrait<RegU, RegTraitNumTwo>(), "RegU should be RegTraitNumTwo");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegT should be RegTraitNumOne");
    static_assert(Std::is_same_v<typename RegT::ActualT, typename ActualU::EleType>,
        "dst type do not match with src complex elements' type");

    RegTensor<typename ActualU::EleType> &srcReal = (RegTensor<typename ActualU::EleType> &)srcReg.reg[0];
    RegTensor<typename ActualU::EleType> &srcImag = (RegTensor<typename ActualU::EleType> &)srcReg.reg[1];
    RegTensor<typename ActualU::EleType> srcRealSquare;
    RegTensor<typename ActualU::EleType> srcImagSquare;

    Mul<T, mode, RegTensor<typename ActualU::EleType>>(srcRealSquare, srcReal, srcReal, mask);
    Mul<T, mode, RegTensor<typename ActualU::EleType>>(srcImagSquare, srcImag, srcImag, mask);
    Add<T, mode, RegTensor<typename ActualU::EleType>>(srcRealSquare, srcRealSquare, srcImagSquare, mask);
    Sqrt<T, mode, RegTensor<typename ActualU::EleType>>(dstReg, srcRealSquare, mask);
}

template <typename T = DefaultType, typename U = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT, typename RegU>
__simd_callee__ inline void AbsImpl(RegT& dstReg, RegU& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(Std::is_same_v<U, DefaultType> || Std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert(SupportType<ActualU, complex32, complex64>(),
        "current src data type ActualU is not supported on current device!");
    static_assert(SupportType<ActualT, half, float>(),
        "current dst data type ActualT is not supported on current device!");
    static_assert(Std::is_same_v<ActualT, typename ActualU::EleType>,
        "dst type do not match with src complex elements' type");
    static_assert(SupportEnum<mode, MaskMergeMode::ZEROING>(),
        "current Abs api only supported Mode ZEROING on current device!");
    if constexpr (CheckRegTrait<RegU, RegTraitNumOne>()) {
        MaskReg maskTrait2;
        MaskPack(maskTrait2, mask);
        RegTensor<ActualU, RegTraitNumTwo> traitTwoSrcReg;
        TraitOneToTaitTwoTmpl<RegTensor<ActualU, RegTraitNumTwo>, RegTensor<ActualU, RegTraitNumOne>,
            typename ActualU::EleType>(traitTwoSrcReg, srcReg);
        ComplexAbsKernel<T, U, mode, RegTensor<ActualT, RegTraitNumOne>, RegTensor<ActualU, RegTraitNumTwo>>(
            dstReg, traitTwoSrcReg, maskTrait2);
    } else {
        ComplexAbsKernel<T, U, mode, RegTensor<ActualT, RegTraitNumOne>, RegTensor<ActualU, RegTraitNumTwo>>(
            dstReg, srcReg, mask);
    }
}

template <MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ReluB64Impl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(sizeof(ActualT) == 8, "RegT data type should be B64");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "RegT should be RegTraitNumTwo");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    Maxs(dstReg, srcReg, (int64_t)0, mask);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ReluImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, int32_t, half, float, int64_t>(), "current data type is not supported on current device!");
    static_assert(
        SupportEnum<mode, MaskMergeMode::ZEROING>(), "current Relu api only supported Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (sizeof(ActualT) != 8) {
        vrelu(dstReg, srcReg, mask, modeValue);
    } else {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            B64TraitOneToTaitTwo(traitTwoSrcReg, srcReg);
            ReluB64Impl<mode>(traitTwoDstReg, traitTwoSrcReg, maskTrait2);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            RegT dstTemp;
            ReluB64Impl<mode>(dstTemp, srcReg, mask);
            dstReg = dstTemp;
        }
    }
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ExpPrecision(RegT& dstReg, RegT& srcReg, MaskReg& maskSubnormal)
{
    RegT regTwo;
    RegT tmpReg0;
    using ActualT = typename RegT::ActualT;
    constexpr ExpSpecificMode sprMode = Internal::GetExpSpecificMode(mode);
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    vdup(regTwo, 2, maskSubnormal, modeValue);
    vdiv(tmpReg0, srcReg, regTwo, maskSubnormal, modeValue);
    vexp(tmpReg0, tmpReg0, maskSubnormal, modeValue);
    vmul(dstReg, tmpReg0, tmpReg0, maskSubnormal, modeValue);
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ExpImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(IsSameType<decltype(mode), MaskMergeMode>::value ||
        IsSameType<decltype(mode), const ExpSpecificMode*>::value,
        "mode type must be either MaskMergeMode or const ExpSpecificMode* ");
    constexpr ExpSpecificMode sprMode = Internal::GetExpSpecificMode(mode);
    static_assert(SupportEnum<sprMode.mrgMode, MaskMergeMode::ZEROING>(),
        "current Exp api only supported Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    if constexpr (sprMode.algo == ExpAlgo::PRECISION_1ULP_FTZ_FALSE) {
        MaskReg maskInf;
        MaskReg maskNegInf;
        MaskReg maskSubnormal;
        RegT tmpReg;
        if constexpr (SupportType<ActualT, float>()) {
            NotNumUnion posInf;
            posInf.i = 0x7f800000;
            NotNumUnion negInf;
            negInf.i = 0xff800000;
            NotNumUnion subnormalBound;
            subnormalBound.i = 0x7fffff;
            vexp(dstReg, srcReg, mask, modeValue);
            vcmps_le(maskSubnormal, dstReg, subnormalBound.f, mask);
            ExpPrecision(tmpReg, srcReg, maskSubnormal);
            vsel(dstReg, tmpReg, dstReg, maskSubnormal);
        } else {
            HalfUnion posInf;
            posInf.i = 0x7C00;
            HalfUnion negInf;
            negInf.i = 0xfC00;
            NotNumUnion subnormalBound;
            subnormalBound.i = 0x3ff;
            vexp(dstReg, srcReg, mask, modeValue);
            vcmps_le(maskSubnormal, dstReg, subnormalBound.f, mask);
            ExpPrecision(tmpReg, srcReg, maskSubnormal);
            vsel(dstReg, tmpReg, dstReg, maskSubnormal);
        }
    } else {
        vexp(dstReg, srcReg, mask, modeValue);
    }
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void SqrtFastInverseImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    constexpr SqrtSpecificMode sprMode = Internal::GetSqrtSpecificMode(mode);
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    /*
    * Improves MicroAPI with high precision mode by using fast_inverse approach with following formula.
    * bool p;
    * p = (b < 1);
    * if (p)
    *     b = b*16777216.0f;  // x = x*2**24, get rid of subnormal
    * float x = errrsqrt(b);  // rsqrt
    * float x1 = x*x;
    * float x2 = 1-b*x1;
    * //float x3 = x2*x*0.5;
    * x = x + x2*x*0.5;
    * x1 = x*b;
    * float err = b - x1*x1;
    * x2 = x*0.5f;
    * x2 = x2*err + x1;
    * if (p)
    *     x2 = x2*0.000244140625f; //x2 = x2 * 2**(-12), 返回input是subnormal的值
    * if (std::isinf(b) || b==0)
    *     x2 = b;
    * return x2;
    */

    constexpr float subnormalBound = 1;
    constexpr float halfFactor = 0.5f;
    constexpr float negOne = -1.0f;
    constexpr float multiplyFactor0 = 16777216.0f;
    constexpr float multiplyFactor1 = 0.000244140625f;
    constexpr uint32_t posInf = 0x7f800000u;
    constexpr uint32_t negZero = 0x80000000u;
    RegTensor<T> regOne;
    RegTensor<T> tmpReg;
    RegTensor<T> errReg;
    RegTensor<T> resReg;
    RegTensor<T> dstRegCopy;
    RegTensor<T> srcRegCopy = srcReg;
    RegTensor<uint32_t> regNegOne;
    RegTensor<uint32_t> zeroReg;

    MaskReg cmpMaskReg;
    MaskReg isInfPreg;
    MaskReg isZeroPreg;
    MaskReg maskFull;
    maskFull = pset_b8(PAT_ALL);

    vcmps_lt(cmpMaskReg, srcRegCopy, subnormalBound, mask);
    vmuls(tmpReg, srcRegCopy, multiplyFactor0, mask, modeValue);
    vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);

    vdup(regOne, 1.0f, maskFull, modeValue);
    vsqrt(tmpReg, srcRegCopy, mask, modeValue);
    vdiv(dstRegCopy, regOne, tmpReg, mask, modeValue);

    vmuls(tmpReg, dstRegCopy, negOne, mask, modeValue);      // -x
    vmul(errReg, dstRegCopy, srcRegCopy, mask, modeValue);   // b*x
    vmula(regOne, errReg, tmpReg, mask, modeValue);      // x2 = 1-b*x*x
    vmuls(tmpReg, dstRegCopy, halfFactor, mask, modeValue);  // 0.5x
    vmula(dstRegCopy, regOne, tmpReg, mask, modeValue);      // x = x + x2*0.5x

    vmul(resReg, dstRegCopy, srcRegCopy, mask, modeValue);   // x1 = x*b
    vmuls(tmpReg, resReg, negOne, mask, modeValue);      // -x1
    vmov(errReg, srcRegCopy);                            // err = b
    vmula(errReg, resReg, tmpReg, mask, modeValue);      // err = b - x1*x1
    vmuls(tmpReg, dstRegCopy, halfFactor, mask, modeValue);  // 0.5x
    vmadd(tmpReg, errReg, resReg, mask, modeValue);      // x2 = x2*err + x1

    vmuls(dstRegCopy, tmpReg, multiplyFactor1, mask, modeValue);
    vsel(tmpReg, dstRegCopy, tmpReg, cmpMaskReg);

    vcmps_eq(isInfPreg, (vector_u32 &)srcRegCopy, posInf, mask);
    vdup(regNegOne, negZero, maskFull, modeValue);
    vor(zeroReg, (vector_u32&)srcRegCopy, regNegOne, mask, modeValue);
    vcmps_eq(isZeroPreg, zeroReg, negZero, mask);
    por(cmpMaskReg, isZeroPreg, isInfPreg, mask);
    vsel(dstReg, srcRegCopy, tmpReg, cmpMaskReg);
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void SqrtImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(
        IsSameType<decltype(mode), MaskMergeMode>::value || IsSameType<decltype(mode), const SqrtSpecificMode *>::value,
        "mode type must be either MaskMergeMode or const SqrtSpecificMode* ");
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    constexpr SqrtSpecificMode sprMode = Internal::GetSqrtSpecificMode(mode);
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<sprMode.mrgMode, MaskMergeMode::ZEROING>(),
        "current Sqrt api only supports Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();

    if constexpr (sprMode.precisionMode) {
        static_assert(SupportType<T, float>(),
                            "MicroAPI Sqrt for high precision mode by using fast_inverse approach only supports float.");
        SqrtFastInverseImpl<T, mode, RegT>(dstReg, srcReg, mask);
    } else {
        if constexpr (sprMode.algo == SqrtAlgo::PRECISION_0ULP_FTZ_FALSE) {
            static_assert(SupportType<T, float>(),
                            "MicroAPI Sqrt for high precision mode by using fast_inverse approach only supports float.");
            SqrtFastInverseImpl<T, mode, RegT>(dstReg, srcReg, mask);
        } else if constexpr (sprMode.algo == SqrtAlgo::PRECISION_1ULP_FTZ_FALSE) {
            static_assert(SupportType<T, half>(),
                            "MicroAPI Sqrt for PRECISION_1ULP_FTZ_FALSE only supports half.");
            HalfUnion multiplyFactor0;
            multiplyFactor0.i = 0x6C00;
            HalfUnion multiplyFactor1;
            multiplyFactor1.i = 0x2400;
            HalfUnion subnormalThreshold;
            subnormalThreshold.i = 0x03FF;
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor0.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            vsqrt(dstRegCopy, srcRegCopy, mask, modeValue);
            vmuls(tmpReg, dstRegCopy, multiplyFactor1.f, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        } else {
            vsqrt(dstReg, srcReg, mask, modeValue);
        }
    }
}

template <typename T = DefaultType, typename RegT, const LogSpecificMode* mode>
__simd_callee__ inline void LnCompute(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    constexpr LogSpecificMode sprMode = Internal::GetLogSpecificMode(mode);
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    if constexpr (sprMode.algo == LogAlgo::PRECISION_1ULP_FTZ_FALSE) {
        if constexpr (IsSameType<ActualT, half>::value) {
            HalfUnion multiplyFactor;
            multiplyFactor.i = 0x6400; // 2^10
            HalfUnion subnormalThreshold;
            subnormalThreshold.i = 0x03FF;
            const half compensationFactor = -6.931471805599453094172;  // -Ln(2^10);
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            vln(dstRegCopy, srcRegCopy, mask, modeValue);
            vadds(tmpReg, dstRegCopy, compensationFactor, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        } else {
            NotNumUnion multiplyFactor;
            multiplyFactor.i = 0x4B000000; // 2^23;
            NotNumUnion subnormalThreshold;
            subnormalThreshold.i = 0x007FFFFF;
            constexpr float compensationFactor = -15.9423851528787421;  // -Ln(2^23);
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            vln(dstRegCopy, srcRegCopy, mask, modeValue);
            vadds(tmpReg, dstRegCopy, compensationFactor, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        }
    } else {
        vln(dstReg, srcReg, mask, modeValue);
    }
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void LogImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(IsSameType<decltype(mode), MaskMergeMode>::value ||
                      IsSameType<decltype(mode), const LogSpecificMode *>::value ||
                      IsSameType<decltype(mode), const LnSpecificMode *>::value,
                  "mode type must be MaskMergeMode or const LogSpecificMode* or const LnSpecificMode* ");
    if constexpr (IsSameType<decltype(mode), const LogSpecificMode *>::value) {
        constexpr LogSpecificMode sprMode = Internal::GetLogSpecificMode(mode);
        static_assert(SupportEnum<sprMode.mrgMode, MaskMergeMode::ZEROING>(),
            "current Log api only supports Mode ZEROING on current device!");
        LnCompute<T, RegT, mode>(dstReg, srcReg, mask);
    } else if constexpr (IsSameType<decltype(mode), const LnSpecificMode *>::value) {
        constexpr LnSpecificMode sprMode = Internal::GetLnSpecificMode(mode);
        static_assert(SupportEnum<sprMode.mrgMode, MaskMergeMode::ZEROING>(),
            "current Ln api only supports Mode ZEROING on current device!");
        if constexpr (sprMode.algo == LnAlgo::PRECISION_1ULP_FTZ_FALSE) {
            static constexpr AscendC::MicroAPI::LogSpecificMode logMode = {MaskMergeMode::ZEROING, LogAlgo::PRECISION_1ULP_FTZ_FALSE};
            LnCompute<T, RegT, &logMode>(dstReg, srcReg, mask);
        } else {
            static constexpr AscendC::MicroAPI::LogSpecificMode logMode = {MaskMergeMode::ZEROING, LogAlgo::INTRINSIC};
            LnCompute<T, RegT, &logMode>(dstReg, srcReg, mask);
        }
    } else {
        constexpr auto modeValue = GetMaskMergeMode<mode>();
        vln(dstReg, srcReg, mask, modeValue);
    }
}

template <MaskMergeMode mode = MaskMergeMode::ZEROING>
__simd_callee__ inline void LogXImpl(
    RegTensor<half>& dstReg, RegTensor<half>& srcReg, MaskReg& mask, const float lnXReciprocal)
{
    vector_f16 f16RegLow;
    vector_f16 f16RegHigh;
    vector_f32 f32RegLow;
    vector_f32 f32RegHigh;
    vector_bool MaskLow;
    vector_bool MaskHigh;
    vector_f16 tmpReg;
    constexpr auto patAll = std::integral_constant<::Pat, static_cast<::Pat>(MicroAPI::MaskPattern::ALL)>();
    vector_bool maskAll = pset_b32(patAll);
    constexpr auto patVal = std::integral_constant<::Pat, static_cast<::Pat>(MicroAPI::MaskPattern::H)>();
    vector_bool selMask = pset_b16(patVal);
    constexpr auto partModeEvenVal = std::integral_constant<::Part, static_cast<::Part>(RegLayout::ZERO)>();
    constexpr auto satModeValue =
        std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(SatMode::NO_SAT)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<RoundMode::CAST_RINT>()>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    punpack(MaskLow, mask, LOWER);
    punpack(MaskHigh, mask, HIGHER);

    vintlv(f16RegLow, f16RegHigh, srcReg, srcReg);

    vcvt(f32RegLow, f16RegLow, maskAll, partModeEvenVal, modeValue);
    vcvt(f32RegHigh, f16RegHigh, maskAll, partModeEvenVal, modeValue);

    vln(f32RegLow, f32RegLow, maskAll, modeValue);
    vln(f32RegHigh, f32RegHigh, maskAll, modeValue);

    vmuls(f32RegLow, f32RegLow, lnXReciprocal, maskAll, modeValue);
    vmuls(f32RegHigh, f32RegHigh, lnXReciprocal, maskAll, modeValue);

    vcvt(f16RegLow, f32RegLow, MaskLow, roundModeValue, satModeValue, partModeEvenVal, modeValue);
    vcvt(f16RegHigh, f32RegHigh, MaskHigh, roundModeValue, satModeValue, partModeEvenVal, modeValue);

    vdintlv(f16RegLow, tmpReg, f16RegLow, tmpReg);
    vdintlv(f16RegHigh, tmpReg, tmpReg, f16RegHigh);
    vsel(dstReg, f16RegLow, f16RegHigh, selMask);
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void Log2Impl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(
        IsSameType<decltype(mode), MaskMergeMode>::value || IsSameType<decltype(mode), const Log2SpecificMode *>::value,
        "mode type must be either MaskMergeMode or const Log2SpecificMode* ");
    constexpr Log2SpecificMode sprMode = Internal::GetLog2SpecificMode(mode);
    static_assert(SupportEnum<sprMode.mrgMode, MaskMergeMode::ZEROING>(),
        "current Log2 api only supports Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    constexpr float ln2Reciprocal = 1.4426950408889634;  // 1.0/Ln2;
    if constexpr (sprMode.algo == Log2Algo::PRECISION_1ULP_FTZ_FALSE) {
        if constexpr (SupportType<ActualT, half>()) {
            HalfUnion multiplyFactor;
            multiplyFactor.i = 0x6400; // 2^10
            HalfUnion subnormalThreshold;
            subnormalThreshold.i = 0x03FF;
            const half compensationFactor = -10;  // -Log2(2^10);
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            LogXImpl<sprMode.mrgMode>(dstRegCopy, srcRegCopy, mask, ln2Reciprocal);
            vadds(tmpReg, dstRegCopy, compensationFactor, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        } else {
            NotNumUnion multiplyFactor;
            multiplyFactor.i = 0x4B000000; // 2^23;
            NotNumUnion subnormalThreshold;
            subnormalThreshold.i = 0x007FFFFF;
            constexpr float compensationFactor = -23;  // -Log2(2^23);
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            vln(dstRegCopy, srcRegCopy, mask, modeValue);
            vmuls(dstRegCopy, dstRegCopy, ln2Reciprocal, mask, modeValue);
            vadds(tmpReg, dstRegCopy, compensationFactor, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        }
    } else {
        if constexpr (SupportType<ActualT, half>()) {
            LogXImpl<mode>(dstReg, srcReg, mask, ln2Reciprocal);
        } else {
            vln(dstReg, srcReg, mask, modeValue);
            vmuls(dstReg, dstReg, ln2Reciprocal, mask, modeValue);
        }
    }
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void Log10Impl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(
        IsSameType<decltype(mode), MaskMergeMode>::value || IsSameType<decltype(mode), const Log10SpecificMode *>::value,
        "mode type must be either MaskMergeMode or const Log10SpecificMode* ");
    constexpr Log10SpecificMode sprMode = Internal::GetLog10SpecificMode(mode);
    static_assert(SupportEnum<sprMode.mrgMode, MaskMergeMode::ZEROING>(),
        "current Log10 api only supports Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    constexpr float ln10Reciprocal = 0.43429448190325176;  // 1.0/Ln10;
    if constexpr (sprMode.algo == Log10Algo::PRECISION_1ULP_FTZ_FALSE) {
        if constexpr (SupportType<ActualT, half>()) {
            HalfUnion multiplyFactor;
            multiplyFactor.i = 0x6400; // 2^10
            HalfUnion subnormalThreshold;
            subnormalThreshold.i = 0x03FF;
            const half compensationFactor = -3.01029995663981;  // -Log10(2^10);
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            LogXImpl<sprMode.mrgMode>(dstRegCopy, srcRegCopy, mask, ln10Reciprocal);
            vadds(tmpReg, dstRegCopy, compensationFactor, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        } else {
            NotNumUnion multiplyFactor;
            multiplyFactor.i = 0x4B000000; // 2^23;
            NotNumUnion subnormalThreshold;
            subnormalThreshold.i = 0x007FFFFF;
            constexpr float compensationFactor = -6.923689900271567;  // -Log10(2^23);
            RegTensor<T> tmpReg;
            RegTensor<T> dstRegCopy;
            RegTensor<T> srcRegCopy = srcReg;
            MaskReg cmpMaskReg;

            vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
            vmuls(tmpReg, srcRegCopy, multiplyFactor.f, mask, modeValue);
            vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
            vln(dstRegCopy, srcRegCopy, mask, modeValue);
            vmuls(dstRegCopy, dstRegCopy, ln10Reciprocal, mask, modeValue);
            vadds(tmpReg, dstRegCopy, compensationFactor, mask, modeValue);
            vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
        }
    } else {
        if constexpr (SupportType<ActualT, half>()) {
            LogXImpl<mode>(dstReg, srcReg, mask, ln10Reciprocal);
        } else {
            vln(dstReg, srcReg, mask, modeValue);
            vmuls(dstReg, dstReg, ln10Reciprocal, mask, modeValue);
        }
    }
}

template <MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void NegB64Impl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(sizeof(ActualT) == 8, "RegT data type should be B64");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "RegT should be RegTraitNumTwo");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    RegT zeroReg;
    Duplicate(zeroReg, (int64_t)0, mask);
    Sub(dstReg, zeroReg, srcReg, mask);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void NegImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t, int64_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(
        SupportEnum<mode, MaskMergeMode::ZEROING>(), "current Neg api only supported Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (sizeof(ActualT) != 8) {
        vneg(dstReg, srcReg, mask, modeValue);
    } else {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            B64TraitOneToTaitTwo(traitTwoSrcReg, srcReg);
            NegB64Impl<mode>(traitTwoDstReg, traitTwoSrcReg, maskTrait2);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            NegB64Impl<mode>(dstReg, srcReg, mask);
        }
    }
}

template <MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void NotB64Impl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(sizeof(ActualT) == 8, "NotB64Impl data type should be B64");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "NotB64Impl RegT should be RegTraitNumTwo");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vnot((RegTensor<uint32_t> &)dstReg.reg[0], (RegTensor<uint32_t> &)srcReg.reg[0], mask, modeValue);
    vnot((RegTensor<uint32_t> &)dstReg.reg[1], (RegTensor<uint32_t> &)srcReg.reg[1], mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void NotImpl(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(Std::is_same_v<T, DefaultType> || Std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, uint64_t, int64_t>(),
        "current data type is not supported on current device!");
    static_assert(
        SupportEnum<mode, MaskMergeMode::ZEROING>(), "current Not api only supported Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (sizeof(ActualT) != 8) {
        vnot(dstReg, srcReg, mask, modeValue);
    } else {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            B64TraitOneToTaitTwo(traitTwoSrcReg, srcReg);
            NotB64Impl<mode>(traitTwoDstReg, traitTwoSrcReg, maskTrait2);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            RegT dstTemp;
            NotB64Impl<mode>(dstTemp, srcReg, mask);
            dstReg = dstTemp;
        }
    }
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H