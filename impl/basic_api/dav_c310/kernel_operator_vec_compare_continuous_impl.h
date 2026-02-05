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
 * \file kernel_operator_vec_compare_continuous_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H

#include "kernel_utils.h"
#include "micro_api/kernel_micro_intf.h"

namespace AscendC {

template <typename T = MicroAPI::DefaultType, typename RegT>
__simd_callee__ inline void CompareEqualDoubleImpl(MicroAPI::MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MicroAPI::MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(SupportType<ActualT, double, uint64_t>(), "CompareEqualDoubleImpl only support double and uint64_t type");
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> tmpSrcReg0 = (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)srcReg0;
	MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> tmpSrcReg1 = (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)srcReg1;
	MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> exponent0;
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> exponent1;
    MicroAPI::ShiftRights(exponent0, tmpSrcReg0, static_cast<int16_t>(52), mask);
    MicroAPI::ShiftRights(exponent1, tmpSrcReg1, static_cast<int16_t>(52), mask);
	MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> scalarExponent;
    MicroAPI::Duplicate(scalarExponent, static_cast<uint64_t>(0x7ff), mask);
    MicroAPI::And(exponent0, exponent0, scalarExponent, mask);
    MicroAPI::And(exponent1, exponent1, scalarExponent, mask);
	MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> mantissa0, mantissa1;
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> scalarMantissa;
    MicroAPI::Duplicate(scalarMantissa, static_cast<uint64_t>(0xfffffffffffff), mask);
    MicroAPI::And(mantissa0, tmpSrcReg0, scalarMantissa, mask);
    MicroAPI::And(mantissa1, tmpSrcReg1, scalarMantissa, mask);
    MicroAPI::MaskReg tmpMask0, tmpMask1;
	MicroAPI::CompareScalar(tmpMask0, exponent0, 0x7ff, mask);
    MicroAPI::CompareScalar(dstMask, exponent1, 0x7ff, tmpMask0);
    MicroAPI::MaskNot(dstMask, dstMask, mask);
    MicroAPI::CompareScalar<uint64_t, CMPMODE::EQ>(tmpMask1, mantissa0, 0, mask);
    MicroAPI::CompareScalar<uint64_t, CMPMODE::EQ>(tmpMask0, mantissa1, 0, tmpMask1);
    MicroAPI::MaskOr(tmpMask0, tmpMask0, dstMask, mask);
    MicroAPI::Compare(dstMask, tmpSrcReg0, tmpSrcReg1, mask);
    MicroAPI::MaskAnd(dstMask, dstMask, tmpMask0, mask);
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> unsignedPart0, unsignedPart1;
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> scalarUnsignedPart;
    MicroAPI::Duplicate(scalarUnsignedPart, static_cast<uint64_t>(0x7fffffffffffffff), mask);
    MicroAPI::And(unsignedPart0, tmpSrcReg0, scalarUnsignedPart, mask);
    MicroAPI::And(unsignedPart1, tmpSrcReg1, scalarUnsignedPart, mask);
    MicroAPI::CompareScalar<uint64_t, CMPMODE::EQ>(tmpMask0, unsignedPart0, 0, mask);
    MicroAPI::CompareScalar<uint64_t, CMPMODE::EQ>(tmpMask1, unsignedPart1, 0, tmpMask0);
    MicroAPI::MaskOr(dstMask, dstMask, tmpMask1, mask);
}

template <typename T = MicroAPI::DefaultType, typename U>
__simd_callee__ inline void CompareEqualDouble(MicroAPI::MaskReg &dstMask, U &srcReg0, U &srcReg1, MicroAPI::MaskReg &mask)
{
    CompareEqualDoubleImpl(dstMask, srcReg0, srcReg1, mask);
}

template <typename T = MicroAPI::DefaultType, typename U>
__simd_callee__ inline void IsNanFull(MicroAPI::MaskReg &dstMask, U &low, U &high, MicroAPI::RegTensor<uint32_t> &scalar0,
    MicroAPI::RegTensor<uint32_t> &scalar1, MicroAPI::RegTensor<uint32_t> &scalar2, MicroAPI::MaskReg &cmpMask,
    MicroAPI::MaskReg &cmpMask0, MicroAPI::MaskReg &cmpMask1, MicroAPI::MaskReg &mask)
{
	MicroAPI::RegTensor<uint32_t> tmpReg, resReg;

    // exp_and_mantissa_high = high & 0x7fffffff
    MicroAPI::And(tmpReg, high, scalar0, mask);
    // exponent = (exp_and_mantissa_high >> 20) & 0x7ff
    MicroAPI::ShiftRights(resReg, tmpReg, static_cast<int16_t>(20), mask);
    MicroAPI::And(resReg, resReg, scalar1, mask);

    // cmpMask = (exponent == 0x7ff)
    MicroAPI::Compare(cmpMask, resReg, scalar1, mask);
    // scenario that cmpMask = true
    // mantissa_high = exp_and_mantissa_high & 0xfffff
    MicroAPI::And(resReg, tmpReg, scalar2, cmpMask);
    // return (mantissa_high != 0) || (low != 0) 
    MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(cmpMask0, resReg, static_cast<uint32_t>(0), cmpMask);
    MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(cmpMask1, low, static_cast<uint32_t>(0), cmpMask);
    // scenario that cmpMask = false, return false
    // cmpMask0 || cmpMask1 -> dstMask && cmpMask
    MicroAPI::MaskOr(dstMask, cmpMask0, cmpMask1, cmpMask);
}

template <typename T = MicroAPI::DefaultType, typename U>
__simd_callee__ inline void IsZero(MicroAPI::MaskReg &dstMask, U &low, U &high, MicroAPI::RegTensor<uint32_t> &scalar0,
    MicroAPI::MaskReg &cmpMask, MicroAPI::MaskReg &mask)
{
	MicroAPI::RegTensor<uint32_t> tmpReg;

    // return (high & 0x7fffffff) == 0 && low == 0
    MicroAPI::And(tmpReg, high, scalar0, mask);
    MicroAPI::CompareScalar(cmpMask, tmpReg, static_cast<uint32_t>(0), mask);
    MicroAPI::CompareScalar(dstMask, low, static_cast<uint32_t>(0), cmpMask);
}

template <typename T = MicroAPI::DefaultType, CMPMODE cmpMode, typename U>
__simd_callee__ inline void CompareLessDouble(MicroAPI::MaskReg &dstMask, U &srcReg0, U &srcReg1, MicroAPI::RegTensor<uint32_t> &scalar0,
    MicroAPI::RegTensor<uint32_t> &scalar1, MicroAPI::RegTensor<uint32_t> &scalar2, MicroAPI::MaskReg &mask)
{
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> tmpSrcReg0 = (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)srcReg0;
	MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> tmpSrcReg1 = (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)srcReg1;
	MicroAPI::RegTensor<uint32_t> sign0, sign1, low0, low1, high0, high1;
    MicroAPI::MaskReg cmpMask, cmpMask0, cmpMask1, cmpMask2;

    // low = bits64 & 0xffffffff
    MicroAPI::Copy(low0, (MicroAPI::RegTensor<uint32_t> &)tmpSrcReg0.reg[0], mask);
    MicroAPI::Copy(low1, (MicroAPI::RegTensor<uint32_t> &)tmpSrcReg1.reg[0], mask);
    // high = (bits64 >> 32) &  0xffffffff
    MicroAPI::Copy(high0, (MicroAPI::RegTensor<uint32_t> &)tmpSrcReg0.reg[1], mask);
    MicroAPI::Copy(high1, (MicroAPI::RegTensor<uint32_t> &)tmpSrcReg1.reg[1], mask);

    // handle nan: any comparision (except for NE) with nan is false
    IsNanFull(cmpMask0, low0, high0, scalar0, scalar1, scalar2, cmpMask, cmpMask2, dstMask, mask);
    IsNanFull(cmpMask1, low1, high1, scalar0, scalar1, scalar2, cmpMask, cmpMask2, dstMask, mask);

    // if is_nan_full(low0, high0) || is_nan_full(low1, high1), return false
    MicroAPI::MaskOr(cmpMask, cmpMask0, cmpMask1, mask);
    // !cmpMask && mask -> dstMask
    MicroAPI::MaskNot(dstMask, cmpMask, mask);

    // handle zeros: +0 and -0 are equal
    IsZero(cmpMask0, low0, high0, scalar0, cmpMask2, mask);
    IsZero(cmpMask1, low1, high1, scalar0, cmpMask2, mask);
    // if is_zero(low0, high0) && is_zero(low1, high1), return false
    MicroAPI::MaskAnd(cmpMask2, cmpMask0, cmpMask1, mask);

    // handle non-zero and non-nan scenario
    // !cmpMask2 && dstMask -> mask
    MicroAPI::MaskNot(mask, cmpMask2, dstMask);
    if constexpr (cmpMode == CMPMODE::LT) {
        // !cmpMask2 && dstMask -> dstMask
        MicroAPI::MaskNot(dstMask, cmpMask2, dstMask);
    } else if constexpr (cmpMode == CMPMODE::LE) {
        MicroAPI::Compare<uint32_t, CMPMODE::NE>(cmpMask0, low0, low1, mask);
        MicroAPI::Compare<uint32_t, CMPMODE::NE>(cmpMask1, high0, high1, mask);
        // handle non-zero and non-nan and non-equal scenario
        // (cmpMask0 || cmpMask1) && mask -> mask
        MicroAPI::MaskOr(mask, cmpMask0, cmpMask1, mask);
    }

    // extract sign bits
    MicroAPI::ShiftRights(sign0, high0, static_cast<int16_t>(31), mask);
    MicroAPI::ShiftRights(sign1, high1, static_cast<int16_t>(31), mask);

    // negative (sign=1) < positive (sign=0)
    // if sign0 != sign1, return sign0 > sign1
    MicroAPI::Compare<uint32_t, CMPMODE::NE>(cmpMask, sign0, sign1, mask);
    MicroAPI::Compare<uint32_t, CMPMODE::GT>(cmpMask0, sign0, sign1, cmpMask);
    MicroAPI::MaskSel(dstMask, cmpMask0, dstMask, cmpMask);

    // if sign0 == sign1
    MicroAPI::MaskNot(mask, cmpMask, mask);
    /*
        if sign0 == 0:
            if high0 != high1:
                return high0 < high1
            return low0 < low1
        else:
            if high0 != high1:
                return high0 > high1
            return low0 > low1
    */
    MicroAPI::CompareScalar(cmpMask, sign0, static_cast<uint32_t>(0), mask);
    MicroAPI::Compare<uint32_t, CMPMODE::NE>(cmpMask0, high0, high1, cmpMask);
    MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpMask1, high0, high1, cmpMask0);
    MicroAPI::MaskSel(dstMask, cmpMask1, dstMask, cmpMask0);
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(cmpMask0, high0, high1, cmpMask);
    MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpMask1, low0, low1, cmpMask0);
    MicroAPI::MaskSel(dstMask, cmpMask1, dstMask, cmpMask0);

    MicroAPI::MaskNot(cmpMask, cmpMask, mask);
    MicroAPI::Compare<uint32_t, CMPMODE::NE>(cmpMask0, high0, high1, cmpMask);
    MicroAPI::Compare<uint32_t, CMPMODE::GT>(cmpMask1, high0, high1, cmpMask0);
    MicroAPI::MaskSel(dstMask, cmpMask1, dstMask, cmpMask0);
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(cmpMask0, high0, high1, cmpMask);
    MicroAPI::Compare<uint32_t, CMPMODE::GT>(cmpMask1, low0, low1, cmpMask0);
    MicroAPI::MaskSel(dstMask, cmpMask1, dstMask, cmpMask0);
}

template <typename T = MicroAPI::DefaultType, CMPMODE cmpMode, typename U>
__simd_callee__ inline void CompareDouble(MicroAPI::MaskReg &dstMask, U &src0Reg, U &src1Reg, MicroAPI::RegTensor<uint32_t> &scalar0,
    MicroAPI::RegTensor<uint32_t> &scalar1, MicroAPI::RegTensor<uint32_t> &scalar2, MicroAPI::MaskReg &mask)
{
    if constexpr (cmpMode == CMPMODE::LT) {
        CompareLessDouble<T, cmpMode>(dstMask, src0Reg, src1Reg, scalar0, scalar1, scalar2, mask);
    } else if constexpr (cmpMode == CMPMODE::GT) {
        CompareLessDouble<T, CMPMODE::LT>(dstMask, src1Reg, src0Reg, scalar0, scalar1, scalar2, mask);
    } else if constexpr (cmpMode == CMPMODE::EQ) {
        CompareEqualDouble(dstMask, src0Reg, src1Reg, mask);
    } else if constexpr (cmpMode == CMPMODE::LE) {
        CompareLessDouble<T, cmpMode>(dstMask, src0Reg, src1Reg, scalar0, scalar1, scalar2, mask);
    } else if constexpr (cmpMode == CMPMODE::GE) {
        CompareLessDouble<T, CMPMODE::LE>(dstMask, src1Reg, src0Reg, scalar0, scalar1, scalar2, mask);
    } else {
        CompareEqualDouble(dstMask, src0Reg, src1Reg, mask);
        MicroAPI::MaskNot(dstMask, dstMask, mask);
    }
}

// Compare::Level 2 - counter mode
template <typename T, typename U, CMPMODE cmpMode>
__simd_vf__ inline void CompareLevel2(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = calCount;
    MicroAPI::MaskReg dstReg, mask;
    MicroAPI::UnalignReg uReg;
    if constexpr (sizeof(T) == 8) {
        repeatElm = repeatElm * 2;
        repeatTime = CeilDivision(calCount, repeatElm);
        __ubuf__ uint32_t *dstT = reinterpret_cast<__ubuf__ uint32_t*>(dst);
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg, src1Reg;
	    MicroAPI::RegTensor<uint32_t> scalar0, scalar1, scalar2;
        MicroAPI::Duplicate(scalar0, static_cast<uint32_t>(0x7fffffff));
        MicroAPI::Duplicate(scalar1, static_cast<uint32_t>(0x7ff));
        MicroAPI::Duplicate(scalar2, static_cast<uint32_t>(0xfffff));
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
            if constexpr (Std::is_same_v<T, double>) {
                CompareDouble<double, cmpMode>(dstReg, src0Reg, src1Reg, scalar0, scalar1, scalar2, mask);
            } else {
                MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
            }
            MicroAPI::StoreUnAlign(dstT, dstReg, uReg);
        }
        MicroAPI::StoreUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
    } else {
        MicroAPI::RegTensor<T> src0Reg, src1Reg;
        uint32_t offset = GetVecLen() / sizeof(T) / 8;
        __ubuf__ T *dstT = reinterpret_cast<__ubuf__ T*>(dst);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
            MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::StoreAlign(dst + i * offset, dstReg);
            } else {
                MicroAPI::StoreUnAlign(dstT, dstReg, uReg);
            }
        }
        if constexpr (sizeof(T) > 1) {
            MicroAPI::StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, uint8_t, int8_t, bfloat16_t,
        uint64_t, int64_t, double>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, int8_t>(), "current data type is not supported!");
    switch (cmpMode) {
        case CMPMODE::LT: {
            CompareLevel2<T, U, CMPMODE::LT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            CompareLevel2<T, U, CMPMODE::GT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            CompareLevel2<T, U, CMPMODE::EQ>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            CompareLevel2<T, U, CMPMODE::LE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            CompareLevel2<T, U, CMPMODE::GE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            CompareLevel2<T, U, CMPMODE::NE>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}


} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H
