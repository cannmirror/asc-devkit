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

/*!
 * \file kernel_operator_vec_compare_continuous_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H

#include "kernel_utils.h"

namespace AscendC {

template <typename T = MicroAPI::DefaultType, CMPMODE mode = CMPMODE::EQ, typename RegT>
__simd_callee__ inline void CompareDoubleImpl(MicroAPI::MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MicroAPI::MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(SupportType<ActualT, double, uint64_t>(), "CompareDoubleImpl only support double and uint64_t type");
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

template <typename T, CMPMODE mode, typename U>
__simd_callee__ inline void CompareDouble(MicroAPI::MaskReg &dstMask, U &srcReg0, U &srcReg1, MicroAPI::MaskReg &mask)
{
    CompareDoubleImpl<T, mode, U>(dstMask, srcReg0, srcReg1, mask);
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
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
            if constexpr (Std::is_same_v<T, double>) {
               CompareDouble<double, cmpMode>(dstReg, src0Reg, src1Reg, mask);     
            } else {
                MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
            }
            MicroAPI::DataCopyUnAlign(dstT, dstReg, uReg);
        }
        MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
    } else {
        MicroAPI::RegTensor<T> src0Reg, src1Reg;
        uint32_t offset = GetVecLen() / sizeof(T) / 8;
        __ubuf__ T *dstT = reinterpret_cast<__ubuf__ T*>(dst);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
            MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::DataCopy(dst + i * offset, dstReg);
            } else {
                MicroAPI::DataCopyUnAlign(dstT, dstReg, uReg);
            }
        }
        if constexpr (sizeof(T) > 1) {
            MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
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
