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
 * \file kernel_micro_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_CMPSEL_IMPL_H

namespace AscendC {
namespace MicroAPI {
template <CMPMODE mode = CMPMODE::EQ, typename RegT>
__simd_callee__ inline void CompareUint64Impl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(SupportType<ActualT, uint64_t>(), "CompareUint64Impl only support uint64_t type");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "CompareUint64Impl only support RegTraitNumTwo");
    MaskReg lowEq;
    MaskReg highEq;
    MaskReg lowCmp;
    MaskReg highCmp;
    if constexpr (mode == CMPMODE::EQ) {
        vcmp_eq(lowEq, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_eq(dstMask, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], lowEq);
    } else if constexpr (mode == CMPMODE::NE) {
        vcmp_ne(lowEq, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_ne(highEq, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        por(dstMask, lowEq, highEq, mask);
    } else if constexpr (mode == CMPMODE::GT) {
        vcmp_eq(highEq, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        vcmp_gt(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_gt(highCmp, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    } else if constexpr (mode == CMPMODE::GE) {
        vcmp_eq(highEq, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        vcmp_ge(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_ge(highCmp, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    } else if constexpr (mode == CMPMODE::LT) {
        vcmp_eq(highEq, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        vcmp_lt(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_lt(highCmp, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    } else if constexpr (mode == CMPMODE::LE) {
        vcmp_eq(highEq, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        vcmp_le(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_le(highCmp, (RegTensor<uint32_t> &)srcReg0.reg[1], (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    }
}

template <CMPMODE mode = CMPMODE::EQ, typename RegT>
__simd_callee__ inline void CompareInt64Impl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(SupportType<ActualT, int64_t>(), "CompareInt64Impl only support int64_t type");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "CompareInt64Impl only support RegTraitNumTwo");
    MaskReg lowEq;
    MaskReg highEq;
    MaskReg lowCmp;
    MaskReg highCmp;
    if constexpr (mode == CMPMODE::EQ) {
        vcmp_eq(lowEq, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_eq(dstMask, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], lowEq);
    } else if constexpr (mode == CMPMODE::NE) {
        vcmp_ne(lowEq, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_ne(highEq, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        por(dstMask, lowEq, highEq, mask);
    } else if constexpr (mode == CMPMODE::GT) {
        vcmp_eq(highEq, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        vcmp_gt(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_gt(highCmp, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    } else if constexpr (mode == CMPMODE::GE) {
        vcmp_eq(highEq, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        vcmp_ge(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_ge(highCmp, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    } else if constexpr (mode == CMPMODE::LT) {
        vcmp_eq(highEq, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        vcmp_lt(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_lt(highCmp, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    } else if constexpr (mode == CMPMODE::LE) {
        vcmp_eq(highEq, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        vcmp_le(lowCmp, (RegTensor<uint32_t> &)srcReg0.reg[0], (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
        vcmp_le(highCmp, (RegTensor<int32_t> &)srcReg0.reg[1], (RegTensor<int32_t> &)srcReg1.reg[1], mask);
        psel(dstMask, lowCmp, highCmp, highEq);
    }
}

template <CMPMODE mode = CMPMODE::EQ, typename RegT>
__simd_callee__ inline void CompareB64Impl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(SupportType<ActualT, uint64_t, int64_t>(),
        "CompareB64Impl only support uint64_t and int64_t type");
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "CompareB64Impl only support RegTraitNumTwo");
    if constexpr (std::is_same_v<ActualT, uint64_t>) {
        CompareUint64Impl<mode, RegT>(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (std::is_same_v<ActualT, int64_t>) {
        CompareInt64Impl<mode, RegT>(dstMask, srcReg0, srcReg1, mask);
    }
}

template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename RegT>
__simd_callee__ inline void CompareImpl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, bfloat16_t,
        uint64_t, int64_t>(),
        "current data type is not supported on current device!");
    if constexpr (sizeof(ActualT) < 8) {
        if constexpr (mode == CMPMODE::EQ) {
            vcmp_eq(dstMask, srcReg0, srcReg1, mask);
        } else if constexpr (mode == CMPMODE::NE) {
            vcmp_ne(dstMask, srcReg0, srcReg1, mask);
        } else if constexpr (mode == CMPMODE::GT) {
            vcmp_gt(dstMask, srcReg0, srcReg1, mask);
        } else if constexpr (mode == CMPMODE::GE) {
            vcmp_ge(dstMask, srcReg0, srcReg1, mask);
        } else if constexpr (mode == CMPMODE::LT) {
            vcmp_lt(dstMask, srcReg0, srcReg1, mask);
        } else if constexpr (mode == CMPMODE::LE) {
            vcmp_le(dstMask, srcReg0, srcReg1, mask);
        }
    } else if constexpr (sizeof(ActualT) == 8) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg0;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg1;
            B64TraitOneToTaitTwo(traitTwoSrcReg0, srcReg0);
            B64TraitOneToTaitTwo(traitTwoSrcReg1, srcReg1);
            CompareB64Impl<mode>(dstMask, traitTwoSrcReg0, traitTwoSrcReg1, maskTrait2);
            MaskUnPack(dstMask, dstMask);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            CompareB64Impl<mode>(dstMask, srcReg0, srcReg1, mask);
        }
    }
}

template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename RegT, typename ScalarT>
__simd_callee__ inline void CompareScalarImpl(MaskReg &dstMask, RegT &srcReg, ScalarT scalar, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, bfloat16_t, uint64_t, int64_t>(),
        "current data type is not supported on current device!");
    static_assert(Std::is_convertible<ScalarT, ActualT>(), "scalar data type could be converted to RegTensor data type");
    if constexpr (sizeof(ActualT) < 8) {
        if constexpr (mode == CMPMODE::EQ) {
            vcmps_eq(dstMask, srcReg, scalar, mask);
        } else if constexpr (mode == CMPMODE::NE) {
            vcmps_ne(dstMask, srcReg, scalar, mask);
        } else if constexpr (mode == CMPMODE::GT) {
            vcmps_gt(dstMask, srcReg, scalar, mask);
        } else if constexpr (mode == CMPMODE::GE) {
            vcmps_ge(dstMask, srcReg, scalar, mask);
        } else if constexpr (mode == CMPMODE::LT) {
            vcmps_lt(dstMask, srcReg, scalar, mask);
        } else if constexpr (mode == CMPMODE::LE) {
            vcmps_le(dstMask, srcReg, scalar, mask);
        }
    } else if constexpr (sizeof(ActualT) == 8) {
        RegT dupReg;
        Duplicate(dupReg, scalar, mask);
        Compare<T, mode, RegT>(dstMask, srcReg, dupReg, mask);
    }
}

template <typename T = DefaultType, typename RegT>
__simd_callee__ inline void SelectImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
        "Select only supports datatype bool/b8/b16/b32/b64 on current device");
    if constexpr (sizeof(ActualT) == 1) {
        vsel((RegTensor<uint8_t> &)dstReg, (RegTensor<uint8_t> &)srcReg0, (RegTensor<uint8_t> &)srcReg1, mask);
    } else if constexpr (sizeof(ActualT) == 2) {
        vsel((RegTensor<uint16_t> &)dstReg, (RegTensor<uint16_t> &)srcReg0, (RegTensor<uint16_t> &)srcReg1, mask);
    } else if constexpr (sizeof(ActualT) == 4) {
        vsel((RegTensor<uint32_t> &)dstReg, (RegTensor<uint32_t> &)srcReg0, (RegTensor<uint32_t> &)srcReg1, mask);
    } else if constexpr (sizeof(ActualT) == 8) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg0;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg1;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            B64TraitOneToTaitTwo(traitTwoSrcReg0, srcReg0);
            B64TraitOneToTaitTwo(traitTwoSrcReg1, srcReg1);
            vsel((RegTensor<uint32_t> &)traitTwoDstReg.reg[0], (RegTensor<uint32_t> &)traitTwoSrcReg0.reg[0],
                (RegTensor<uint32_t> &)traitTwoSrcReg1.reg[0], maskTrait2);
            vsel((RegTensor<uint32_t> &)traitTwoDstReg.reg[1], (RegTensor<uint32_t> &)traitTwoSrcReg0.reg[1],
                (RegTensor<uint32_t> &)traitTwoSrcReg1.reg[1], maskTrait2);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            vsel((RegTensor<uint32_t> &)dstReg.reg[0], (RegTensor<uint32_t> &)srcReg0.reg[0],
                (RegTensor<uint32_t> &)srcReg1.reg[0], mask);
            vsel((RegTensor<uint32_t> &)dstReg.reg[1], (RegTensor<uint32_t> &)srcReg0.reg[1],
                (RegTensor<uint32_t> &)srcReg1.reg[1], mask);
        }
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_CMPSEL_IMPL_H