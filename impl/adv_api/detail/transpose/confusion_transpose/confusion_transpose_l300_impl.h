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
 * \file confusion_transpose_l300_impl.h
 * \brief
 */
#ifndef IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_L300_IMPL_H
#define IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_L300_IMPL_H

namespace AscendC {
template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ConfusionTransposeCommonGatherVF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2, uint32_t tail,
    uint32_t count, uint16_t mainLoop, uint32_t dtypeSize, uint32_t tailLoop)
{
    MicroAPI::RegTensor<U, Trait> indexReg;
    MicroAPI::RegTensor<T, Trait> MainVreg;
    MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg mainMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<T, Trait>(count);
    MicroAPI::UnalignReg ureg0;
    Arange((MicroAPI::RegTensor<X, Trait> &)indexReg, static_cast<X>(0));
    Muls(indexReg, indexReg, static_cast<U>(srcStride2), indexFullMask);
    for (uint16_t i = 0; i < static_cast<uint16_t>(forLoop0); i++) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(forLoop1); j++) {
            uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)((i * forLoop1 + j) * forLoop2 * dtypeSize);
            for (uint16_t k = 0; k < static_cast<uint16_t>(mainLoop); k++) {
                DataCopyGather(MainVreg, srcAddr + i * srcStride0 + j * srcStride1 + k * vlSize * srcStride2,
                    indexReg, mainMask);
                MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), MainVreg, ureg0, vlSize);
            }
            for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
                DataCopyGather(MainVreg, srcAddr + i * srcStride0 + j * srcStride1 + mainLoop * vlSize * srcStride2,
                    indexReg, tailMask);
                MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), MainVreg, ureg0, tail);
            }
            MicroAPI::DataCopyUnAlignPost(((__ubuf__ T *&)hoistDstAddr), ureg0, 0);
        }
    }
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__aicore__ inline void ConfusionTransposeCommonGather(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t tail = forLoop2 % vlSize;
    uint32_t count = tail;
    uint16_t mainLoop = forLoop2 / vlSize;
    uint32_t dtypeSize = sizeof(T);
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    ConfusionTransposeCommonGatherVF<T, X, U, Trait, vlSize>(dstAddr, srcAddr, forLoop0,
        forLoop1, forLoop2, srcStride0, srcStride1, srcStride2, tail, count, mainLoop, dtypeSize, tailLoop);
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ConfusionTransposeCommonGatherB8VF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2, uint32_t tail,
    uint32_t count, uint16_t mainLoop, uint32_t dtypeSize, uint32_t tailLoop)
{
    MicroAPI::RegTensor<U, Trait> indexReg;
    MicroAPI::RegTensor<T, Trait> vreg0;
    MicroAPI::RegTensor<T, Trait> vreg1;
    MicroAPI::RegTensor<T, Trait> vreg2;
    MicroAPI::RegTensor<T, Trait> vreg3;
    MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg mainMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg tailMask0 = MicroAPI::UpdateMask<U, Trait>(count);
    MicroAPI::MaskReg tailMask1 = MicroAPI::UpdateMask<U, Trait>(count);
    MicroAPI::UnalignReg ureg0;
    Arange((MicroAPI::RegTensor<X, Trait> &)indexReg, static_cast<X>(0));
    Muls(indexReg, indexReg, static_cast<U>(srcStride2), indexFullMask);
    for (uint16_t i = 0; i < static_cast<uint16_t>(forLoop0); i++) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(forLoop1); j++) {
            uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)((i * forLoop1 + j) * forLoop2 * dtypeSize);
            for (uint16_t k = 0; k < static_cast<uint16_t>(mainLoop); k++) {
                DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg0,
                    srcAddr + i * srcStride0 + j * srcStride1 + k * vlSize * srcStride2, indexReg, mainMask);
                DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg1,
                    srcAddr + i * srcStride0 + j * srcStride1 + k * vlSize * srcStride2 + vlSize / 2 * srcStride2,
                    indexReg, mainMask);
                DeInterleave(vreg2, vreg3, vreg0, vreg1);
                MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), vreg2, ureg0, vlSize);
            }
            for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
                DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg0,
                    srcAddr + i * srcStride0 + j * srcStride1 + mainLoop * vlSize * srcStride2, indexReg,
                    tailMask0);
                DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg1,
                    srcAddr + i * srcStride0 + j * srcStride1 + mainLoop * vlSize * srcStride2 +
                    vlSize / 2 * srcStride2,
                    indexReg, tailMask1);
                DeInterleave(vreg2, vreg3, vreg0, vreg1);
                MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), vreg2, ureg0, tail);
            }
            MicroAPI::DataCopyUnAlignPost(((__ubuf__ T *&)hoistDstAddr), ureg0, 0);
        }
    }
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__aicore__ inline void ConfusionTransposeCommonGatherB8(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t tail = forLoop2 % vlSize;
    uint32_t count = tail;
    uint16_t mainLoop = forLoop2 / vlSize;
    uint32_t dtypeSize = sizeof(T);
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    ConfusionTransposeCommonGatherB8VF<T, X, U, Trait, vlSize>(dstAddr, srcAddr, forLoop0,
        forLoop1, forLoop2, srcStride0, srcStride1, srcStride2, tail, count, mainLoop, dtypeSize, tailLoop);
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ConfusionTransposeComplexGatherVF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2, uint32_t factor,
    uint32_t mainSize, uint16_t mainLoop, uint32_t tail, uint32_t tailLoop, uint32_t mainCount, uint32_t tailCount,
    uint32_t count, uint32_t dtypeSize)
{
    MicroAPI::RegTensor<U, Trait> indexReg0;
    MicroAPI::RegTensor<U, Trait> indexReg1;
    MicroAPI::RegTensor<U, Trait> indexReg2;
    MicroAPI::RegTensor<U, Trait> indexReg3;
    MicroAPI::RegTensor<U, Trait> indexReg4;
    MicroAPI::RegTensor<U, Trait> indexReg5;
    MicroAPI::RegTensor<T, Trait> MainVreg;
    MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg mainMask = MicroAPI::UpdateMask<T, Trait>(mainCount);
    MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<T, Trait>(tailCount);
    MicroAPI::UnalignReg ureg0;
    Arange((MicroAPI::RegTensor<X, Trait> &)indexReg0, static_cast<X>(0));
    Duplicate(indexReg1, static_cast<U>(forLoop2));
    Div(indexReg2, indexReg0, indexReg1, indexFullMask);
    Muls(indexReg3, indexReg2, static_cast<U>(srcStride1), indexFullMask);
    // k%c: k - k/c*c
    Mul(indexReg4, indexReg2, indexReg1, indexFullMask);
    Sub(indexReg4, indexReg0, indexReg4, indexFullMask);
    Muls(indexReg4, indexReg4, static_cast<U>(srcStride2), indexFullMask);
    Add(indexReg5, indexReg3, indexReg4, indexFullMask);
    for (uint16_t i = 0; i < static_cast<uint16_t>(forLoop0); i++) {
        uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)(i * forLoop1 * forLoop2 * dtypeSize);
        for (uint16_t j = 0; j < static_cast<uint16_t>(mainLoop); j++) {
            DataCopyGather(MainVreg, srcAddr + i * srcStride0 + j * srcStride1 * factor, indexReg5, mainMask);
            MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), MainVreg, ureg0, mainSize);
        }
        for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
            DataCopyGather(MainVreg, srcAddr + i * srcStride0 + mainLoop * srcStride1 * factor, indexReg5,
                tailMask);
            MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), MainVreg, ureg0, tail);
        }
        MicroAPI::DataCopyUnAlignPost(((__ubuf__ T *&)hoistDstAddr), ureg0, 0);
    }
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__aicore__ inline void ConfusionTransposeComplexGather(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t factor = vlSize / forLoop2;
    uint32_t mainSize = factor * forLoop2;
    uint16_t mainLoop = forLoop1 / factor;
    uint32_t tail = forLoop1 % factor * forLoop2;
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    uint32_t mainCount = mainSize;
    uint32_t tailCount = tail;

    uint32_t count = tail;
    uint32_t dtypeSize = sizeof(T);
    ConfusionTransposeComplexGatherVF<T, X, U, Trait, vlSize>(dstAddr, srcAddr, forLoop0, forLoop1, forLoop2,
        srcStride0, srcStride1, srcStride2, factor, mainSize, mainLoop, tail, tailLoop, mainCount, tailCount,
        count, dtypeSize);
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ConfusionTransposeComplexGatherB8VF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2, uint32_t factor,
    uint32_t mainSize, uint16_t mainLoop, uint32_t tail, uint32_t tailLoop, uint32_t mainCount, uint32_t tailCount,
    uint32_t dtypeSize, uint32_t halfVlSize)
{
    MicroAPI::RegTensor<U, Trait> indexReg0;
    MicroAPI::RegTensor<U, Trait> indexReg1;
    MicroAPI::RegTensor<U, Trait> indexReg2;
    MicroAPI::RegTensor<U, Trait> indexReg3;
    MicroAPI::RegTensor<U, Trait> indexReg4;
    MicroAPI::RegTensor<U, Trait> indexReg5;
    MicroAPI::RegTensor<U, Trait> indexReg6;
    MicroAPI::RegTensor<U, Trait> indexReg7;
    MicroAPI::RegTensor<U, Trait> indexReg8;
    MicroAPI::RegTensor<U, Trait> indexReg9;
    MicroAPI::RegTensor<U, Trait> indexReg10;
    MicroAPI::RegTensor<T, Trait> vreg0;
    MicroAPI::RegTensor<T, Trait> vreg1;
    MicroAPI::RegTensor<T, Trait> vreg2;
    MicroAPI::RegTensor<T, Trait> vreg3;
    MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg mainMask0 = MicroAPI::UpdateMask<U, Trait>(mainCount);
    MicroAPI::MaskReg mainMask1 = MicroAPI::UpdateMask<U, Trait>(mainCount);
    MicroAPI::MaskReg tailMask0 = MicroAPI::UpdateMask<U, Trait>(tailCount);
    MicroAPI::MaskReg tailMask1 = MicroAPI::UpdateMask<U, Trait>(tailCount);
    MicroAPI::UnalignReg ureg0;
    Arange((MicroAPI::RegTensor<X, Trait> &)indexReg0, static_cast<X>(0));
    Duplicate(indexReg1, static_cast<U>(forLoop2));
    Div(indexReg2, indexReg0, indexReg1, indexFullMask);
    Muls(indexReg3, indexReg2, static_cast<U>(srcStride1), indexFullMask);
    // k%c: k - k/c*c
    Mul(indexReg4, indexReg2, indexReg1, indexFullMask);
    Sub(indexReg4, indexReg0, indexReg4, indexFullMask);
    Muls(indexReg4, indexReg4, static_cast<U>(srcStride2), indexFullMask);
    Add(indexReg5, indexReg3, indexReg4, indexFullMask);

    // the other half vl index
    Arange((MicroAPI::RegTensor<X, Trait> &)indexReg6, static_cast<X>(halfVlSize));
    Div(indexReg7, indexReg6, indexReg1, indexFullMask);
    Muls(indexReg8, indexReg7, static_cast<U>(srcStride1), indexFullMask);
    // k%c: k - k/c*c
    Mul(indexReg9, indexReg7, indexReg1, indexFullMask);
    Sub(indexReg9, indexReg6, indexReg9, indexFullMask);
    Muls(indexReg9, indexReg9, static_cast<U>(srcStride2), indexFullMask);
    Add(indexReg10, indexReg8, indexReg9, indexFullMask);
    for (uint16_t i = 0; i < static_cast<uint16_t>(forLoop0); i++) {
        uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)(i * forLoop1 * forLoop2 * dtypeSize);
        for (uint16_t j = 0; j < static_cast<uint16_t>(mainLoop); j++) {
            DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg0,
                srcAddr + i * srcStride0 + j * srcStride1 * factor, indexReg5, mainMask0);
            DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg1,
                srcAddr + i * srcStride0 + j * srcStride1 * factor, indexReg10, mainMask1);
            DeInterleave(vreg2, vreg3, vreg0, vreg1);
            MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), vreg2, ureg0, mainSize);
        }
        for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
            DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg0,
                srcAddr + i * srcStride0 + mainLoop * srcStride1 * factor, indexReg5, tailMask0);
            DataCopyGather((MicroAPI::RegTensor<U, Trait> &)vreg1,
                srcAddr + i * srcStride0 + mainLoop * srcStride1 * factor, indexReg10, tailMask1);
            DeInterleave(vreg2, vreg3, vreg0, vreg1);
            MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), vreg2, ureg0, tail);
        }
        MicroAPI::DataCopyUnAlignPost(((__ubuf__ T *&)hoistDstAddr), ureg0, 0);
    }
}

template <typename T, typename X, typename U, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__aicore__ inline void ConfusionTransposeComplexGatherB8(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t factor = vlSize / forLoop2;
    uint32_t mainSize = factor * forLoop2;
    uint16_t mainLoop = forLoop1 / factor;
    uint32_t tail = forLoop1 % factor * forLoop2;
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    uint32_t mainCount = mainSize;
    uint32_t tailCount = tail;
    uint32_t dtypeSize = sizeof(T);
    uint32_t halfVlSize = vlSize / 2;

    ConfusionTransposeComplexGatherB8VF<T, X, U, Trait, vlSize>(dstAddr, srcAddr, forLoop0, forLoop1, forLoop2,
        srcStride0, srcStride1, srcStride2, factor, mainSize, mainLoop, tail, tailLoop, mainCount, tailCount,
        dtypeSize, halfVlSize);
}

template <typename T, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ConfusionTransposeCopySrcToDst(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t totalCount)
{
    uint32_t count = totalCount;
    uint32_t forLoop = (totalCount + vlSize - 1) / vlSize;
    MicroAPI::RegTensor<T, Trait> vreg0;
    MicroAPI::MaskReg mainMask;
    for (uint16_t i = 0; i < static_cast<uint16_t>(forLoop); i++) {
        mainMask = MicroAPI::UpdateMask<T, Trait>(count);
        DataCopy(vreg0, srcAddr + i * vlSize);
        DataCopy(dstAddr + i * vlSize, vreg0, mainMask);
    }
}

template <typename T, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ConfusionTransposeCommonDataCopyVF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1, uint32_t tail, uint32_t count,
    uint16_t mainLoop, uint32_t dtypeSize, uint32_t tailLoop)
{
    MicroAPI::RegTensor<T, Trait> MainVreg;
    MicroAPI::UnalignReg DstUreg;
    MicroAPI::UnalignReg SrcUreg;
    for (uint16_t i = 0; i < static_cast<uint16_t>(forLoop0); i++) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(forLoop1); j++) {
            uint64_t hoistSrcAddr = (uint64_t)srcAddr + (uint64_t)((i * srcStride0 + j * srcStride1) * dtypeSize);
            uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)((i * forLoop1 + j) * forLoop2 * dtypeSize);
            for (uint16_t k = 0; k < static_cast<uint16_t>(mainLoop); k++) {
                MicroAPI::DataCopyUnAlignPre(SrcUreg, ((__ubuf__ T *&)hoistSrcAddr));
                MicroAPI::DataCopyUnAlign(MainVreg, SrcUreg, ((__ubuf__ T *&)hoistSrcAddr), vlSize);
                MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), MainVreg, DstUreg, vlSize);
            }
            for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
                MicroAPI::DataCopyUnAlignPre(SrcUreg, ((__ubuf__ T *&)hoistSrcAddr));
                MicroAPI::DataCopyUnAlign(MainVreg, SrcUreg, ((__ubuf__ T *&)hoistSrcAddr), tail);
                MicroAPI::DataCopyUnAlign(((__ubuf__ T *&)hoistDstAddr), MainVreg, DstUreg, tail);
            }
            MicroAPI::DataCopyUnAlignPost(((__ubuf__ T *&)hoistDstAddr), DstUreg, 0);
        }
    }
}

template <typename T, const MicroAPI::RegTrait &Trait, const uint16_t vlSize>
__aicore__ inline void ConfusionTransposeCommonDataCopy(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t forLoop0,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride0, uint32_t srcStride1)
{
    uint32_t tail = forLoop2 % vlSize;
    uint32_t count = tail;
    uint16_t mainLoop = forLoop2 / vlSize;
    uint32_t dtypeSize = sizeof(T);
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    ConfusionTransposeCommonDataCopyVF<T, Trait, vlSize>(dstAddr, srcAddr, forLoop0,
        forLoop1, forLoop2, srcStride0, srcStride1, tail, count, mainLoop, dtypeSize, tailLoop);
}

/*
scene 13：{ shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"};
          { shape:[N, H, W], format:"ND"} -->{ shape:[N, W, H], format:"ND"};
*/
template <typename T>
__aicore__ inline void ConfusionTranspose021(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, ConfusionTranspose021Tiling& tiling)
{
    constexpr uint16_t vlSize =
        IsSameType<T, int64_t>::value ? GetVecLen() / sizeof(float) : GetVecLen() / sizeof(T);
    if ((tiling.dim1 == 1) || (tiling.dim2 == 1)) {
        uint32_t totalCount = tiling.dim0 * tiling.dim1 * tiling.dim2;
        ConfusionTransposeCopySrcToDst<T, MicroAPI::RegTraitNumOne, vlSize>((__ubuf__ T *)dstTensor.GetPhyAddr(),
            (__ubuf__ T *)srcTensor.GetPhyAddr(), totalCount);
    } else {
        uint32_t srcStride0 = tiling.dim1 * tiling.dim2;
        uint32_t srcStride1 = 1;
        uint32_t srcStride2 = tiling.dim2;
        if (tiling.dim1 > vlSize / 2) {
            if constexpr (SupportBytes<T, 4>()) {
                ConfusionTransposeCommonGather<T, int32_t, uint32_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim0,
                    tiling.dim2, tiling.dim1, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 2>()) {
                ConfusionTransposeCommonGather<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim0,
                    tiling.dim2, tiling.dim1, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 1>()) {
                ConfusionTransposeCommonGatherB8<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim0,
                    tiling.dim2, tiling.dim1, srcStride0, srcStride1, srcStride2);
            }
        } else {
            if constexpr (SupportBytes<T, 4>()) {
                ConfusionTransposeComplexGather<T, int32_t, uint32_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim0,
                    tiling.dim2, tiling.dim1, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 2>()) {
                ConfusionTransposeComplexGather<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim0,
                    tiling.dim2, tiling.dim1, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 1>()) {
                ConfusionTransposeComplexGatherB8<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim0,
                    tiling.dim2, tiling.dim1, srcStride0, srcStride1, srcStride2);
            }
        }
    }
}

/*
scene 14： { shape:[N, H, W], format:"ND"} -->{ shape:[H, N, W], format:"ND"};
*/
template <typename T>
__aicore__ inline void ConfusionTranspose102(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, ConfusionTranspose102Tiling& tiling)
{
    constexpr uint16_t vlSize =
        IsSameType<T, int64_t>::value ? GetVecLen() / sizeof(float) : GetVecLen() / sizeof(T);
    if ((tiling.dim0 == 1) || (tiling.dim1 == 1)) {
        uint32_t totalCount = tiling.dim0 * tiling.dim1 * tiling.dim2;
        ConfusionTransposeCopySrcToDst<T, MicroAPI::RegTraitNumOne, vlSize>((__ubuf__ T *)dstTensor.GetPhyAddr(),
            (__ubuf__ T *)srcTensor.GetPhyAddr(), totalCount);
    } else {
        uint32_t srcStride0 = tiling.dim2;
        uint32_t srcStride1 = tiling.dim1 * tiling.dim2;
        uint32_t srcStride2 = 1;
        if (tiling.dim2 > vlSize / 2) {
            ConfusionTransposeCommonDataCopy<T, MicroAPI::RegTraitNumOne, vlSize>((__ubuf__ T *)dstTensor.GetPhyAddr(),
                (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim1, tiling.dim0, tiling.dim2, srcStride0, srcStride1);
        } else {
            if constexpr (SupportBytes<T, 4>()) {
                ConfusionTransposeComplexGather<T, int32_t, uint32_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim1,
                    tiling.dim0, tiling.dim2, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 2>()) {
                ConfusionTransposeComplexGather<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim1,
                    tiling.dim0, tiling.dim2, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 1>()) {
                ConfusionTransposeComplexGatherB8<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim1,
                    tiling.dim0, tiling.dim2, srcStride0, srcStride1, srcStride2);
            }
        }
    }
}

/*
scene 15： { shape:[N, H, W], format:"ND"} -->{ shape:[W, H, N], format:"ND"};
*/
template <typename T>
__aicore__ inline void ConfusionTranspose210(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, ConfusionTranspose210Tiling& tiling)
{
    constexpr uint16_t vlSize =
        IsSameType<T, int64_t>::value ? GetVecLen() / sizeof(float) : GetVecLen() / sizeof(T);
    ConfusionTranspose021Tiling tiling021 = { 1, tiling.dim1, tiling.dim2 };
    if (tiling.dim0 == 1 || tiling.dim1 == 1 || tiling.dim2 == 1) {
        ConfusionTranspose021Tiling tiling021 = { 1, tiling.dim1, tiling.dim2 };
        if (tiling.dim1 == 1) {
            tiling021.dim1 = tiling.dim0;
        } else if (tiling.dim2 == 1) {
            tiling021.dim1 = tiling.dim0;
            tiling021.dim2 = tiling.dim1;
        }
        ConfusionTranspose021(dstTensor, srcTensor, tiling021);
    } else {
        uint32_t srcStride0 = 1;
        uint32_t srcStride1 = tiling.dim2;
        uint32_t srcStride2 = tiling.dim1 * tiling.dim2;
        if (tiling.dim0 > vlSize / 2) {
            if constexpr (SupportBytes<T, 4>()) {
                ConfusionTransposeCommonGather<T, int32_t, uint32_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim2,
                    tiling.dim1, tiling.dim0, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 2>()) {
                ConfusionTransposeCommonGather<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim2,
                    tiling.dim1, tiling.dim0, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 1>()) {
                ConfusionTransposeCommonGatherB8<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim2,
                    tiling.dim1, tiling.dim0, srcStride0, srcStride1, srcStride2);
            }
        } else {
            if constexpr (SupportBytes<T, 4>()) {
                ConfusionTransposeComplexGather<T, int32_t, uint32_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim2,
                    tiling.dim1, tiling.dim0, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 2>()) {
                ConfusionTransposeComplexGather<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim2,
                    tiling.dim1, tiling.dim0, srcStride0, srcStride1, srcStride2);
            } else if constexpr (SupportBytes<T, 1>()) {
                ConfusionTransposeComplexGatherB8<T, int16_t, uint16_t, MicroAPI::RegTraitNumOne, vlSize>(
                    (__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), tiling.dim2,
                    tiling.dim1, tiling.dim0, srcStride0, srcStride1, srcStride2);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionTransposeND2NZWithInlv(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, ConfusionTranspose210Tiling& tiling)
{
    ConfusionTransposeND2NZWithInlvImpl((__ubuf__ T*) dstTensor.GetPhyAddr(), (__ubuf__ T*) srcTensor.GetPhyAddr(), tiling);
}

template <typename T>
__simd_vf__ inline void ConfusionTransposeND2NZWithInlvVFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t regWidth,
    uint32_t c0Size, uint32_t factor, uint32_t height, uint32_t width, uint32_t tailWidth, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> vSrcReg0;
    AscendC::MicroAPI::RegTensor<T> vSrcReg1;
    AscendC::MicroAPI::RegTensor<T> vDstReg0;
    AscendC::MicroAPI::RegTensor<T> vDstReg1;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t srcOffset = 0;
    uint32_t dstOffset = 0;
    const uint16_t heightLoopNum = height / factor;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        srcOffset = i * regWidth;
        dstOffset = i * regWidth * heightLoopNum * factor;
        for (uint16_t j = 0; j < heightLoopNum; j++) {
            uint64_t rowStrideAddr = uint64_t(srcAddr + srcOffset + factor * j * width);
            uint64_t nerborStrideAddr = uint64_t(srcAddr + srcOffset + (factor * j + 1) * width);

            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                vSrcReg0, (__ubuf__ T*&)rowStrideAddr, 1, 0, mask);
            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                vSrcReg1, (__ubuf__ T*&)nerborStrideAddr, 1, 0, mask);
            AscendC::MicroAPI::Interleave<T>(vDstReg0, vDstReg1, vSrcReg0, vSrcReg1);

            uint64_t dstRowStrideAddr = uint64_t(dstAddr + dstOffset + j * c0Size);
            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                (__ubuf__ T*&)dstRowStrideAddr, vDstReg0, heightLoopNum, 0, mask);

            uint64_t dstNerborStrideAddr = uint64_t(dstAddr + dstOffset + regWidth * heightLoopNum + j * c0Size);
            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                (__ubuf__ T*&)dstNerborStrideAddr, vDstReg1, heightLoopNum, 0, mask);
        }
    }

    if (tailWidth) {
        if (repeatTimes != 0) {
            srcOffset += regWidth;
            dstOffset += regWidth * heightLoopNum * factor;
        }
        const uint32_t regWidthHalf = 128;
        for (uint16_t j = 0; j < heightLoopNum; j++) {
            uint64_t rowStrideAddr = uint64_t(srcAddr + srcOffset + factor * j * width);
            uint64_t nerborStrideAddr = uint64_t(srcAddr + srcOffset + (factor * j + 1) * width);
            uint32_t mask_size = tailWidth * factor;
            if (tailWidth > regWidthHalf) {
                mask_size = regWidth;
            }
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<T>(mask_size);
            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                vSrcReg0, (__ubuf__ T*&)rowStrideAddr, 1, 0, mask);
            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                vSrcReg1, (__ubuf__ T*&)nerborStrideAddr, 1, 0, mask);
            AscendC::MicroAPI::Interleave<T>(vDstReg0, vDstReg1, vSrcReg0, vSrcReg1);

            uint64_t dstRowStrideAddr = uint64_t(dstAddr + dstOffset + j * c0Size);
            AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                (__ubuf__ T*&)dstRowStrideAddr, vDstReg0, heightLoopNum, 0, mask);

            if (tailWidth > regWidthHalf) {
                mask_size = tailWidth * factor - regWidth;
                AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<T>(mask_size);
                uint64_t dstNerborStrideAddr = uint64_t(dstAddr + dstOffset + regWidth * heightLoopNum + j * c0Size);
                AscendC::MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>(
                    (__ubuf__ T*&)dstNerborStrideAddr, vDstReg1, heightLoopNum, 0, mask1);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionTransposeND2NZWithInlvImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, ConfusionTranspose210Tiling& tiling)
{
    constexpr uint32_t regWidth = 256; // vf reg length
    constexpr uint32_t c0Size = 32; // factal size
    constexpr uint32_t factor = 2; // row factor
    uint32_t height = tiling.dim1;
    uint32_t width = tiling.dim2;
    uint32_t tailWidth = width % regWidth;
    uint16_t repeatTimes = width / regWidth;
    ConfusionTransposeND2NZWithInlvVFImpl<T>(dstAddr, srcAddr, regWidth, c0Size, factor, height, width, tailWidth, repeatTimes);
}
} // namespace AscendC
#endif // IMPL_TRANSPOSE_CONFUSION_TRANSPOSE_CONFUSION_TRANSPOSE_C310_IMPL_H