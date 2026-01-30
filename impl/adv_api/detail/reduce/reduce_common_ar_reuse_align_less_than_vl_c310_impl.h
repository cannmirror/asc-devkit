/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef IMPL_REDUCE_REDUCE_COMMON_AR_REUSE_ALIGN_LESS_THAN_VL_C310_IMPL_H
#define IMPL_REDUCE_REDUCE_COMMON_AR_REUSE_ALIGN_LESS_THAN_VL_C310_IMPL_H

#include "kernel_basic_intf.h"
#include "kernel_tensor.h"
#include "reduce_common_util_impl.h"
#include "reduce_common_util_c310_impl.h"

namespace AscendC {
template <class T, class U, const MicroAPI::RegTrait &Trait, const MicroAPI::CastTrait &CastTraitUppper,
    const MicroAPI::CastTrait &CastTraitLower, const uint16_t vlSize, auto Binaryfunc, auto Reducefunc>
__simd_vf__ inline void ReduceARCastLessThanVL(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t dimA, uint32_t dimR)
{
    if (dimR <= (vlSize / 2)) {
        MicroAPI::RegTensor<T, Trait> vreg0;
        MicroAPI::RegTensor<T, Trait> vreg1;
        MicroAPI::RegTensor<U, Trait> vreg0CastUpper;
        MicroAPI::RegTensor<U, Trait> vreg1CastUpper;
        MicroAPI::UnalignReg uDst;
        uint32_t sreg1 = dimR;
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<U>(sreg1);
        for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_US_B16>(vreg0, srcAddr + loopA * dimR);
            } else {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_US_B8>(vreg0, srcAddr + loopA * dimR);
            }
            MicroAPI::Cast<U, T, CastTraitUppper>(vreg0CastUpper, vreg0, mask);
            Reducefunc(vreg1CastUpper, vreg0CastUpper, mask);
            MicroAPI::Cast<T, U, CastTraitLower>(vreg1, vreg1CastUpper, mask);
            MicroAPI::StoreUnAlign((__ubuf__ T *&)dstAddr, vreg1, uDst, 1);
        }
        MicroAPI::StoreUnAlignPost((__ubuf__ T *&)dstAddr, uDst, 0);
    } else {
        MicroAPI::RegTensor<T, Trait> vreg0;
        MicroAPI::RegTensor<T, Trait> vreg1;
        MicroAPI::RegTensor<T, Trait> vreg2;
        MicroAPI::RegTensor<U, Trait> vreg0CastB32;
        MicroAPI::RegTensor<U, Trait> vreg1CastB32;
        MicroAPI::UnalignReg uDst;
        uint32_t sreg1 = dimR;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<U>(sreg1);
        mask = MicroAPI::UpdateMask<U>(sreg1);
        MicroAPI::MaskPack(mask, mask);
        for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
            MicroAPI::LoadAlign(vreg0, srcAddr + loopA * dimR);
            MicroAPI::LoadAlign(vreg1, srcAddr + vlSize / 2 + loopA * dimR);
            Binaryfunc(vreg2, vreg0, vreg1, mask);
            Select(vreg2, vreg2, vreg0, mask);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t, Trait> &)vreg2,
                    (MicroAPI::RegTensor<uint16_t, Trait> &)vreg2);
            } else {
                MicroAPI::UnPack((MicroAPI::RegTensor<uint16_t, Trait> &)vreg2,
                    (MicroAPI::RegTensor<uint8_t, Trait> &)vreg2);
            }
            MicroAPI::Cast<U, T, CastTraitUppper>(vreg0CastB32, vreg2, fullMask);
            Reducefunc(vreg1CastB32, vreg0CastB32, fullMask);
            MicroAPI::Cast<T, U, CastTraitLower>(vreg1, vreg1CastB32, fullMask);
            MicroAPI::StoreUnAlign((__ubuf__ T *&)dstAddr, vreg1, uDst, 1);
        }
        MicroAPI::StoreUnAlignPost((__ubuf__ T *&)dstAddr, uDst, 0);
    }
}

template <class T, const MicroAPI::RegTrait &Trait, auto Reducefunc>
__simd_vf__ inline void ReduceARLessThanVL(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t dimA,
    uint32_t dimR)
{
    MicroAPI::RegTensor<T, Trait> vreg0;
    MicroAPI::RegTensor<T, Trait> vreg1;
    MicroAPI::UnalignReg uDst;
    uint32_t sreg1 = dimR;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T, Trait>(sreg1);
    for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
        MicroAPI::LoadAlign(vreg0, srcAddr + loopA * dimR);
        Reducefunc(vreg1, vreg0, mask);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dstAddr, vreg1, uDst, 1);
    }
    MicroAPI::StoreUnAlignPost((__ubuf__ T *&)dstAddr, uDst, 0);
}

enum class ReduceType {
    IS_REDUCE_SUM,
    IS_REDUCE_MAX,
    IS_REDUCE_MIN,
    OTHERS,
};

template <class T, const MicroAPI::RegTrait &Trait, ReduceType reduceType>
__simd_callee__ inline void GroupReduce(
    MicroAPI::RegTensor<T, Trait>& dst, MicroAPI::RegTensor<T, Trait>& src, MicroAPI::MaskReg& mask)
{
    if constexpr (reduceType == ReduceType::IS_REDUCE_SUM) {
        MicroAPI::ReduceSumWithDataBlock(dst, src, mask);
    } else if constexpr (reduceType == ReduceType::IS_REDUCE_MAX) {
        MicroAPI::ReduceMaxWithDataBlock(dst, src, mask);
    } else if constexpr (reduceType == ReduceType::IS_REDUCE_MIN) {
        MicroAPI::ReduceMinWithDataBlock(dst, src, mask);
    }
}

template <class T, const MicroAPI::RegTrait &Trait, const uint16_t vlSize, auto Binaryfunc,
          auto GroupReduceType, bool NeedFoldR = true>
__simd_vf__ inline void GroupReduceVf(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t innerFoldNum,
                                      uint16_t fusedA, uint16_t strideA, uint32_t outerRepElement)
{
    constexpr uint16_t blockDataLen = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t blockNumInVl = vlSize / blockDataLen;
    MicroAPI::RegTensor<T, Trait> vreg0;
    MicroAPI::RegTensor<T, Trait> vreg1;
    MicroAPI::UnalignReg uDst;
    MicroAPI::MaskReg mask;
    const uint16_t innerFoldBinaryNum = innerFoldNum - 1;

    for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(fusedA); ++loopA) {
        mask = MicroAPI::UpdateMask<T, Trait>(outerRepElement);
        if constexpr (NeedFoldR) {
            auto srcAddrFold = srcAddr + blockDataLen;
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg0, srcAddr, innerFoldNum, strideA, mask);
            for (uint16_t loopR = 0; loopR < innerFoldBinaryNum; ++loopR) {
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg1, srcAddrFold, innerFoldNum, 1, mask);
                Binaryfunc(vreg0, vreg1, vreg0, mask);
            }
        } else {
            MicroAPI::LoadAlign(vreg0, srcAddr + vlSize * loopA);
        }
        GroupReduce<T, Trait, GroupReduceType>(vreg1, vreg0, mask);
        MicroAPI::StoreUnAlign((__ubuf__ T*&)dstAddr, vreg1, uDst, blockNumInVl);
    }
    MicroAPI::StoreUnAlignPost((__ubuf__ T*&)dstAddr, uDst, blockNumInVl);
}

template <class T, const MicroAPI::RegTrait &Trait, const uint16_t vlSize, auto Binaryfunc,
          auto groupReduceType, bool NeedFoldR = true>
__aicore__ inline void GroupReduceARLessThanVL(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t blockDataLen = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t blockNumInVl = vlSize / blockDataLen;
    const auto innerFoldNum = dimR / blockDataLen;
    const auto fusedA = CeilDivision(dimA, blockNumInVl);
    const auto strideA = (blockNumInVl * dimR) / blockDataLen;
    uint32_t outerRepElement = dimA * blockDataLen;
    GroupReduceVf<T, Trait, vlSize, Binaryfunc, groupReduceType, NeedFoldR>(
        dstAddr, srcAddr, innerFoldNum, fusedA, strideA, outerRepElement);
}

template <class T, const MicroAPI::RegTrait &Trait, const uint16_t vlSize,
          auto Binaryfunc, auto Reducefunc, ReduceType groupReduceType = ReduceType::OTHERS>
__aicore__ inline void ReduceARReuseSourceLessThanVL(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t dimA,
    uint32_t dimR)
{
    if (dimR == 1) {
        ReduceOpInternal::ReduceCopyOutImpl<T>(dstAddr, srcAddr, dimA);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        ReduceARCastLessThanVL<T, float, Trait, ReduceOpInternal::CastTraitBF16F32, ReduceOpInternal::CastTraitF32BF16,
            vlSize, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
    } else if constexpr (SupportBytes<T, 1>()) {
        ReduceARCastLessThanVL<T, half, Trait, ReduceOpInternal::CastTraitB8F16, ReduceOpInternal::CastTraitF16B8,
            vlSize, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
    } else if constexpr (groupReduceType != ReduceType::OTHERS) {
        if (dimR == GetDataBlockSizeInBytes() / sizeof(T)) {
            GroupReduceARLessThanVL<T, Trait, vlSize, Binaryfunc, groupReduceType, false>(dstAddr, srcAddr, dimA, dimR);
        } else if (dimR < vlSize / 2) {
            GroupReduceARLessThanVL<T, Trait, vlSize, Binaryfunc, groupReduceType, true>(dstAddr, srcAddr, dimA, dimR);
        } else {
            ReduceARLessThanVL<T, Trait, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
        }
    } else {
        ReduceARLessThanVL<T, Trait, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
    }
}
} // namespace AscendC
#endif // IMPL_REDUCE_REDUCE_COMMON_AR_REUSE_ALIGN_LESS_THAN_VL_C310_IMPL_H