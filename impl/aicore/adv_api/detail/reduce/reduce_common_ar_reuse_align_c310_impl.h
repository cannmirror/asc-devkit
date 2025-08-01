/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICORE_ADV_API_DETAIL_REDUCE_REDUCE_COMMON_AR_REUSE_ALIGN_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_REDUCE_REDUCE_COMMON_AR_REUSE_ALIGN_C310_IMPL_H

#include "kernel_operator_intf.h"
#include "kernel_tensor.h"
#include "reduce_common_util_impl.h"
#include "reduce_common_util_c310_impl.h"
#include "reduce_common_ar_reuse_align_less_than_vl_c310_impl.h"

namespace AscendC {
template <class T, class U, const MicroAPI::RegTrait& Trait, const MicroAPI::CastTrait& CastTraitUppper,
    const MicroAPI::CastTrait& CastTraitLower, const uint16_t vlSize, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceARCastfoldZero(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR, MicroAPI::MaskReg& fullMask)
{
    using UnpackSrcT = typename ReduceOpInternal::ExtractUnsignedTypeBySize<sizeof(T)>::T;
    using UnpackDstT = typename ReduceOpInternal::ExtractUnsignedTypeBySize<sizeof(U)>::T;
    MicroAPI::RegTensor<U, Trait> vreg0CastB32;
    MicroAPI::RegTensor<U, Trait> vreg1CastB32;
    MicroAPI::RegTensor<T, Trait> vreg0;
    MicroAPI::RegTensor<T, Trait> vreg1;
    MicroAPI::UnalignReg uDst;
    for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
        DataCopy(vreg0, srcAddr + loopA * dimR);
        DataCopy(vreg1, srcAddr + vlSize / 2 + loopA * dimR);
        Binaryfunc(vreg0, vreg0, vreg1, fullMask);
        MicroAPI::UnPack(
            (MicroAPI::RegTensor<UnpackDstT, Trait>&)vreg0, (MicroAPI::RegTensor<UnpackSrcT, Trait>&)vreg0);
        MicroAPI::Cast<U, T, ReduceOpInternal::CastTraitBF16F32>(vreg0CastB32, vreg0, fullMask);
        Reducefunc(vreg1CastB32, vreg0CastB32, fullMask);
        MicroAPI::Cast<T, U, ReduceOpInternal::CastTraitF32BF16>(vreg1, vreg1CastB32, fullMask);
        DataCopyUnAlign((__ubuf__ T*&)dstAddr, vreg1, uDst, 1);
    }
    MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
}

template <class T, const MicroAPI::RegTrait& Trait, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceARfoldOneToThree(
    MicroAPI::RegTensor<T, Trait>& vreg0, MicroAPI::RegTensor<T, Trait>& vreg2, MicroAPI::MaskReg& fullMask)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        ReduceOpInternal::ReduceARCastfoldOneToThree<T, float, Trait, ReduceOpInternal::CastTraitBF16F32,
            ReduceOpInternal::CastTraitF32BF16, Binaryfunc, Reducefunc>(vreg0, vreg2, fullMask);
    } else if constexpr (SupportBytes<T, 1>()) {
        ReduceOpInternal::ReduceARCastfoldOneToThree<T, half, Trait, ReduceOpInternal::CastTraitB8F16,
            ReduceOpInternal::CastTraitF16B8, Binaryfunc, Reducefunc>(vreg0, vreg2, fullMask);
    } else {
        Reducefunc(vreg2, vreg0, fullMask);
    }
}

template <class T, const MicroAPI::RegTrait& Trait, const uint16_t vlSize, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceARB64ReuseSourceOverVL(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
{
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, true, vlSize);
    uint32_t tailR = dimR - mainR;
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
    uint16_t inplaceRepeats = (tailR + vlSize - 1) / vlSize;
    uint32_t dtypeSize = sizeof(T);

    uint16_t base = mainR / vlSize;
    uint16_t folds = ReduceOpInternal::CalculateFolds(base);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD_B64;
    uint16_t mainTimes = folds / avgFolds;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T, Trait> b64VregMain;
        MicroAPI::RegTensor<T, Trait> b64VregTail;
        MicroAPI::MaskReg mask;
        // Process mainR and tailR
        for (uint16_t i = 0; i < needInplaceAdd; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                uint32_t sreg0 = tailR;
                for (uint16_t loopR = 0; loopR < inplaceRepeats; loopR++) {
                    mask = MicroAPI::UpdateMask<T, Trait>(sreg0);
                    DataCopy(b64VregMain, srcAddr + loopA * dimR + loopR * vlSize);
                    DataCopy(b64VregTail, srcAddr + loopA * dimR + mainR + loopR * vlSize);
                    Binaryfunc(b64VregMain, b64VregMain, b64VregTail, mask);
                    DataCopy(srcAddr + loopA * dimR + loopR * vlSize, b64VregMain, mask);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // MainFolds need 8*2 register
        MicroAPI::RegTensor<T, Trait> b64Vreg0;
        MicroAPI::RegTensor<T, Trait> b64Vreg1;
        MicroAPI::RegTensor<T, Trait> b64Vreg2;
        MicroAPI::RegTensor<T, Trait> b64Vreg3;
        MicroAPI::RegTensor<T, Trait> b64Vreg4;
        MicroAPI::RegTensor<T, Trait> b64Vreg5;
        MicroAPI::RegTensor<T, Trait> b64Vreg6;
        MicroAPI::RegTensor<T, Trait> b64Vreg7;
        MicroAPI::MaskReg fullMask;
        MicroAPI::UnalignReg uDst;

        // Process main folds
        uint16_t loopRNum = base;
        fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, Trait>();
        for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
            loopRNum = loopRNum >> avgFolds;
            uint16_t offsetR = loopRNum * vlSize;
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                    // L0
                    DataCopy(b64Vreg0, srcAddr + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg1, srcAddr + offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg2, srcAddr + 2 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg3, srcAddr + 3 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg4, srcAddr + 4 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg5, srcAddr + 5 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg6, srcAddr + 6 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(b64Vreg7, srcAddr + 7 * offsetR + loopA * dimR + loopR * vlSize);
                    // L1
                    Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg4, fullMask);
                    Binaryfunc(b64Vreg1, b64Vreg1, b64Vreg5, fullMask);
                    Binaryfunc(b64Vreg2, b64Vreg2, b64Vreg6, fullMask);
                    Binaryfunc(b64Vreg3, b64Vreg3, b64Vreg7, fullMask);
                    // L2
                    Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg2, fullMask);
                    Binaryfunc(b64Vreg1, b64Vreg1, b64Vreg3, fullMask);
                    // L3
                    Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, fullMask);
                    DataCopy(srcAddr + loopA * dimR + loopR * vlSize, b64Vreg0, fullMask);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // Process tail folds
        for (uint16_t i = 0; i < foldOne; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                DataCopy(b64Vreg0, srcAddr + loopA * dimR);
                DataCopy(b64Vreg1, srcAddr + vlSize + loopA * dimR);
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, fullMask);
                Reducefunc(b64Vreg2, b64Vreg0, fullMask);
                DataCopyUnAlign((__ubuf__ T*&)dstAddr, b64Vreg2, uDst, 1);
            }
            MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
        }

        for (uint16_t i = 0; i < foldTwo; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                // L0
                DataCopy(b64Vreg0, srcAddr + loopA * dimR);
                DataCopy(b64Vreg1, srcAddr + vlSize + loopA * dimR);
                DataCopy(b64Vreg2, srcAddr + 2 * vlSize + loopA * dimR);
                DataCopy(b64Vreg3, srcAddr + 3 * vlSize + loopA * dimR);
                // L1
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg2, fullMask);
                Binaryfunc(b64Vreg1, b64Vreg1, b64Vreg3, fullMask);
                // L2
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, fullMask);
                Reducefunc(b64Vreg2, b64Vreg0, fullMask);
                DataCopyUnAlign((__ubuf__ T*&)dstAddr, b64Vreg2, uDst, 1);
            }
            MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
        }

        // Reduce to 1
        uint32_t sreg = mainR;
        for (uint16_t i = 0; i < foldZero; i++) {
            mask = MicroAPI::UpdateMask<T, Trait>(sreg);
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                DataCopy(b64Vreg0, srcAddr + loopA * dimR);
                Reducefunc(b64Vreg1, b64Vreg0, mask);
                DataCopyUnAlign((__ubuf__ T*&)dstAddr, b64Vreg1, uDst, 1);
            }
            MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
        }
    }
}

template <class T, const MicroAPI::RegTrait& Trait, const uint16_t vlSize, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceARReuseSourceOverVL(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
{
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, true, vlSize);
    uint32_t tailR = dimR - mainR;
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
    uint16_t inplaceRepeats = (tailR + vlSize - 1) / vlSize;
    uint32_t dtypeSize = sizeof(T);

    uint16_t base = mainR / vlSize;
    uint16_t folds = ReduceOpInternal::CalculateFolds(base);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD;
    uint16_t mainTimes = folds / avgFolds;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;
    uint16_t foldThree = (tailFolds == ReduceOpInternal::FLOD_THREE) ? 1 : 0;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T, Trait> vregMain;
        MicroAPI::RegTensor<T, Trait> vregTail;
        MicroAPI::MaskReg mask;
        // Process mainR and tailR
        for (uint16_t i = 0; i < needInplaceAdd; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                uint32_t sreg0 = tailR;
                for (uint16_t loopR = 0; loopR < inplaceRepeats; loopR++) {
                    mask = MicroAPI::UpdateMask<T, Trait>(sreg0);
                    DataCopy(vregMain, srcAddr + loopA * dimR + loopR * vlSize);
                    DataCopy(vregTail, srcAddr + loopA * dimR + mainR + loopR * vlSize);
                    Binaryfunc(vregMain, vregMain, vregTail, mask);
                    DataCopy(srcAddr + loopA * dimR + loopR * vlSize, vregMain, mask);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // MainFolds need 16 register
        MicroAPI::RegTensor<T, Trait> vreg0;
        MicroAPI::RegTensor<T, Trait> vreg1;
        MicroAPI::RegTensor<T, Trait> vreg2;
        MicroAPI::RegTensor<T, Trait> vreg3;
        MicroAPI::RegTensor<T, Trait> vreg4;
        MicroAPI::RegTensor<T, Trait> vreg5;
        MicroAPI::RegTensor<T, Trait> vreg6;
        MicroAPI::RegTensor<T, Trait> vreg7;
        MicroAPI::RegTensor<T, Trait> vreg8;
        MicroAPI::RegTensor<T, Trait> vreg9;
        MicroAPI::RegTensor<T, Trait> vreg10;
        MicroAPI::RegTensor<T, Trait> vreg11;
        MicroAPI::RegTensor<T, Trait> vreg12;
        MicroAPI::RegTensor<T, Trait> vreg13;
        MicroAPI::RegTensor<T, Trait> vreg14;
        MicroAPI::RegTensor<T, Trait> vreg15;
        MicroAPI::MaskReg fullMask;
        MicroAPI::UnalignReg uDst;

        // Process main folds
        uint16_t loopRNum = base;
        fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, Trait>();
        for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
            loopRNum = loopRNum >> avgFolds;
            uint16_t offsetR = loopRNum * vlSize;
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                    // L0
                    DataCopy(vreg0, srcAddr + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg1, srcAddr + offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg2, srcAddr + 2 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg3, srcAddr + 3 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg4, srcAddr + 4 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg5, srcAddr + 5 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg6, srcAddr + 6 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg7, srcAddr + 7 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg8, srcAddr + 8 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg9, srcAddr + 9 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg10, srcAddr + 10 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg11, srcAddr + 11 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg12, srcAddr + 12 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg13, srcAddr + 13 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg14, srcAddr + 14 * offsetR + loopA * dimR + loopR * vlSize);
                    DataCopy(vreg15, srcAddr + 15 * offsetR + loopA * dimR + loopR * vlSize);
                    // L1
                    Binaryfunc(vreg0, vreg0, vreg8, fullMask);
                    Binaryfunc(vreg1, vreg1, vreg9, fullMask);
                    Binaryfunc(vreg2, vreg2, vreg10, fullMask);
                    Binaryfunc(vreg3, vreg3, vreg11, fullMask);
                    Binaryfunc(vreg4, vreg4, vreg12, fullMask);
                    Binaryfunc(vreg5, vreg5, vreg13, fullMask);
                    Binaryfunc(vreg6, vreg6, vreg14, fullMask);
                    Binaryfunc(vreg7, vreg7, vreg15, fullMask);
                    // L2
                    Binaryfunc(vreg0, vreg0, vreg4, fullMask);
                    Binaryfunc(vreg1, vreg1, vreg5, fullMask);
                    Binaryfunc(vreg2, vreg2, vreg6, fullMask);
                    Binaryfunc(vreg3, vreg3, vreg7, fullMask);
                    // L3
                    Binaryfunc(vreg0, vreg0, vreg2, fullMask);
                    Binaryfunc(vreg1, vreg1, vreg3, fullMask);
                    // L4
                    Binaryfunc(vreg0, vreg0, vreg1, fullMask);
                    DataCopy(srcAddr + loopA * dimR + loopR * vlSize, vreg0, fullMask);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // Process tail folds
        for (uint16_t i = 0; i < foldOne; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                DataCopy(vreg0, srcAddr + loopA * dimR);
                DataCopy(vreg1, srcAddr + vlSize + loopA * dimR);
                Binaryfunc(vreg0, vreg0, vreg1, fullMask);
                ReduceARfoldOneToThree<T, Trait, Binaryfunc, Reducefunc>(vreg0, vreg2, fullMask);
                DataCopyUnAlign((__ubuf__ T*&)dstAddr, vreg2, uDst, 1);
            }
            MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
        }

        for (uint16_t i = 0; i < foldTwo; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                // L0
                DataCopy(vreg0, srcAddr + loopA * dimR);
                DataCopy(vreg1, srcAddr + vlSize + loopA * dimR);
                DataCopy(vreg2, srcAddr + 2 * vlSize + loopA * dimR);
                DataCopy(vreg3, srcAddr + 3 * vlSize + loopA * dimR);
                // L1
                Binaryfunc(vreg0, vreg0, vreg2, fullMask);
                Binaryfunc(vreg1, vreg1, vreg3, fullMask);
                // L2
                Binaryfunc(vreg0, vreg0, vreg1, fullMask);
                ReduceARfoldOneToThree<T, Trait, Binaryfunc, Reducefunc>(vreg0, vreg2, fullMask);
                DataCopyUnAlign((__ubuf__ T*&)dstAddr, vreg2, uDst, 1);
            }
            MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
        }

        for (uint16_t i = 0; i < foldThree; i++) {
            for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                // L0
                DataCopy(vreg0, srcAddr + loopA * dimR);
                DataCopy(vreg1, srcAddr + vlSize + loopA * dimR);
                DataCopy(vreg2, srcAddr + 2 * vlSize + loopA * dimR);
                DataCopy(vreg3, srcAddr + 3 * vlSize + loopA * dimR);
                DataCopy(vreg4, srcAddr + 4 * vlSize + loopA * dimR);
                DataCopy(vreg5, srcAddr + 5 * vlSize + loopA * dimR);
                DataCopy(vreg6, srcAddr + 6 * vlSize + loopA * dimR);
                DataCopy(vreg7, srcAddr + 7 * vlSize + loopA * dimR);
                // L1
                Binaryfunc(vreg0, vreg0, vreg4, fullMask);
                Binaryfunc(vreg1, vreg1, vreg5, fullMask);
                Binaryfunc(vreg2, vreg2, vreg6, fullMask);
                Binaryfunc(vreg3, vreg3, vreg7, fullMask);
                // L2
                Binaryfunc(vreg0, vreg0, vreg2, fullMask);
                Binaryfunc(vreg1, vreg1, vreg3, fullMask);
                // L3
                Binaryfunc(vreg0, vreg0, vreg1, fullMask);
                ReduceARfoldOneToThree<T, Trait, Binaryfunc, Reducefunc>(vreg0, vreg2, fullMask);
                DataCopyUnAlign((__ubuf__ T*&)dstAddr, vreg2, uDst, 1);
            }
            MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
        }

        // Reduce to 1
        uint32_t sreg = mainR;
        for (uint16_t i = 0; i < foldZero; i++) {
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ReduceARCastfoldZero<T, float, Trait, ReduceOpInternal::CastTraitBF16F32,
                    ReduceOpInternal::CastTraitF32BF16, vlSize, Binaryfunc, Reducefunc>(
                    dstAddr, srcAddr, dimA, dimR, fullMask);
            } else if constexpr (SupportBytes<T, 1>()) {
                ReduceARCastfoldZero<T, half, Trait, ReduceOpInternal::CastTraitB8F16, ReduceOpInternal::CastTraitF16B8,
                    vlSize, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR, fullMask);
            } else {
                mask = MicroAPI::UpdateMask<T, Trait>(sreg);
                for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
                    DataCopy(vreg0, srcAddr + loopA * dimR);
                    Reducefunc(vreg1, vreg0, mask);
                    DataCopyUnAlign((__ubuf__ T*&)dstAddr, vreg1, uDst, 1);
                }
                MicroAPI::DataCopyUnAlignPost((__ubuf__ T*&)dstAddr, uDst, 0);
            }
        }
    }
}

template <class T, const MicroAPI::RegTrait& Trait, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceARReuseSource(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = SupportBytes<T, 8>() ? GetVecLen() / sizeof(float) : GetVecLen() / sizeof(T);
    if (dimR < vlSize) {
        ReduceARReuseSourceLessThanVL<T, Trait, vlSize, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
    } else {
        if constexpr (SupportBytes<T, 8>()) {
            ReduceARB64ReuseSourceOverVL<T, Trait, vlSize, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
        } else {
            ReduceARReuseSourceOverVL<T, Trait, vlSize, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
        }
    }
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_REDUCE_REDUCE_COMMON_AR_REUSE_ALIGN_C310_IMPL_H