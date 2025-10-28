/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_REDUCE_REDUCE_COMMON_RA_REUSE_ALIGN_C310_IMPL_H
#define IMPL_REDUCE_REDUCE_COMMON_RA_REUSE_ALIGN_C310_IMPL_H

#include "kernel_operator_intf.h"
#include "kernel_tensor.h"
#include "reduce_common_util_impl.h"
#include "reduce_common_util_c310_impl.h"

namespace AscendC {
template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRAOverVLVFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint16_t dimA,
    uint32_t dimR, uint32_t mainR, uint32_t tailR, uint16_t loopANum, uint16_t loopANumFinal, uint16_t folds,
    uint16_t avgFolds, uint16_t foldZero, uint16_t foldOne, uint16_t foldTwo, uint16_t foldThree)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
    uint16_t mainTimes = folds / avgFolds;
    // Process vlSize axisA each time
    uint32_t inplaceA = dimA;
    uint32_t processA = dimA;
    uint32_t tailA = dimA;
    uint32_t copyA = dimA;
    uint32_t dtypeSize = sizeof(T);
    uint32_t aTailOffset = mainR * dimA;
    
    __ubuf__ T *addr;
    if constexpr (!isReuseSource) {
        MicroAPI::RegTensor<T, Trait> vregTmp;
        MicroAPI::MaskReg mask;
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(copyA);
            // 0 to tailR will be merge later, no need to move
            for (uint16_t loopR = static_cast<uint16_t>(tailR); loopR < static_cast<uint16_t>(mainR); loopR++) {
                DataCopy(vregTmp, srcAddr + loopA * vlSize + loopR * dimA);
                DataCopy(tmpAddr + loopA * vlSize + loopR * dimA, vregTmp, mask);
            }
        }
        addr = tmpAddr;
    } else {
        addr = srcAddr;
    }
    MicroAPI::RegTensor<T, Trait> vregMain;
    MicroAPI::RegTensor<T, Trait> vregTail;
    MicroAPI::MaskReg mask;
    // Process mainR and tailR
    for (uint16_t i = 0; i < needInplaceAdd; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(inplaceA);
            for (uint16_t loopR = 0; loopR < static_cast<uint16_t>(tailR); loopR++) {
                DataCopy(vregMain, srcAddr + loopA * vlSize + loopR * dimA);
                DataCopy(vregTail, srcAddr + loopA * vlSize + aTailOffset + loopR * dimA);
                Binaryfunc(vregMain, vregMain, vregTail, mask);
                DataCopy(addr + loopA * vlSize + loopR * dimA, vregMain, mask);
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

    // Process main folds
    uint16_t loopRNum = mainR;
    for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
        loopRNum = loopRNum >> avgFolds;
        uint16_t offsetR = loopRNum * dimA;
        uint32_t mainA = dimA;
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(mainA);
            for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                // L0
                DataCopy(vreg0, addr + loopA * vlSize + loopR * dimA);
                DataCopy(vreg1, addr + offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg2, addr + 2 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg3, addr + 3 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg4, addr + 4 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg5, addr + 5 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg6, addr + 6 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg7, addr + 7 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg8, addr + 8 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg9, addr + 9 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg10, addr + 10 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg11, addr + 11 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg12, addr + 12 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg13, addr + 13 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg14, addr + 14 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(vreg15, addr + 15 * offsetR + loopA * vlSize + loopR * dimA);
                // L1
                Binaryfunc(vreg0, vreg0, vreg8, mask);
                Binaryfunc(vreg1, vreg1, vreg9, mask);
                Binaryfunc(vreg2, vreg2, vreg10, mask);
                Binaryfunc(vreg3, vreg3, vreg11, mask);
                Binaryfunc(vreg4, vreg4, vreg12, mask);
                Binaryfunc(vreg5, vreg5, vreg13, mask);
                Binaryfunc(vreg6, vreg6, vreg14, mask);
                Binaryfunc(vreg7, vreg7, vreg15, mask);
                // L2
                Binaryfunc(vreg0, vreg0, vreg4, mask);
                Binaryfunc(vreg1, vreg1, vreg5, mask);
                Binaryfunc(vreg2, vreg2, vreg6, mask);
                Binaryfunc(vreg3, vreg3, vreg7, mask);
                // L3
                Binaryfunc(vreg0, vreg0, vreg2, mask);
                Binaryfunc(vreg1, vreg1, vreg3, mask);
                // L4
                Binaryfunc(vreg0, vreg0, vreg1, mask);
                DataCopy(addr + loopA * vlSize + loopR * dimA, vreg0, mask);
            }
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }

    // Process tail folds
    mask = MicroAPI::UpdateMask<T, Trait>(tailA);
    for (uint16_t i = 0; i < foldOne; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            // L0
            DataCopy(vreg0, addr + loopA * vlSize);
            DataCopy(vreg1, addr + dimA + loopA * vlSize);
            // L1
            Binaryfunc(vreg0, vreg0, vreg1, mask);
            DataCopy(dstAddr + loopA * vlSize, vreg0, mask);
        }
    }

    for (uint16_t i = 0; i < foldTwo; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            // L0
            DataCopy(vreg0, addr + loopA * vlSize);
            DataCopy(vreg1, addr + dimA + loopA * vlSize);
            DataCopy(vreg2, addr + 2 * dimA + loopA * vlSize);
            DataCopy(vreg3, addr + 3 * dimA + loopA * vlSize);
            // L1
            Binaryfunc(vreg0, vreg0, vreg2, mask);
            Binaryfunc(vreg1, vreg1, vreg3, mask);
            // L2
            Binaryfunc(vreg0, vreg0, vreg1, mask);
            DataCopy(dstAddr + loopA * vlSize, vreg0, mask);
        }
    }

    for (uint16_t i = 0; i < foldThree; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            // L0
            DataCopy(vreg0, addr + loopA * vlSize);
            DataCopy(vreg1, addr + dimA + loopA * vlSize);
            DataCopy(vreg2, addr + 2 * dimA + loopA * vlSize);
            DataCopy(vreg3, addr + 3 * dimA + loopA * vlSize);
            DataCopy(vreg4, addr + 4 * dimA + loopA * vlSize);
            DataCopy(vreg5, addr + 5 * dimA + loopA * vlSize);
            DataCopy(vreg6, addr + 6 * dimA + loopA * vlSize);
            DataCopy(vreg7, addr + 7 * dimA + loopA * vlSize);
            // L1
            Binaryfunc(vreg0, vreg0, vreg4, mask);
            Binaryfunc(vreg1, vreg1, vreg5, mask);
            Binaryfunc(vreg2, vreg2, vreg6, mask);
            Binaryfunc(vreg3, vreg3, vreg7, mask);
            // L2
            Binaryfunc(vreg0, vreg0, vreg2, mask);
            Binaryfunc(vreg1, vreg1, vreg3, mask);
            // L3
            Binaryfunc(vreg0, vreg0, vreg1, mask);
            DataCopy(dstAddr + loopA * vlSize, vreg0, mask);
        }
    }

    // Reduce to 1
    for (uint16_t i = 0; i < foldZero; i++) {
        for (uint16_t loopA = 0; loopA < loopANumFinal; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(processA);
            DataCopy(vreg0, addr + loopA * vlSize);
            DataCopy(dstAddr + loopA * vlSize, vreg0, mask);
        }
    }
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceRAOverVLImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint16_t dimA,
    uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
    uint32_t tailR = dimR - mainR;

    uint16_t loopANum = CeilDivision(dimA, vlSize);
    // move by fold zero only if R axis is 1
    uint16_t loopANumFinal = loopANum;
    if constexpr (!isReuseSource) {
        if (mainR == 1) {
            VF_CALL<ReduceOpInternal::ReduceCopyOutImpl<T>>(dstAddr, srcAddr, dimA);
            return;
        }
    }

    if constexpr (!isReuseSource) {
        if (tailR == 0 && mainR > 1) {
            mainR = mainR / 2;
            tailR = mainR;
        }
    }

    uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;
    uint16_t foldThree = (tailFolds == ReduceOpInternal::FLOD_THREE) ? 1 : 0;

    ReduceRAOverVLVFImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR,
        mainR, tailR, loopANum, loopANumFinal, folds, avgFolds, foldZero, foldOne, foldTwo, foldThree);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRALessThanVLDimR1VFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    MicroAPI::RegTensor<T, Trait> vregTmp;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T, Trait>(dimA);
    DataCopy(vregTmp, srcAddr);
    DataCopy(dstAddr, vregTmp, mask);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRALessThanVLVFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
    uint32_t dimR, uint32_t mainR, uint32_t tailR, uint16_t folds, uint16_t avgFolds, uint16_t foldZero, uint16_t foldOne,
    uint16_t foldTwo, uint16_t foldThree)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
    uint16_t mainTimes = folds / avgFolds;
        // Process vlSize axisA each time
    uint32_t processA = dimA;
    uint32_t dtypeSize = sizeof(T);
    uint32_t aTailOffset = mainR * dimA;
    uint32_t copyNum = (mainR - tailR) * dimA;
    uint32_t tailNum = tailR * dimA;
    uint16_t loopRNum = mainR;

    __ubuf__ T *addr;
    MicroAPI::MaskReg mask;
    mask = MicroAPI::UpdateMask<T, Trait>(processA);
    MicroAPI::MaskReg counterMask;

    if constexpr (!isReuseSource) {
        MicroAPI::RegTensor<T, Trait> vregTmp;
        uint16_t mainRepeat = CeilDivision(copyNum, vlSize);
        // 0 to tailR will be merge later, no need to move
        for (uint16_t loopMain = 0; loopMain < mainRepeat; loopMain++) {
            counterMask = MicroAPI::UpdateMask<T, Trait>(copyNum);
            DataCopy(vregTmp, srcAddr + tailNum + loopMain * vlSize);
            DataCopy(tmpAddr + tailNum + loopMain * vlSize, vregTmp, counterMask);
        }
        addr = tmpAddr;
    } else {
        addr = srcAddr;
    }

    MicroAPI::RegTensor<T, Trait> vregMain;
    MicroAPI::RegTensor<T, Trait> vregTail;
    // Process mainR and tailR
    for (uint16_t i = 0; i < needInplaceAdd; i++) {
        uint16_t tailRepeat = CeilDivision(tailNum, vlSize);
        for (uint16_t loopTail = 0; loopTail < tailRepeat; loopTail++) {
            counterMask = MicroAPI::UpdateMask<T, Trait>(tailNum);
            DataCopy(vregMain, srcAddr + loopTail * vlSize);
            DataCopy(vregTail, srcAddr + aTailOffset + loopTail * vlSize);
            Binaryfunc(vregMain, vregMain, vregTail, counterMask);
            DataCopy(addr + loopTail * vlSize, vregMain, counterMask);
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

    // Process main folds
    for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
        loopRNum = loopRNum >> avgFolds;
        auto tmpSrcAddr = addr;
        for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
            // L0
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg4, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg5, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg6, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg7, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg8, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg9, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg10, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg11, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg12, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg13, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg14, tmpSrcAddr, dimA);
            DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg15, tmpSrcAddr, dimA);
            // L1
            Binaryfunc(vreg0, vreg0, vreg8, mask);
            Binaryfunc(vreg1, vreg1, vreg9, mask);
            Binaryfunc(vreg2, vreg2, vreg10, mask);
            Binaryfunc(vreg3, vreg3, vreg11, mask);
            Binaryfunc(vreg4, vreg4, vreg12, mask);
            Binaryfunc(vreg5, vreg5, vreg13, mask);
            Binaryfunc(vreg6, vreg6, vreg14, mask);
            Binaryfunc(vreg7, vreg7, vreg15, mask);
            // L2
            Binaryfunc(vreg0, vreg0, vreg4, mask);
            Binaryfunc(vreg1, vreg1, vreg5, mask);
            Binaryfunc(vreg2, vreg2, vreg6, mask);
            Binaryfunc(vreg3, vreg3, vreg7, mask);
            // L3
            Binaryfunc(vreg0, vreg0, vreg2, mask);
            Binaryfunc(vreg1, vreg1, vreg3, mask);
            // L4
            Binaryfunc(vreg0, vreg0, vreg1, mask);
            DataCopy(addr + loopR * dimA, vreg0, mask);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }

    // Process tail folds
    for (uint16_t i = 0; i < foldOne; i++) {
        // L0
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, addr, dimA);
        // L1
        Binaryfunc(vreg0, vreg0, vreg1, mask);
        DataCopy(dstAddr, vreg0, mask);
    }

    for (uint16_t i = 0; i < foldTwo; i++) {
        // L0
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, addr, dimA);
        // L1
        Binaryfunc(vreg0, vreg0, vreg2, mask);
        Binaryfunc(vreg1, vreg1, vreg3, mask);
        // L2
        Binaryfunc(vreg0, vreg0, vreg1, mask);
        DataCopy(dstAddr, vreg0, mask);
    }

    for (uint16_t i = 0; i < foldThree; i++) {
        // L0
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg4, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg5, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg6, addr, dimA);
        DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg7, addr, dimA);
        // L1
        Binaryfunc(vreg0, vreg0, vreg4, mask);
        Binaryfunc(vreg1, vreg1, vreg5, mask);
        Binaryfunc(vreg2, vreg2, vreg6, mask);
        Binaryfunc(vreg3, vreg3, vreg7, mask);
        // L2
        Binaryfunc(vreg0, vreg0, vreg2, mask);
        Binaryfunc(vreg1, vreg1, vreg3, mask);
        // L3
        Binaryfunc(vreg0, vreg0, vreg1, mask);
        DataCopy(dstAddr, vreg0, mask);
    }

    // Reduce to 1
    for (uint16_t i = 0; i < foldZero; i++) {
        DataCopy(vreg0, addr);
        DataCopy(dstAddr, vreg0, mask);
    }
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceRALessThanVLImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
    uint32_t tailR = dimR - mainR;
    if constexpr (!isReuseSource) {
        if (tailR == 0) {
            mainR = mainR / 2;
            tailR = mainR;
        }
    }
    if (dimR == 1) {
        ReduceRALessThanVLDimR1VFImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
        return;
    }

    uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;
    uint16_t foldThree = (tailFolds == ReduceOpInternal::FLOD_THREE) ? 1 : 0;

    ReduceRALessThanVLVFImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA,
        dimR, mainR, tailR, folds, avgFolds, foldZero, foldOne, foldTwo, foldThree);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRAConcatDimR1VFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    MicroAPI::RegTensor<T, Trait> vregTmp;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T, Trait>(dimA);
    DataCopy(vregTmp, srcAddr);
    DataCopy(dstAddr, vregTmp, mask);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRAConcatDimR2VFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    MicroAPI::RegTensor<T, Trait> vregMain;
    MicroAPI::RegTensor<T, Trait> vregTail;
    uint32_t maskScalar = dimA;
    MicroAPI::MaskReg counterMask = MicroAPI::UpdateMask<T, Trait>(maskScalar);
    DataCopy(vregMain, srcAddr);
    DataCopy(vregTail, srcAddr + dimA);
    Binaryfunc(vregMain, vregMain, vregTail, counterMask);
    DataCopy(dstAddr, vregMain, counterMask);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRAConcatVFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
    uint32_t dimR, uint16_t foldTime, uint32_t mainR, uint32_t tailR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
    // Process vlSize axisA each time
    uint32_t processA = dimA;
    uint32_t dtypeSize = sizeof(T);
    uint32_t aTailOffset = mainR * dimA;
    // do mainR-tailR copy, do tailR binary op
    uint32_t copyNum = (mainR - tailR) * dimA;
    uint32_t tailNum = tailR * dimA;
    uint16_t loopDataNum = mainR * dimA;
    
    __ubuf__ T *addr;
    MicroAPI::MaskReg fullMask;
    fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::MaskReg counterMask;

    if constexpr (!isReuseSource) {
        MicroAPI::RegTensor<T, Trait> vregTmp;
        uint16_t mainRepeat = CeilDivision(copyNum, vlSize);
        // 0 to tailR will be merge later, no need to move
        for (uint16_t loopMain = 0; loopMain < mainRepeat; loopMain++) {
            counterMask = MicroAPI::UpdateMask<T, Trait>(copyNum);
            DataCopy(vregTmp, srcAddr + tailNum + loopMain * vlSize);
            DataCopy(tmpAddr + tailNum + loopMain * vlSize, vregTmp, counterMask);
        }
        addr = tmpAddr;
    } else {
        addr = srcAddr;
    }

    MicroAPI::RegTensor<T, Trait> vregMain;
    MicroAPI::RegTensor<T, Trait> vregTail;
    // Process mainR and tailR
    for (uint16_t i = 0; i < needInplaceAdd; i++) {
        uint16_t tailRepeat = CeilDivision(tailNum, vlSize);
        for (uint16_t loopTail = 0; loopTail < tailRepeat; loopTail++) {
            counterMask = MicroAPI::UpdateMask<T, Trait>(tailNum);
            DataCopy(vregMain, srcAddr + loopTail * vlSize);
            DataCopy(vregTail, srcAddr + aTailOffset + loopTail * vlSize);
            Binaryfunc(vregMain, vregMain, vregTail, counterMask);
            DataCopy(addr + loopTail * vlSize, vregMain, counterMask);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }

    for (uint16_t fold = 0; fold < foldTime; fold++) {
        mainR = mainR >> 1;
        uint32_t foldDataNum = mainR * dimA;
        uint16_t foldRepeat = CeilDivision(foldDataNum, vlSize);
        for (uint16_t i = 0; i < foldRepeat; i++) {
            DataCopy(vregMain, addr + i * vlSize);
            DataCopy(vregTail, addr + foldDataNum + i * vlSize);
            Binaryfunc(vregMain, vregMain, vregTail, fullMask);
            DataCopy(addr + i * vlSize, vregMain, fullMask);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }

    // final fold is less than vl, no repeat
    uint32_t maskScalar = dimA;
    counterMask = MicroAPI::UpdateMask<T, Trait>(maskScalar);
    DataCopy(vregMain, addr);
    DataCopy(vregTail, addr + dimA);
    Binaryfunc(vregMain, vregMain, vregTail, counterMask);
    DataCopy(dstAddr, vregMain, counterMask);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceRAConcatImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint16_t foldTime = Internal::FindClosestPowerOfTwo(dimR);
    uint32_t mainR = 1 << foldTime;
    // last fold not in main loop, main R == 1 will not enter main loop
    foldTime = foldTime - 1;
    uint32_t tailR = dimR - mainR;

    if constexpr (!isReuseSource) {
        if (tailR == 0) {
            mainR = mainR / 2;
            tailR = mainR;
            foldTime = foldTime - 1;
        }
    }
    if (dimR == 1) {
        ReduceRAConcatDimR1VFImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
        return;
    } else if (dimR == 2) {
        ReduceRAConcatDimR2VFImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
        return;
    }

    ReduceRAConcatVFImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr,
        dimA, dimR, foldTime, mainR, tailR);
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceRAImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
    uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    if (dimA <= vlSize / ReduceOpInternal::REGULAR_FOLD_NUM || dimA > ReduceOpInternal::U16_STRIDE) {
        ReduceRAConcatImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
    } else if (dimA <= vlSize) {
        ReduceRALessThanVLImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
    } else {
        ReduceRAOverVLImpl<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, static_cast<uint16_t>(dimA), dimR);
    }
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceRAB64ReuseSourceVF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
    uint32_t dimR, uint32_t mainR, uint32_t tailR, uint16_t loopANum, uint16_t loopANumFinal, uint16_t folds, uint16_t avgFolds,
    uint16_t foldZero, uint16_t foldOne, uint16_t foldTwo)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(float);
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
    uint16_t mainTimes = folds / avgFolds;
    // Process vlSize axisA each time
    uint32_t inplaceA = dimA;
    uint32_t processA = dimA;
    uint32_t tailA = dimA;
    uint32_t copyA = dimA;
    uint32_t aTailOffset = mainR * dimA;
    
    __ubuf__ T *addr;
    if constexpr (!isReuseSource) {
        MicroAPI::RegTensor<T, Trait> vregTmp;
        MicroAPI::MaskReg mask;
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(copyA);
            for (uint16_t loopR = static_cast<uint16_t>(tailR); loopR < static_cast<uint16_t>(mainR); loopR++) {
                DataCopy(vregTmp, srcAddr + loopA * vlSize + loopR * dimA);
                DataCopy(tmpAddr + loopA * vlSize + loopR * dimA, vregTmp, mask);
            }
        }
        addr = tmpAddr;
    } else {
        addr = srcAddr;
    }
    MicroAPI::RegTensor<T, Trait> b64VregMain;
    MicroAPI::RegTensor<T, Trait> b64VregTail;
    MicroAPI::MaskReg mask;
    // Add mainR and tailR
    for (uint16_t i = 0; i < needInplaceAdd; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(inplaceA);
            for (uint16_t loopR = 0; loopR < static_cast<uint16_t>(tailR); loopR++) {
                DataCopy(b64VregMain, srcAddr + loopA * vlSize + loopR * dimA);
                DataCopy(b64VregTail, srcAddr + loopA * vlSize + aTailOffset + loopR * dimA);
                Binaryfunc(b64VregMain, b64VregMain, b64VregTail, mask);
                DataCopy(addr + loopA * vlSize + loopR * dimA, b64VregMain, mask);
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

    // Process main folds
    uint16_t loopRNum = mainR;
    for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
        loopRNum = loopRNum >> avgFolds;
        uint16_t offsetR = loopRNum * dimA;
        uint32_t mainA = dimA;
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(mainA);
            for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                // L0
                DataCopy(b64Vreg0, addr + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg1, addr + offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg2, addr + 2 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg3, addr + 3 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg4, addr + 4 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg5, addr + 5 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg6, addr + 6 * offsetR + loopA * vlSize + loopR * dimA);
                DataCopy(b64Vreg7, addr + 7 * offsetR + loopA * vlSize + loopR * dimA);
                // L1
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg4, mask);
                Binaryfunc(b64Vreg1, b64Vreg1, b64Vreg5, mask);
                Binaryfunc(b64Vreg2, b64Vreg2, b64Vreg6, mask);
                Binaryfunc(b64Vreg3, b64Vreg3, b64Vreg7, mask);
                // L2
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg2, mask);
                Binaryfunc(b64Vreg1, b64Vreg1, b64Vreg3, mask);
                // L3
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, mask);
                DataCopy(addr + loopA * vlSize + loopR * dimA, b64Vreg0, mask);
            }
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }

    // Process tail folds
    mask = MicroAPI::UpdateMask<T, Trait>(tailA);
    for (uint16_t i = 0; i < foldOne; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            // L0
            DataCopy(b64Vreg0, addr + loopA * vlSize);
            DataCopy(b64Vreg1, addr + dimA + loopA * vlSize);
            // L1
            Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, mask);
            DataCopy(dstAddr + loopA * vlSize, b64Vreg0, mask);
        }
    }

    for (uint16_t i = 0; i < foldTwo; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            // L0
            DataCopy(b64Vreg0, addr + loopA * vlSize);
            DataCopy(b64Vreg1, addr + dimA + loopA * vlSize);
            DataCopy(b64Vreg2, addr + 2 * dimA + loopA * vlSize);
            DataCopy(b64Vreg3, addr + 3 * dimA + loopA * vlSize);
            // L1
            Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg2, mask);
            Binaryfunc(b64Vreg1, b64Vreg1, b64Vreg3, mask);
            // L2
            Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, mask);
            DataCopy(dstAddr + loopA * vlSize, b64Vreg0, mask);
        }
    }

    // Reduce to 1
    for (uint16_t i = 0; i < foldZero; i++) {
        for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
            mask = MicroAPI::UpdateMask<T, Trait>(processA);
            DataCopy(b64Vreg0, addr + loopA * vlSize);
            DataCopy(dstAddr + loopA * vlSize, b64Vreg0, mask);
        }
    }
}

template <class T, const MicroAPI::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceRAB64ReuseSource(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(float);
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
    uint32_t tailR = dimR - mainR;

    uint16_t loopANum = (dimA + vlSize - 1) / vlSize;
    // move by fold zero only if R axis is 1
    uint16_t loopANumFinal = loopANum;
    if constexpr (!isReuseSource) {
        if (mainR == 1) {
            loopANum = 0;
            tmpAddr = srcAddr;
        }
    }
    
    if constexpr (!isReuseSource) {
        if (tailR == 0 && mainR > 1) {
            mainR = mainR / 2;
            tailR = mainR;
        }
    }
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;

    uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD_B64;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;

    ReduceRAB64ReuseSourceVF<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA,
        dimR, mainR, tailR, loopANum, loopANumFinal, folds, avgFolds, foldZero, foldOne, foldTwo);
}
} // namespace AscendC
#endif // IMPL_REDUCE_REDUCE_COMMON_RA_REUSE_ALIGN_C310_IMPL_H