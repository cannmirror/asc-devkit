/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICORE_ADV_API_DETAIL_REDUCE_REDUCE_COMMON_RA_REUSE_ALIGN_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_REDUCE_REDUCE_COMMON_RA_REUSE_ALIGN_C310_IMPL_H

#include "kernel_operator_intf.h"
#include "kernel_tensor.h"
#include "reduce_common_util_impl.h"
#include "reduce_common_util_c310_impl.h"

namespace AscendC {
template <class T, const MicroAPI::RegTrait& Trait, auto Binaryfunc>
__aicore__ inline void ReduceRAReuseSource(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
    uint32_t tailR = dimR - mainR;
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;

    uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD;
    uint16_t mainTimes = folds / avgFolds;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;
    uint16_t foldThree = (tailFolds == ReduceOpInternal::FLOD_THREE) ? 1 : 0;

    // Process vlSize axisA each time
    uint16_t loopANum = (dimA + vlSize - 1) / vlSize;
    uint32_t inplaceA = dimA;
    uint32_t processA = dimA;
    uint32_t tailA = dimA;
    uint32_t dtypeSize = sizeof(T);
    uint32_t aTailOffset = mainR * dimA;

    __VEC_SCOPE__
    {
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
                    DataCopy(srcAddr + loopA * vlSize + loopR * dimA, vregMain, mask);
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
                    DataCopy(vreg0, srcAddr + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg1, srcAddr + offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg2, srcAddr + 2 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg3, srcAddr + 3 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg4, srcAddr + 4 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg5, srcAddr + 5 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg6, srcAddr + 6 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg7, srcAddr + 7 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg8, srcAddr + 8 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg9, srcAddr + 9 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg10, srcAddr + 10 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg11, srcAddr + 11 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg12, srcAddr + 12 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg13, srcAddr + 13 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg14, srcAddr + 14 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(vreg15, srcAddr + 15 * offsetR + loopA * vlSize + loopR * dimA);
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
                    DataCopy(srcAddr + loopA * vlSize + loopR * dimA, vreg0, mask);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // Process tail folds
        mask = MicroAPI::UpdateMask<T, Trait>(tailA);
        for (uint16_t i = 0; i < foldOne; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                // L0
                DataCopy(vreg0, srcAddr + loopA * vlSize);
                DataCopy(vreg1, srcAddr + dimA + loopA * vlSize);
                // L1
                Binaryfunc(vreg0, vreg0, vreg1, mask);
                DataCopy(dstAddr + loopA * vlSize, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < foldTwo; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                // L0
                DataCopy(vreg0, srcAddr + loopA * vlSize);
                DataCopy(vreg1, srcAddr + dimA + loopA * vlSize);
                DataCopy(vreg2, srcAddr + 2 * dimA + loopA * vlSize);
                DataCopy(vreg3, srcAddr + 3 * dimA + loopA * vlSize);
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
                DataCopy(vreg0, srcAddr + loopA * vlSize);
                DataCopy(vreg1, srcAddr + dimA + loopA * vlSize);
                DataCopy(vreg2, srcAddr + 2 * dimA + loopA * vlSize);
                DataCopy(vreg3, srcAddr + 3 * dimA + loopA * vlSize);
                DataCopy(vreg4, srcAddr + 4 * dimA + loopA * vlSize);
                DataCopy(vreg5, srcAddr + 5 * dimA + loopA * vlSize);
                DataCopy(vreg6, srcAddr + 6 * dimA + loopA * vlSize);
                DataCopy(vreg7, srcAddr + 7 * dimA + loopA * vlSize);
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
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = MicroAPI::UpdateMask<T, Trait>(processA);
                DataCopy(vreg0, srcAddr + loopA * vlSize);
                DataCopy(dstAddr + loopA * vlSize, vreg0, mask);
            }
        }
    }
}

template <class T, const MicroAPI::RegTrait& Trait, auto Binaryfunc>
__aicore__ inline void ReduceRAB64ReuseSource(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(float);
    uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
    uint32_t tailR = dimR - mainR;
    uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;

    uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
    uint16_t avgFolds = ReduceOpInternal::BASE_FOLD_B64;
    uint16_t mainTimes = folds / avgFolds;
    uint16_t tailFolds = folds % avgFolds;
    uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
    uint16_t foldOne = (tailFolds == ReduceOpInternal::FLOD_ONE) ? 1 : 0;
    uint16_t foldTwo = (tailFolds == ReduceOpInternal::FLOD_TWO) ? 1 : 0;

    // Process vlSize axisA each time
    uint16_t loopANum = (dimA + vlSize - 1) / vlSize;
    uint32_t inplaceA = dimA;
    uint32_t processA = dimA;
    uint32_t tailA = dimA;
    uint32_t dtypeSize = sizeof(T);
    uint32_t aTailOffset = mainR * dimA;

    __VEC_SCOPE__
    {
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
                    DataCopy(srcAddr + loopA * vlSize + loopR * dimA, b64VregMain, mask);
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
                    DataCopy(b64Vreg0, srcAddr + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg1, srcAddr + offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg2, srcAddr + 2 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg3, srcAddr + 3 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg4, srcAddr + 4 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg5, srcAddr + 5 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg6, srcAddr + 6 * offsetR + loopA * vlSize + loopR * dimA);
                    DataCopy(b64Vreg7, srcAddr + 7 * offsetR + loopA * vlSize + loopR * dimA);
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
                    DataCopy(srcAddr + loopA * vlSize + loopR * dimA, b64Vreg0, mask);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // Process tail folds
        mask = MicroAPI::UpdateMask<T, Trait>(tailA);
        for (uint16_t i = 0; i < foldOne; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                // L0
                DataCopy(b64Vreg0, srcAddr + loopA * vlSize);
                DataCopy(b64Vreg1, srcAddr + dimA + loopA * vlSize);
                // L1
                Binaryfunc(b64Vreg0, b64Vreg0, b64Vreg1, mask);
                DataCopy(dstAddr + loopA * vlSize, b64Vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < foldTwo; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                // L0
                DataCopy(b64Vreg0, srcAddr + loopA * vlSize);
                DataCopy(b64Vreg1, srcAddr + dimA + loopA * vlSize);
                DataCopy(b64Vreg2, srcAddr + 2 * dimA + loopA * vlSize);
                DataCopy(b64Vreg3, srcAddr + 3 * dimA + loopA * vlSize);
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
                DataCopy(b64Vreg0, srcAddr + loopA * vlSize);
                DataCopy(dstAddr + loopA * vlSize, b64Vreg0, mask);
            }
        }
    }
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_REDUCE_REDUCE_COMMON_RA_REUSE_ALIGN_C310_IMPL_H