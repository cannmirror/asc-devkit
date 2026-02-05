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
 * \file kernel_operator_vec_sel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_SEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_SEL_IMPL_H

#include "kernel_utils.h"
#include "micro_api/kernel_micro_intf.h"

namespace AscendC {
namespace SelInternal {
    constexpr uint32_t maskBitToByte = 8;
}
/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */
template <typename T, bool isCounterMode>
__simd_vf__ inline void SelectWithoutMaskMode0ImplVF(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t *tempBuf, int32_t repeat, const BinaryRepeatParams repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    MicroAPI::MaskReg maskReg, selMask;
    MicroAPI::RegTensor<uint32_t> selReg;
    MicroAPI::UnalignReg ureg;
    uint16_t newRepeatTimes = repeat;
    uint32_t sreg;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    if constexpr (sizeof(T) == 2) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint32_t *)tempBuf);
    } else if constexpr (sizeof(T) == 4) {
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint32_t *)tempBuf);
        MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint32_t *)tempBuf);
        MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, selReg);
    }
    if constexpr (isCounterMode) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
        newRepeatTimes = CeilDivision(sreg, oneRepSize);
    } else {
        maskReg = MicroAPI::MoveMask<T>();
    }
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0,
            src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1,
            src1 + i * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
        MicroAPI::Select(dstReg, srcReg0, srcReg1, selMask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dst + i * blockElm * repeatParams.dstRepStride,
            dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
    }
}

template <typename T, bool isCounterMode>
__simd_vf__ inline void SelectWithoutMaskMode2ImplVF(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t *tempBuf, uint64_t selAddr, int32_t repeat, const BinaryRepeatParams repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    MicroAPI::MaskReg maskReg, selMask;
    MicroAPI::RegTensor<uint8_t> selReg;
    MicroAPI::UnalignReg ureg;
    uint16_t newRepeatTimes = repeat;
    constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    uint32_t sreg;
    if constexpr (isCounterMode) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
        newRepeatTimes = CeilDivision(sreg, oneRepSize);
    } else {
        maskReg = MicroAPI::MoveMask<T>();
    }
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
        if constexpr (sizeof(T) == 2) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)selAddr + i * selOffset);
        } else if constexpr (sizeof(T) == 4) {
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)selAddr + i * selOffset);
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)selAddr + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            srcReg0, src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            srcReg1, src1 + i * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
        MicroAPI::Select(dstReg, srcReg0, srcReg1, selMask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
    }
}

template <typename T, SELMODE selMode>
__aicore__ inline void SelectCal(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, int32_t repeat, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    if constexpr (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        if constexpr (sizeof(T) == 2) {
            (*(__ubuf__ uint64_t *)((__ubuf__ uint64_t *)tempBuf)) = Internal::g_cmpMaskLow;
            (*(__ubuf__ uint64_t *)((__ubuf__ uint64_t *)tempBuf + 1)) = Internal::g_cmpMaskHigh;
        } else {
            (*(__ubuf__ uint64_t *)((__ubuf__ uint64_t *)tempBuf)) = Internal::g_cmpMaskLow;
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (isCounterMode) {
            SelectWithoutMaskMode0ImplVF<T, true>(dst, src0, src1, tempBuf, repeat, repeatParams);
        } else {
            SelectWithoutMaskMode0ImplVF<T, false>(dst, src0, src1, tempBuf, repeat, repeatParams);
        }
    }
    else if constexpr (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        uint64_t selAddr = Internal::g_cmpMaskLow;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (isCounterMode) {
            SelectWithoutMaskMode2ImplVF<T, true>(dst, src0, src1, tempBuf, selAddr, repeat, repeatParams);
        } else {
            SelectWithoutMaskMode2ImplVF<T, false>(dst, src0, src1, tempBuf, selAddr, repeat, repeatParams);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
}

template <typename T, typename U, bool isCounterMode>
__simd_vf__ inline void SelectWithoutMaskMode1ImplVF(
    __ubuf__ T *dst, __ubuf__ U *sel, __ubuf__ T *src0, T scalar, __ubuf__ uint64_t *tempBuf, int32_t repeat, const BinaryRepeatParams repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    MicroAPI::MaskReg maskReg, selMask;
    MicroAPI::RegTensor<uint8_t> selReg;
    MicroAPI::UnalignReg ureg;
    uint16_t newRepeatTimes = repeat;
    uint32_t sreg;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
    if constexpr (isCounterMode) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
        newRepeatTimes = CeilDivision(sreg, oneRepSize);
    } else {
        maskReg = MicroAPI::MoveMask<T>();
    }
    MicroAPI::Duplicate(srcReg1, scalar);
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
        if constexpr (sizeof(T) == 2) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
        } else if constexpr (sizeof(T) == 4) {
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            srcReg0, src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
        MicroAPI::Select(dstReg, srcReg0, srcReg1, selMask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
    }
}

template <typename T, typename U>
__aicore__ inline void SelectCal(
    __ubuf__ T *dst, __ubuf__ U *sel, __ubuf__ T *src0, int32_t repeat, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    T scalar = *reinterpret_cast<T*>(&Internal::g_cmpMaskLow);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    if (isCounterMode) {
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        SelectWithoutMaskMode1ImplVF<T, U, true>(dst, sel, src0, scalar, tempBuf, repeat, repeatParams);
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        SelectWithoutMaskMode1ImplVF<T, U, false>(dst, sel, src0, scalar, nullptr, repeat, repeatParams);
    }
}

// ============ select mode: 0/2 ============
// ================Level2====================
template <typename T, typename U, bool isBitMap, bool isCounterMode>
__simd_vf__ inline void SelectMode0Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
    MicroAPI::MaskReg maskReg;
    uint32_t sreg;
    uint16_t newRepeatTimes = repeatTime;
    if constexpr (isCounterMode) {
        sreg = static_cast<uint32_t>(mask);
        newRepeatTimes = CeilDivision(sreg, oneRepSize);
    } else {
        if constexpr (isBitMap) {
            maskReg = MicroAPI::MoveMask<T>();
        } else {
            sreg = static_cast<uint32_t>(mask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
    }
    MicroAPI::MaskReg selMask;
    MicroAPI::LoadAlign<U, MicroAPI::MaskDist::DIST_US>(selMask, sel);
    if constexpr (sizeof(T) == 4) {
        MicroAPI::MaskUnPack(selMask, selMask);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        if constexpr (isCounterMode) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
        MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
    }
}

template <typename T, typename U, bool isBitMap, bool isCounterMode>
__simd_vf__ inline void SelectMode2Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = repeatTime;
    uint32_t sreg;
    if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        MicroAPI::RegTensor<T> src0Reg, src1Reg, src2Reg, src3Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        if constexpr (isCounterMode) {
            sreg = static_cast<uint32_t>(mask);
            newRepeatTimes = CeilDivision(sreg, oneRepSize);
        } else {
            if constexpr (isBitMap) {
                maskReg = MicroAPI::MoveMask<T>();
            } else {
                sreg = static_cast<uint32_t>(mask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
        }
        
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = newRepeatTimes % unRollConstant;
        newRepeatTimes = newRepeatTimes / unRollConstant;
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * unRollConstant * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + i * unRollConstant * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dst0Reg, src0Reg, src1Reg, selMask0);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src2Reg, src0 + (i * unRollConstant + 1) * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src3Reg, src1 + (i * unRollConstant + 1) * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dst1Reg, src2Reg, src3Reg, selMask1);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
        MicroAPI::RegTensor<T> src4Reg, src5Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
        uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.src1RepStride * blockElm;
        uint32_t offset2 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(tailSreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src4Reg, src0 + offset0, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src5Reg, src1 + offset1, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dst2Reg, src4Reg, src5Reg, selMask2);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + offset2, dst2Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg maskReg;
        if constexpr (isCounterMode) {
            sreg = static_cast<uint32_t>(mask);
            newRepeatTimes = CeilDivision(sreg, oneRepSize);
        } else {
            if constexpr (isBitMap) {
                maskReg = MicroAPI::MoveMask<T>();
            } else {
                sreg = static_cast<uint32_t>(mask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
        }
        MicroAPI::MaskReg selMask;
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            SelectMode0Level0<T, U, false, true>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            SelectMode2Level0<T, U, false, true>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        }
    } else {
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            SelectMode0Level0<T, U, false, false>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            SelectMode2Level0<T, U, false, false>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    SetVectorMask<T>(mask[1], mask[0]);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            SelectMode0Level0<T, U, true, true>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            SelectMode2Level0<T, U, true, true>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    } else {
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            SelectMode0Level0<T, U, true, false>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            SelectMode2Level0<T, U, true, false>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    }
}
// ============ select mode: 1 ============
// ================Level0====================

template <typename T, typename U, bool isBitMap, bool isCounterMode>
__simd_vf__ inline void SelectMode1Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = repeatTime;
    uint32_t sreg;
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::Duplicate(src1Reg, (const T &) src1);
        MicroAPI::MaskReg maskReg;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
        if constexpr (isCounterMode) {
            sreg = static_cast<uint32_t>(mask);
            newRepeatTimes = CeilDivision(sreg, oneRepSize);
        } else {
            if constexpr (isBitMap) {
                maskReg = MicroAPI::MoveMask<T>();
            } else {
                sreg = static_cast<uint32_t>(mask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
        }
        MicroAPI::MaskReg selMask;
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
    } else {
        MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        if constexpr (isCounterMode) {
            sreg = static_cast<uint32_t>(mask);
            newRepeatTimes = CeilDivision(sreg, oneRepSize);
        } else {
            if constexpr (isBitMap) {
                maskReg = MicroAPI::MoveMask<T>();
            } else {
                sreg = static_cast<uint32_t>(mask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
        }
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = newRepeatTimes % unRollConstant;
        newRepeatTimes = newRepeatTimes / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &) src1);
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * unRollConstant * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src0 + (i * unRollConstant + 1) * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
        MicroAPI::RegTensor<T> src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
        uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(tailSreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src2Reg, src0 + offset0, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
            MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + offset1, dst2Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        SelectMode1Level0<T, U, false, true>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
    } else {
        SelectMode1Level0<T, U, false, false>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    SetVectorMask<T>(mask[1], mask[0]);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        SelectMode1Level0<T, U, true, true>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    } else {
        SelectMode1Level0<T, U, true, false>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    }
}
// ===============  Src0 Scalar =====================
template <typename T, typename U, bool isBitMap, bool isCounterMode>
__simd_vf__ inline void SelectSrc0ScalarMode1Level0(__ubuf__ T* dst, __ubuf__ U* sel, T src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    uint32_t sreg;
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = repeatTime;
    MicroAPI::MaskReg maskReg;
    if constexpr (isCounterMode) {
        sreg = static_cast<uint32_t>(mask);
        newRepeatTimes = CeilDivision(sreg, oneRepSize);
    } else {
        if constexpr (isBitMap) {
            maskReg = MicroAPI::MoveMask<T>();
        } else {
            sreg = static_cast<uint32_t>(mask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
    }
    if constexpr (sizeof(T) == 2) {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::Duplicate(src0Reg, (const T &) src0);
        MicroAPI::MaskReg selMask;
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
    } else {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = newRepeatTimes % unRollConstant;
        newRepeatTimes = newRepeatTimes / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &) src0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src1 + i * unRollConstant * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dst0Reg, scalarReg, src0Reg, selMask0);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + (i * unRollConstant + 1) * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dst1Reg, scalarReg, src1Reg, selMask1);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
        MicroAPI::RegTensor<T> src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src1RepStride * blockElm;
        uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(tailSreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src2Reg, src1 + offset0, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
            MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + offset1, dst2Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, T src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float,  bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        SelectSrc0ScalarMode1Level0<T, U, false, true>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
    } else {
        SelectSrc0ScalarMode1Level0<T, U, false, false>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, T src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float,  bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    SetVectorMask<T>(mask[1], mask[0]);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        SelectSrc0ScalarMode1Level0<T, U, true, true>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    } else {
        SelectSrc0ScalarMode1Level0<T, U, true, false>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    }
}

// both src0 / src1 are tensor
template <typename T, typename U, bool isBitMap, uint8_t scalarIdx, MicroAPI::LoadDist pattern, bool isCounterMode>
__simd_vf__ inline void SelectBothTensorMode1Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t newRepeatTimes = repeatTime;
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint32_t sreg;
    MicroAPI::MaskReg maskReg, selMask;
    if constexpr (isCounterMode) {
        sreg = static_cast<uint32_t>(mask);
        newRepeatTimes = CeilDivision(sreg, oneRepSize);
    } else {
        if constexpr (isBitMap) {
            maskReg = MicroAPI::MoveMask<T>();
        } else {
            sreg = static_cast<uint32_t>(mask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
    }
    if constexpr (scalarIdx == 0) {
        if constexpr (sizeof(T) == 2) {
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
            MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            }
        } else {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = newRepeatTimes % unRollConstant;
            newRepeatTimes = newRepeatTimes / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src0Reg, src1 + i * unRollConstant * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
                MicroAPI::Select(dst0Reg, scalarReg, src0Reg, selMask0);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src1Reg, src1 + (i * unRollConstant + 1) * blockElm * repeatParams.src1RepStride, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
                MicroAPI::Select(dst1Reg, scalarReg, src1Reg, selMask1);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            }
            MicroAPI::RegTensor<T> src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
            uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
            for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(tailSreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src2Reg, src1 + offset0, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
                MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + offset1, dst2Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            }
        }
    } else if constexpr (scalarIdx == 1) {
        if constexpr (sizeof(T) == 2) {
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
            MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
            for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * blockElm * repeatParams.dstRepStride, dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            }
        } else {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = newRepeatTimes % unRollConstant;
            newRepeatTimes = newRepeatTimes / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src1);
            for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src0Reg, src0 + i * unRollConstant * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
                MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src1Reg, src0 + (i * unRollConstant + 1) * blockElm * repeatParams.src0RepStride, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
                MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            }
            MicroAPI::RegTensor<T> src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
            uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
            for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(tailSreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src2Reg, src0 + offset0, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
                MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + offset1, dst2Reg, static_cast<uint32_t>(repeatParams.dstBlkStride), maskReg);
            }
        }
    }
}

template <typename T, typename U, uint8_t scalarIdx>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if constexpr (sizeof(T) == 2) {
            SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, true>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, true>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        }
    } else {
        if constexpr (sizeof(T) == 2) {
            SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, false>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, false>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        }
    }
}

template <typename T, typename U, uint8_t scalarIdx>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, bfloat16_t>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    SetVectorMask<T>(mask[1], mask[0]);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if constexpr (sizeof(T) == 2) {
            SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, true>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, true>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    } else {
        if constexpr (sizeof(T) == 2) {
            SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, false>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, false>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    }
}


// ============ select mode: 0/2 ============
// =============== LEVEL2 ===================
template <typename T, typename U, typename RegT>
__simd_vf__ inline void SelectMode0Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * RegT::trait.REG_NUM;
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg, tmpMask;
        MicroAPI::RegTensor<uint32_t> selReg;
        MicroAPI::UnalignReg ureg;
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint32_t *)sel);
        MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint32_t *)sel);
        MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, selReg);
        MicroAPI::MaskInterleave<uint32_t>(selMask, tmpMask, selMask, selMask);
        MicroAPI::MaskDeInterleave<uint32_t>(selMask, tmpMask, selMask, selMask);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            maskReg = MicroAPI::UpdateMask<T, RegT::trait>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else {
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg maskReg, selMask;
        if constexpr (sizeof(T) == 1) {
            MicroAPI::LoadAlign<U>(selMask, sel);
        } else {
            MicroAPI::LoadAlign<U, MicroAPI::MaskDist::DIST_US>(selMask, sel);
            if constexpr (sizeof(T) == 4) {
                MicroAPI::MaskUnPack(selMask, selMask);
            }
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <typename T, typename U, typename RegT>
__simd_vf__ inline void SelectMode2Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * RegT::trait.REG_NUM;
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * RegT::trait.REG_NUM;
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint8_t> selReg;
        MicroAPI::UnalignReg ureg;
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
            maskReg = MicroAPI::UpdateMask<T, RegT::trait>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        RegT src0Reg, src1Reg, src2Reg, src3Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = repeatTime % unRollConstant;
        uint16_t newRepeatTimes = repeatTime / unRollConstant;
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * unRollConstant * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * unRollConstant * repeatElm);
            MicroAPI::Select(dst0Reg, src0Reg, src1Reg, selMask0);
            MicroAPI::StoreAlign<T>(dst + i * unRollConstant * repeatElm, dst0Reg, maskReg);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src2Reg, src0 + (i * unRollConstant + 1) * repeatElm);
            MicroAPI::LoadAlign<T>(src3Reg, src1 + (i * unRollConstant + 1) * repeatElm);
            MicroAPI::Select(dst1Reg, src2Reg, src3Reg, selMask1);
            MicroAPI::StoreAlign<T>(dst + (i * unRollConstant + 1) * repeatElm, dst1Reg, maskReg);
        }
        RegT src4Reg, src5Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src4Reg, src0 + offset);
            MicroAPI::LoadAlign<T>(src5Reg, src1 + offset);
            MicroAPI::Select(dst2Reg, src4Reg, src5Reg, selMask2);
            MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg maskReg, selMask;
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            if constexpr (sizeof(T) == 2) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            } else {
                MicroAPI::LoadAlign<uint8_t>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            }
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float, uint64_t, int64_t, complex32, complex64>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        if constexpr (sizeof(T) == 8) {
            SelectMode0Level2<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>(dst, sel, src0, src1, calCount);
        } else {
            SelectMode0Level2<T, U, MicroAPI::RegTensor<T>>(dst, sel, src0, src1, calCount);
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        if constexpr (sizeof(T) == 8) {
            SelectMode2Level2<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>(dst, sel, src0, src1, calCount);
        } else {
            SelectMode2Level2<T, U, MicroAPI::RegTensor<T>>(dst, sel, src0, src1, calCount);
        }
    }
}

// ============ select mode: 1 ============
// =============== LEVEL2 ===================
template <typename T, typename U, typename RegT>
__simd_vf__ inline void SelectMode1Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * RegT::trait.REG_NUM;
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * RegT::trait.REG_NUM;
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint8_t> selReg;
        MicroAPI::Duplicate(src1Reg, (const T &)src1);
        MicroAPI::UnalignReg ureg;
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
            maskReg = MicroAPI::UpdateMask<T, RegT::trait>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        RegT src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = repeatTime % unRollConstant;
        uint16_t newRepeatTimes = repeatTime / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &)src1);
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * unRollConstant * repeatElm);
            MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
            MicroAPI::StoreAlign<T>(dst + i * unRollConstant * repeatElm, dst0Reg, maskReg);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src1Reg, src0 + (i * unRollConstant + 1) * repeatElm);
            MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
            MicroAPI::StoreAlign<T>(dst + (i * unRollConstant + 1) * repeatElm, dst1Reg, maskReg);
        }
        RegT src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src2Reg, src0 + offset);
            MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
            MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
        RegT src0Reg, src1Reg, dstReg;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        MicroAPI::MaskReg maskReg, selMask;
        MicroAPI::Duplicate(src1Reg, (const T &)src1);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            if constexpr (sizeof(T) == 2) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            } else {
                MicroAPI::LoadAlign<uint8_t>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            }
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float, uint64_t, int64_t, complex32, complex64>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    if constexpr (sizeof(T) == 8) {
        SelectMode1Level2<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>(dst, sel, src0, src1, calCount);
    } else {
        SelectMode1Level2<T, U, MicroAPI::RegTensor<T>>(dst, sel, src0, src1, calCount);
    }
}
// Src0Scalar
template <typename T, typename U, typename RegT>
__simd_vf__ inline void SelectSrc0ScalarMode1Level2(__ubuf__ T* dst, __ubuf__ U* sel, T src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * RegT::trait.REG_NUM;
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * RegT::trait.REG_NUM;
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint8_t> selReg;
        MicroAPI::Duplicate(src0Reg, (const T &)src0);
        MicroAPI::UnalignReg ureg;
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
            maskReg = MicroAPI::UpdateMask<T, RegT::trait>(sreg);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        RegT src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = repeatTime % unRollConstant;
        uint16_t newRepeatTimes = repeatTime / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &)src0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src1 + i * unRollConstant * repeatElm);
            MicroAPI::Select(dst0Reg, scalarReg, src0Reg, selMask0);
            MicroAPI::StoreAlign<T>(dst + i * unRollConstant * repeatElm, dst0Reg, maskReg);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + (i * unRollConstant + 1) * repeatElm);
            MicroAPI::Select(dst1Reg, scalarReg, src1Reg, selMask1);
            MicroAPI::StoreAlign<T>(dst + (i * unRollConstant + 1) * repeatElm, dst1Reg, maskReg);
        }
        RegT src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src2Reg, src1 + offset);
            MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
            MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
        RegT src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg maskReg, selMask;
        MicroAPI::Duplicate(src0Reg, (const T &)src0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            if constexpr (sizeof(T) == 2) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            } else {
                MicroAPI::LoadAlign<uint8_t>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            }
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, T src0,__ubuf__ T* src1, 
    SELMODE selMode, uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float, uint64_t, int64_t, complex32, complex64>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    if constexpr (sizeof(T) == 8) {
        SelectSrc0ScalarMode1Level2<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>(dst, sel, src0, src1, calCount);
    } else {
        SelectSrc0ScalarMode1Level2<T, U, MicroAPI::RegTensor<T>>(dst, sel, src0, src1, calCount);
    }
}
// both src0 / src1 Tensor
template <typename T, typename U, typename RegT, uint8_t scalarIdx, MicroAPI::LoadDist pattern = MicroAPI::LoadDist::DIST_BRC_B32>
__simd_vf__ inline void SelectBothTensorMode1Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * RegT::trait.REG_NUM;
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (scalarIdx == 0) {
        if constexpr (sizeof(T) == 8) {
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * RegT::trait.REG_NUM;
            RegT src0Reg, src1Reg, dstReg, tmpReg;
            MicroAPI::MaskReg selMask, maskReg;
            MicroAPI::RegTensor<uint8_t> selReg;
            MicroAPI::UnalignReg ureg, uregDup;
            // Unalign DataCopy do not support TraitNumTwo right now
            MicroAPI::LoadUnAlignPre(uregDup, (__ubuf__ T *)src0);
            MicroAPI::LoadUnAlign(tmpReg, uregDup, (__ubuf__ T *)src0);
            MicroAPI::DeInterleave<uint32_t>((MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[0], (MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[1], 
                (MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[0], (MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[0]);
            MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(src0Reg, tmpReg, maskFull);
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel);
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
                MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
                maskReg = MicroAPI::UpdateMask<T, RegT::trait>(sreg);
                MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
            }
        } else if constexpr (sizeof(T) == 4) {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            RegT src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg maskReg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = repeatTime % unRollConstant;
            uint16_t newRepeatTimes = repeatTime / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src0Reg, src1 + i * unRollConstant * repeatElm);
                MicroAPI::Select(dst0Reg, scalarReg, src0Reg, selMask0);
                MicroAPI::StoreAlign<T>(dst + i * unRollConstant * repeatElm, dst0Reg, maskReg);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src1Reg, src1 + (i * unRollConstant + 1) * repeatElm);
                MicroAPI::Select(dst1Reg, scalarReg, src1Reg, selMask1);
                MicroAPI::StoreAlign<T>(dst + (i * unRollConstant + 1) * repeatElm, dst1Reg, maskReg);
            }
            RegT src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src2Reg, src1 + offset);
                MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
                MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
            }
        } else {
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
            RegT src0Reg, src1Reg, dstReg;
            MicroAPI::MaskReg maskReg, selMask;
            MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
                if constexpr (sizeof(T) == 2) {
                    MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                } else {
                    MicroAPI::LoadAlign<uint8_t>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                }
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
            }
        }
    } else if constexpr (scalarIdx == 1) {
        if constexpr (sizeof(T) == 8) {
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * RegT::trait.REG_NUM;
            RegT src0Reg, src1Reg, dstReg, tmpReg;
            MicroAPI::MaskReg selMask, maskReg;
            MicroAPI::RegTensor<uint8_t> selReg;
            MicroAPI::UnalignReg ureg, uregDup;
            MicroAPI::LoadUnAlignPre(uregDup, (__ubuf__ T *)src1);
            MicroAPI::LoadUnAlign(tmpReg, uregDup, (__ubuf__ T *)src1);
            MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DeInterleave<uint32_t>((MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[0], (MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[1], 
                (MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[0], (MicroAPI::RegTensor<uint32_t>&)tmpReg.reg[0]);
            MicroAPI::Duplicate(src1Reg, tmpReg, maskFull);
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel);
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
                MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
                maskReg = MicroAPI::UpdateMask<T, RegT::trait>(sreg);
                MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
            }
        } else if constexpr (sizeof(T) == 4) {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            RegT src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg maskReg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = repeatTime % unRollConstant;
            uint16_t newRepeatTimes = repeatTime / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src1);
            for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src0Reg, src0 + i * unRollConstant * repeatElm);
                MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
                MicroAPI::StoreAlign<T>(dst + i * unRollConstant * repeatElm, dst0Reg, maskReg);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src1Reg, src0 + (i * unRollConstant + 1) * repeatElm);
                MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
                MicroAPI::StoreAlign<T>(dst + (i * unRollConstant + 1) * repeatElm, dst1Reg, maskReg);
            }
            RegT src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src2Reg, src0 + offset);
                MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
                MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
            }
        } else {
            constexpr uint32_t selOffset = GetVecLen() / SelInternal::maskBitToByte / sizeof(T);
            RegT src0Reg, src1Reg, dstReg;
            MicroAPI::MaskReg maskReg, selMask;
            MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
                if constexpr (sizeof(T) == 2) {
                    MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                } else {
                    MicroAPI::LoadAlign<uint8_t>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                }
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
            }
        }
    }
}

template <typename T, typename U, uint8_t scalarIdx>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,__ubuf__ T* src1, 
    SELMODE selMode, uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float, uint64_t, int64_t, complex32, complex64>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>(), "current data type is not supported!");
    if constexpr (sizeof(T) == 1) {
        SelectBothTensorMode1Level2<T, U, MicroAPI::RegTensor<T>, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B8>(dst, sel, src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        SelectBothTensorMode1Level2<T, U, MicroAPI::RegTensor<T>, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16>(dst, sel, src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        SelectBothTensorMode1Level2<T, U, MicroAPI::RegTensor<T>, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32>(dst, sel, src0, src1, calCount);
    } else {
        SelectBothTensorMode1Level2<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32>(dst, sel, src0, src1, calCount);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_SEL_IMPL_H
