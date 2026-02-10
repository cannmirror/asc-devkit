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
 * \file kernel_operator_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H

#include "kernel_utils.h"
#include "kernel_tpipe.h"
#include "kernel_operator_vec_template_impl.h"
#include "micro_api/kernel_micro_intf.h"

namespace AscendC {
namespace CmpSelInternal {
    constexpr uint32_t maskBitToByte = 8;
}
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
// Compare written to CMPMASK
template <typename T, bool isSetMask, CMPMODE cmpMode>
__aicore__ inline void CompareWithoutDstCounterModeImplVF(
    __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t* tempBuf, const uint64_t mask, const BinaryRepeatParams &repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::MaskReg maskReg, dstReg;
    MicroAPI::UnalignReg uReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
    }
    maskReg = MicroAPI::UpdateMask<T>(sreg);
    MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0, src0, (uint32_t)repeatParams.src0BlkStride, maskReg);
    MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1, src1, (uint32_t)repeatParams.src1BlkStride, maskReg);
    MicroAPI::Compare<T, cmpMode>(dstReg, srcReg0, srcReg1, maskReg);
    MicroAPI::StoreUnAlign((__ubuf__ T *&)tempBuf, dstReg, uReg);
    MicroAPI::StoreUnAlignPost<uint64_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(tempBuf, uReg, 0);
}

template <typename T, bool isSetMask, bool isBitMap, CMPMODE cmpMode>
__aicore__ inline void CompareWithoutDstNormalModeImplVF(
    __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t* tempBuf, const uint64_t mask, const BinaryRepeatParams &repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::MaskReg maskReg, dstReg;
    MicroAPI::UnalignReg uReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    if constexpr (isBitMap) {
        maskReg = MicroAPI::MoveMask<T>();
    } else {
        if constexpr (isSetMask) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        } else {
            maskReg = MicroAPI::MoveMask<T>();
        }
    }
    MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0, src0, (uint32_t)repeatParams.src0BlkStride, maskReg);
    MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1, src1, (uint32_t)repeatParams.src1BlkStride, maskReg);
    MicroAPI::Compare<T, cmpMode>(dstReg, srcReg0, srcReg1, maskReg);
    MicroAPI::StoreUnAlign((__ubuf__ T *&)tempBuf, dstReg, uReg);
    MicroAPI::StoreUnAlignPost<uint64_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(tempBuf, uReg, 0);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(
    __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint64_t mask[], const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
    if (isCounterMode) {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LT>>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GT>>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::EQ>>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LE>>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GE>>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::NE>>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            default:
                break;
        }
    } else {
        if constexpr (isSetMask) {
            SetVectorMask<T>(mask[1], mask[0]);
        }
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::LT>>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::GT>>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::EQ>>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::LE>>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::GE>>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::NE>>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            default:
                break;
        }
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    if constexpr (sizeof(T) == 2) {
        Internal::g_cmpMaskLow = static_cast<uint64_t>(tempBuf[0]);
        Internal::g_cmpMaskHigh = static_cast<uint64_t>(tempBuf[1]);
    } else if constexpr (sizeof(T) == 4) {
        Internal::g_cmpMaskLow = static_cast<uint64_t>(tempBuf[0]);
        Internal::g_cmpMaskHigh = static_cast<uint64_t>(0);
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(
    __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint64_t mask, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
    if (isCounterMode) {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LT>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GT>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::EQ>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LE>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GE>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::NE>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            default:
                break;
        }
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::LT>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::GT>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::EQ>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::LE>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::GE>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::NE>>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            default:
                break;
        }
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    if constexpr (sizeof(T) == 2) {
        Internal::g_cmpMaskLow = static_cast<uint64_t>(tempBuf[0]);
        Internal::g_cmpMaskHigh = static_cast<uint64_t>(tempBuf[1]);
    } else if constexpr (sizeof(T) == 4) {
        Internal::g_cmpMaskLow = static_cast<uint64_t>(tempBuf[0]);
        Internal::g_cmpMaskHigh = static_cast<uint64_t>(0);
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
}

// Compare::Level 0 - bit mode / Continuous mode support b16/b32
template <typename T, typename U, CMPMODE cmpMode>
__aicore__ inline void CompareLevel0CounterMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const BinaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = CeilDivision(sreg, oneRepSize);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, (uint32_t)repeatParams.src0BlkStride, (uint32_t)repeatParams.src0RepStride, maskReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, (uint32_t)repeatParams.src1BlkStride, (uint32_t)repeatParams.src1RepStride, maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

// Compare::Level 0 - bit mode / Continuous mode support b8/b16/b32
template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode>
__aicore__ inline void CompareLevel0NormalMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (isBitMapMode) {
        maskReg = MicroAPI::MoveMask<T>();
    } else {
        uint32_t sreg = static_cast<uint32_t>(mask);
        maskReg = MicroAPI::UpdateMask<T>(sreg);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, (uint32_t)repeatParams.src0BlkStride, (uint32_t)repeatParams.src0RepStride, maskReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, (uint32_t)repeatParams.src1BlkStride, (uint32_t)repeatParams.src1RepStride, maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, int8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::LT>>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::GT>>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::EQ>>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::LE>>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::GE>>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::NE>>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            default:
                break;
        }
    } else {
        if constexpr (isSetMask) {
            SetVectorMask<T>(mask[1], mask[0]);
        }
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::LT, true>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::GT, true>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::EQ, true>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::LE, true>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::GE, true>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::NE, true>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, int8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::LT>>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::GT>>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::EQ>>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::LE>>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::GE>>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareLevel0CounterMode<T, U, CMPMODE::NE>>(dst, src0, src1, mask, repeatParams);
                break;
            }
            default:
                break;
        }
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::LT, false>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::GT, false>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::EQ, false>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::LE, false>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::GE, false>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareLevel0NormalMode<T, U, CMPMODE::NE, false>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

/* ***************************************************************************************
 * ************************************** CompareScalar ****************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - bit mode / continuous mode
template <typename T, typename U, CMPMODE cmpMode, bool isSetMask>
__aicore__ inline void CompareScalarLevel0CounterMode(__ubuf__ U *dst, __ubuf__ T *src0, const T src1, const uint64_t mask, __ubuf__ uint64_t *tempBuf,
    const UnaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
    }
    uint16_t newRepeatTimes = CeilDivision(sreg, oneRepSize);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, bool isSetMask>
__aicore__ inline void CompareScalarLevel0NormalMode(__ubuf__ U *dst, __ubuf__ T *src0, const T src1,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (isBitMapMode) {
        maskReg = MicroAPI::MoveMask<T>();
    } else {
        if constexpr (isSetMask) {
            uint32_t sreg = static_cast<uint32_t>(mask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        } else {
            maskReg = MicroAPI::MoveMask<T>();
        }
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, const T src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            default:
                break;
        }
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        if constexpr (isSetMask) {
            SetVectorMask<T>(mask[1], mask[0]);
        }
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::LT, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::GT, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::EQ, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::LE, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::GE, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::NE, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, const T src1, CMPMODE cmpMode, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            default:
                break;
        }
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::LT, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::GT, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::EQ, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::LE, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::GE, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareScalarLevel0NormalMode<T, U, CMPMODE::NE, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

// CompareScalar::Level 2 - counter mode
template <typename T, typename U, CMPMODE cmpMode>
__aicore__ inline void CompareScalarLevel2(__ubuf__ U *dst, __ubuf__ T *src0, const T src1, const uint32_t calCount)
{
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg;
    uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        repeatElm = repeatElm * 2;
        repeatTime = CeilDivision(calCount, repeatElm);
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
            MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
            MicroAPI::StoreUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
        }
        MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
    } else {
        MicroAPI::RegTensor<T> src0Reg;
        constexpr uint32_t offset = GetVecLen() / sizeof(T) / CmpSelInternal::maskBitToByte;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
            MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::StoreAlign(dst + i * offset, dstReg);
            } else {
                MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
            }
        }
        if constexpr (sizeof(T) > 1) {
            MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(
    __ubuf__ U *dst, __ubuf__ T *src0, const T src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float, 
        uint64_t, int64_t>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    switch (cmpMode) {
        case CMPMODE::LT: {
            VF_CALL<CompareScalarLevel2<T, U, CMPMODE::LT>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            VF_CALL<CompareScalarLevel2<T, U, CMPMODE::GT>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            VF_CALL<CompareScalarLevel2<T, U, CMPMODE::EQ>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            VF_CALL<CompareScalarLevel2<T, U, CMPMODE::LE>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            VF_CALL<CompareScalarLevel2<T, U, CMPMODE::GE>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            VF_CALL<CompareScalarLevel2<T, U, CMPMODE::NE>>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}

/* ***************************************************************************************
 * ************************************** CompareScalar src0 scalar****************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - bit mode / continuous mode
template <typename T, typename U, CMPMODE cmpMode, bool isSetMask>
__aicore__ inline void CompareSrc0ScalarLevel0CounterMode(__ubuf__ U *dst, const T src0, __ubuf__ T *src1, const uint64_t mask,
    __ubuf__ uint64_t *tempBuf, const UnaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    MicroAPI::Duplicate(src0Reg, src0);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
    }
    uint16_t newRepeatTimes = CeilDivision(sreg, oneRepSize);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, bool isSetMask>
__aicore__ inline void CompareSrc0ScalarLevel0NormalMode(__ubuf__ U *dst, const T src0, __ubuf__ T *src1,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (isBitMapMode) {
        maskReg = MicroAPI::MoveMask<T>();
    } else {
        if constexpr (isSetMask) {
            uint32_t sreg = static_cast<uint32_t>(mask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        } else {
            maskReg = MicroAPI::MoveMask<T>();
        }
    }
    MicroAPI::Duplicate(src0Reg, src0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, const T src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            default:
                break;
        }
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        if constexpr (isSetMask) {
            SetVectorMask<T>(mask[1], mask[0]);
        }
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LT, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GT, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::EQ, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LE, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GE, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::NE, true, isSetMask>>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, const T src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
         __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            default:
                break;
        }
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LT, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GT, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::EQ, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LE, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GE, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::NE, false, isSetMask>>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

// CompareScalar::Level 2 - counter mode

template <typename T, typename U, CMPMODE cmpMode>
__aicore__ inline void CompareSrc0ScalarLevel2(__ubuf__ U *dst, const T src0, __ubuf__ T *src1, const uint32_t calCount)
{
    uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (sizeof(T) == 8) {
        repeatElm = repeatElm * 2;
        repeatTime = CeilDivision(calCount, repeatElm);
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg, src1Reg;
        MicroAPI::Duplicate(src0Reg, src0);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
            MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
            MicroAPI::StoreUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
        }
        MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
    } else {
        MicroAPI::RegTensor<T> src0Reg, src1Reg;
        constexpr uint32_t offset = GetVecLen() / sizeof(T) / CmpSelInternal::maskBitToByte;
        MicroAPI::Duplicate(src0Reg, src0);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
            MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::StoreAlign(dst + i * offset, dstReg);
            } else {
                MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
            }
        }
        if constexpr (sizeof(T) > 1) {
            MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(
    __ubuf__ U *dst, const T src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float,
        uint64_t, int64_t>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    switch (cmpMode) {
        case CMPMODE::LT: {
            VF_CALL<CompareSrc0ScalarLevel2<T, U, CMPMODE::LT>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            VF_CALL<CompareSrc0ScalarLevel2<T, U, CMPMODE::GT>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            VF_CALL<CompareSrc0ScalarLevel2<T, U, CMPMODE::EQ>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            VF_CALL<CompareSrc0ScalarLevel2<T, U, CMPMODE::LE>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            VF_CALL<CompareSrc0ScalarLevel2<T, U, CMPMODE::GE>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            VF_CALL<CompareSrc0ScalarLevel2<T, U, CMPMODE::NE>>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}
/* ***************************************************************************************
 * ************************************** CompareScalar Both Tensor****************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - bit mode / continuous mode

template <typename T, typename U, CMPMODE cmpMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern>
__aicore__ inline void CompareScalarBothTensorLevel2(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg;
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    constexpr uint32_t offset = GetVecLen() / sizeof(T) / CmpSelInternal::maskBitToByte;
    if constexpr (scalarIdx == 0) {
        MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
    } else {
        MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
        } else {
            MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        if constexpr (sizeof(T) == 1) {
            MicroAPI::StoreAlign(dst + i * offset, dstReg);
        } else {
            MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
        }
    }
    if constexpr (sizeof(T) > 1) {
        MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
    }
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1, MicroAPI::LoadDist pattern>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            VF_CALL<CompareScalarBothTensorLevel2<T, U, CMPMODE::LT, scalarIdx, pattern>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            VF_CALL<CompareScalarBothTensorLevel2<T, U, CMPMODE::GT, scalarIdx, pattern>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            VF_CALL<CompareScalarBothTensorLevel2<T, U, CMPMODE::EQ, scalarIdx, pattern>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            VF_CALL<CompareScalarBothTensorLevel2<T, U, CMPMODE::LE, scalarIdx, pattern>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            VF_CALL<CompareScalarBothTensorLevel2<T, U, CMPMODE::GE, scalarIdx, pattern>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            VF_CALL<CompareScalarBothTensorLevel2<T, U, CMPMODE::NE, scalarIdx, pattern>>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, CMPMODE cmpMode, uint8_t scalarIdx>
__aicore__ inline void CompareScalarLevel2B64(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * 2;
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg, dupuReg;
    repeatTime = CeilDivision(calCount, repeatElm);
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg, src1Reg, dupReg;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> preReg;

    MicroAPI::RegTensor<uint32_t> zeroReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0, maskFull);

    if constexpr (scalarIdx == 0) {
        MicroAPI::LoadUnAlignPre(dupuReg, (__ubuf__ T *)src0);
        MicroAPI::LoadUnAlign(preReg, dupuReg, (__ubuf__ T *)src0);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1],
            (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(src0Reg, dupReg, maskFull);
    } else {
        MicroAPI::LoadUnAlignPre(dupuReg, (__ubuf__ T *)src1);
        MicroAPI::LoadUnAlign(preReg, dupuReg, (__ubuf__ T *)src1);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1],
            (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(src1Reg, dupReg, maskFull);
    }

    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
        } else {
            MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void VcmpvsImplB64(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            VF_CALL<CompareScalarLevel2B64<T, U, CMPMODE::LT, scalarIdx>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            VF_CALL<CompareScalarLevel2B64<T, U, CMPMODE::GT, scalarIdx>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            VF_CALL<CompareScalarLevel2B64<T, U, CMPMODE::EQ, scalarIdx>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            VF_CALL<CompareScalarLevel2B64<T, U, CMPMODE::LE, scalarIdx>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            VF_CALL<CompareScalarLevel2B64<T, U, CMPMODE::GE, scalarIdx>>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            VF_CALL<CompareScalarLevel2B64<T, U, CMPMODE::NE, scalarIdx>>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void VcmpvsImpl(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float,
        uint64_t, int64_t>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");

    if constexpr (sizeof(T) == 1) {
        VcmpvsImpl<T, U, isSetMask, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B8>(dst, src0, src1, cmpMode, calCount);
    } else if constexpr (sizeof(T) == 2) {
        VcmpvsImpl<T, U, isSetMask, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16>(dst, src0, src1, cmpMode, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VcmpvsImpl<T, U, isSetMask, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32>(dst, src0, src1, cmpMode, calCount);
    } else if constexpr (sizeof(T) == 8) {
        VcmpvsImplB64<T, U, isSetMask, scalarIdx>(dst, src0, src1, cmpMode, calCount);
    }
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern, bool isSetMask>
__aicore__ inline void CompareScalarBothTensorLevel0CounterMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, __ubuf__ uint64_t *tempBuf, const UnaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    uint32_t countSreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        countSreg = static_cast<uint32_t>(tempBuf[0]);
    }
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = CeilDivision(countSreg, oneRepSize);
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (scalarIdx == 0) {
        MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
    } else {
        MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(countSreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src1Reg, src1, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        } else {
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src0Reg, src0, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern, bool isSetMask>
__aicore__ inline void CompareScalarBothTensorLevel0NormalMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (isBitMapMode) {
        maskReg = MicroAPI::MoveMask<T>();
    } else {
        if constexpr (isSetMask) {
            uint32_t sreg = static_cast<uint32_t>(mask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        } else {
            maskReg = MicroAPI::MoveMask<T>();
        }
    }
    if constexpr (scalarIdx == 0) {
        MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
    } else {
        MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        if constexpr (scalarIdx == 0) {
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src1Reg, src1, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        } else {
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src0Reg, src0, (uint32_t)repeatParams.srcBlkStride, (uint32_t)repeatParams.srcRepStride, maskReg);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::StoreUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::StoreUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true, bool isBitMapMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::LT, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::GT, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::EQ, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::LE, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::GE, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::NE, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            default:
                break;
        }
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                VF_CALL<CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::LT, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                VF_CALL<CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::GT, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                VF_CALL<CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::EQ, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                VF_CALL<CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::LE, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                VF_CALL<CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::GE, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                VF_CALL<CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::NE, isBitMapMode, scalarIdx, pattern, isSetMask>>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            default:
                break;
        }
    }
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, uint8_t, int8_t, half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    bool isCounterMode = Internal::IsCounterMode();
    if (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    if constexpr (sizeof(T) == 2) {
        VcmpvsImpl<T, U, isSetMask, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16>(dst, src0, src1, cmpMode, mask[0], repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        VcmpvsImpl<T, U, isSetMask, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32>(dst, src0, src1, cmpMode, mask[0], repeatTime, repeatParams);
    }
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float>(),
        "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    if constexpr (sizeof(T) == 2) {
        VcmpvsImpl<T, U, isSetMask, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16>(dst, src0, src1, cmpMode, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        VcmpvsImpl<T, U, isSetMask, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32>(dst, src0, src1, cmpMode, mask, repeatTime, repeatParams);
    }
}

/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */


template <typename T, bool isCounterMode>
__aicore__ inline void SelectWithoutMaskMode0ImplVF(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t *tempBuf, int32_t repeat, const BinaryRepeatParams &repeatParams)
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
__aicore__ inline void SelectWithoutMaskMode2ImplVF(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t *tempBuf, uint64_t selAddr, int32_t repeat, const BinaryRepeatParams &repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    MicroAPI::MaskReg maskReg, selMask;
    MicroAPI::RegTensor<uint8_t> selReg;
    MicroAPI::UnalignReg ureg;
    uint16_t newRepeatTimes = repeat;
    constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
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
            VF_CALL<SelectWithoutMaskMode0ImplVF<T, true>>(dst, src0, src1, tempBuf, repeat, repeatParams);
        } else {
            VF_CALL<SelectWithoutMaskMode0ImplVF<T, false>>(dst, src0, src1, tempBuf, repeat, repeatParams);
        }
    }
    else if constexpr (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        uint64_t selAddr = Internal::g_cmpMaskLow;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (isCounterMode) {
            VF_CALL<SelectWithoutMaskMode2ImplVF<T, true>>(dst, src0, src1, tempBuf, selAddr, repeat, repeatParams);
        } else {
            VF_CALL<SelectWithoutMaskMode2ImplVF<T, false>>(dst, src0, src1, tempBuf, selAddr, repeat, repeatParams);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
}

template <typename T, typename U, bool isCounterMode>
__aicore__ inline void SelectWithoutMaskMode1ImplVF(
    __ubuf__ T *dst, __ubuf__ U *sel, __ubuf__ T *src0, T scalar, __ubuf__ uint64_t *tempBuf, int32_t repeat, const BinaryRepeatParams &repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    MicroAPI::MaskReg maskReg, selMask;
    MicroAPI::RegTensor<uint8_t> selReg;
    MicroAPI::UnalignReg ureg;
    uint16_t newRepeatTimes = repeat;
    uint32_t sreg;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
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
        VF_CALL<SelectWithoutMaskMode1ImplVF<T, U, true>>(dst, sel, src0, scalar, tempBuf, repeat, repeatParams);
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        VF_CALL<SelectWithoutMaskMode1ImplVF<T, U, false>>(dst, sel, src0, scalar, nullptr, repeat, repeatParams);
    }
}

// ============ select mode: 0/2 ============
// ================Level2====================
template <typename T, typename U, bool isBitMap, bool isCounterMode>
__aicore__ inline void SelectMode0Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams) {
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
    for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
        MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + i * blockElm * repeatParams.dstRepStride, dstReg, (uint32_t)repeatParams.dstBlkStride, maskReg);
    }
}

template <typename T, typename U, bool isBitMap, bool isCounterMode>
__aicore__ inline void SelectMode2Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = repeatTime;
    uint32_t sreg;
    if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
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
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * unRollConstant * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + i * unRollConstant * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dst0Reg, src0Reg, src1Reg, selMask0);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src2Reg, src0 + (i * unRollConstant + 1) * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src3Reg, src1 + (i * unRollConstant + 1) * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dst1Reg, src2Reg, src3Reg, selMask1);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
        }
        MicroAPI::RegTensor<T> src4Reg, src5Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
        uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.src1RepStride * blockElm;
        uint32_t offset2 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset; 
        uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
        for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(tailSreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src4Reg, src0 + offset0, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src5Reg, src1 + offset1, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dst2Reg, src4Reg, src5Reg, selMask2);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + offset2, dst2Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
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
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * blockElm * repeatParams.dstRepStride, dstReg, (uint32_t)repeatParams.dstBlkStride, maskReg);
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
            VF_CALL<SelectMode0Level0<T, U, false, true>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            VF_CALL<SelectMode2Level0<T, U, false, true>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        }
    } else {
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            VF_CALL<SelectMode0Level0<T, U, false, false>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            VF_CALL<SelectMode2Level0<T, U, false, false>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
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
            VF_CALL<SelectMode0Level0<T, U, true, true>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            VF_CALL<SelectMode2Level0<T, U, true, true>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    } else {
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            VF_CALL<SelectMode0Level0<T, U, true, false>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            VF_CALL<SelectMode2Level0<T, U, true, false>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    }
}
// ============ select mode: 1 ============
// ================Level0====================

template <typename T, typename U, bool isBitMap, bool isCounterMode>
__aicore__ inline void SelectMode1Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams) {
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = repeatTime;
    uint32_t sreg;
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::Duplicate(src1Reg, (const T &) src1);
        MicroAPI::MaskReg maskReg;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
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
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * blockElm * repeatParams.dstRepStride, dstReg, (uint32_t)repeatParams.dstBlkStride, maskReg);
        }
    } else {
        MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
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
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src0 + i * unRollConstant * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src0 + (i * unRollConstant + 1) * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
        }
        MicroAPI::RegTensor<T> src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
        uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
        for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(tailSreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src2Reg, src0 + offset0, (uint32_t)repeatParams.src0BlkStride, maskReg);
            MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + offset1, dst2Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
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
        VF_CALL<SelectMode1Level0<T, U, false, true>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
    } else {
        VF_CALL<SelectMode1Level0<T, U, false, false>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
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
        VF_CALL<SelectMode1Level0<T, U, true, true>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    } else {
        VF_CALL<SelectMode1Level0<T, U, true, false>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    }
}
// ===============  Src0 Scalar =====================
template <typename T, typename U, bool isBitMap, bool isCounterMode>
__aicore__ inline void SelectSrc0ScalarMode1Level0(__ubuf__ T* dst, __ubuf__ U* sel, T src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams) {
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
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::Duplicate(src0Reg, (const T &) src0);
        MicroAPI::MaskReg selMask;
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * blockElm * repeatParams.dstRepStride, dstReg, (uint32_t)repeatParams.dstBlkStride, maskReg);
        }
    } else {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = newRepeatTimes % unRollConstant;
        newRepeatTimes = newRepeatTimes / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &) src0);
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src0Reg, src1 + i * unRollConstant * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dst0Reg, scalarReg, src0Reg, selMask0);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            }
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src1Reg, src1 + (i * unRollConstant + 1) * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dst1Reg, scalarReg, src1Reg, selMask1);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
        }
        MicroAPI::RegTensor<T> src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src1RepStride * blockElm;
        uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
        for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
            if constexpr (isCounterMode) {
                maskReg = MicroAPI::UpdateMask<T>(tailSreg);
            }
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                src2Reg, src1 + offset0, (uint32_t)repeatParams.src1BlkStride, maskReg);
            MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
            MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                dst + offset1, dst2Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
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
        VF_CALL<SelectSrc0ScalarMode1Level0<T, U, false, true>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
    } else {
        VF_CALL<SelectSrc0ScalarMode1Level0<T, U, false, false>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
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
        VF_CALL<SelectSrc0ScalarMode1Level0<T, U, true, true>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    } else {
        VF_CALL<SelectSrc0ScalarMode1Level0<T, U, true, false>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
    }
}

// both src0 / src1 are tensor
template <typename T, typename U, bool isBitMap, uint8_t scalarIdx, MicroAPI::LoadDist pattern, bool isCounterMode>
__aicore__ inline void SelectBothTensorMode1Level0(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams) {
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
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
            MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src1Reg, src1 + i * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * blockElm * repeatParams.dstRepStride, dstReg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            }
        } else {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = newRepeatTimes % unRollConstant;
            newRepeatTimes = newRepeatTimes / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src0);
            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src0Reg, src1 + i * unRollConstant * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
                MicroAPI::Select(dst0Reg, scalarReg, src0Reg, selMask0);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src1Reg, src1 + (i * unRollConstant + 1) * blockElm * repeatParams.src1RepStride, (uint32_t)repeatParams.src1BlkStride, maskReg);
                MicroAPI::Select(dst1Reg, scalarReg, src1Reg, selMask1);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            }
            MicroAPI::RegTensor<T> src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
            uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
            for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(tailSreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src2Reg, src1 + offset0, (uint32_t)repeatParams.src1BlkStride, maskReg);
                MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + offset1, dst2Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            }
        }
    } else if constexpr (scalarIdx == 1) {
        if constexpr (sizeof(T) == 2) {
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
            MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src0Reg, src0 + i * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * blockElm * repeatParams.dstRepStride, dstReg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            }
        } else {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            MicroAPI::RegTensor<T> scalarReg, src0Reg, src1Reg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = newRepeatTimes % unRollConstant;
            newRepeatTimes = newRepeatTimes / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src1);
            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(tmpMask0, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src0Reg, src0 + i * unRollConstant * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
                MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + i * unRollConstant * blockElm * repeatParams.dstRepStride, dst0Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(sreg);
                }
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src1Reg, src0 + (i * unRollConstant + 1) * blockElm * repeatParams.src0RepStride, (uint32_t)repeatParams.src0BlkStride, maskReg);
                MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + (i * unRollConstant + 1) * blockElm * repeatParams.dstRepStride, dst1Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
            }
            MicroAPI::RegTensor<T> src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset0 = newRepeatTimes * unRollConstant * repeatParams.src0RepStride * blockElm;
            uint32_t offset1 = newRepeatTimes * unRollConstant * repeatParams.dstRepStride * blockElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            uint32_t tailSreg = sreg - unRollConstant * newRepeatTimes * oneRepSize;
            for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
                if constexpr (isCounterMode) {
                    maskReg = MicroAPI::UpdateMask<T>(tailSreg);
                }
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    src2Reg, src0 + offset0, (uint32_t)repeatParams.src0BlkStride, maskReg);
                MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    dst + offset1, dst2Reg, (uint32_t)repeatParams.dstBlkStride, maskReg);
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
            VF_CALL<SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, true>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            VF_CALL<SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, true>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        }
    } else {
        if constexpr (sizeof(T) == 2) {
            VF_CALL<SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, false>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            VF_CALL<SelectBothTensorMode1Level0<T, U, false, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, false>>(dst, sel, src0, src1, mask, repeatTime, repeatParams);
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
            VF_CALL<SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, true>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            VF_CALL<SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, true>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    } else {
        if constexpr (sizeof(T) == 2) {
            VF_CALL<SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16, false>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        } else if constexpr (sizeof(T) == 4) {
            VF_CALL<SelectBothTensorMode1Level0<T, U, true, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32, false>>(dst, sel, src0, src1, mask[0], repeatTime, repeatParams);
        }
    }
}


// ============ select mode: 0/2 ============
// =============== LEVEL2 ===================
template <typename T, typename U>
__aicore__ inline void SelectMode0Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint32_t> selReg;
        MicroAPI::UnalignReg ureg;
        MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint32_t *)sel);
        MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint32_t *)sel);
        MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, selReg);
        MicroAPI::MaskUnPack(selMask, selMask);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }          
    } else {
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg maskReg, selMask;
        if constexpr (sizeof(T) == 1) {
            MicroAPI::LoadAlign<U>(selMask, sel);
        } else {
            MicroAPI::LoadAlign<U, MicroAPI::MaskDist::DIST_US>(selMask, sel);
            if constexpr (sizeof(T) == 4) {
                MicroAPI::MaskUnPack(selMask, selMask);
            }
        }
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void SelectMode2Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint8_t> selReg;
        MicroAPI::UnalignReg ureg;
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
            MicroAPI::MaskUnPack(selMask, selMask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        MicroAPI::RegTensor<T> src0Reg, src1Reg, src2Reg, src3Reg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = repeatTime % unRollConstant;
        uint16_t newRepeatTimes = repeatTime / unRollConstant;
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
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
        MicroAPI::RegTensor<T> src4Reg, src5Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src4Reg, src0 + offset);
            MicroAPI::LoadAlign<T>(src5Reg, src1 + offset);
            MicroAPI::Select(dst2Reg, src4Reg, src5Reg, selMask2);
            MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        uint32_t sreg = (uint32_t)calCount;
        MicroAPI::MaskReg maskReg, selMask;
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
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
        VF_CALL<SelectMode0Level2<T, U>>(dst, sel, src0, src1, calCount);
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        VF_CALL<SelectMode2Level2<T, U>>(dst, sel, src0, src1, calCount);
    }
}

// ============ select mode: 1 ============
// =============== LEVEL2 ===================
template <typename T, typename U>
__aicore__ inline void SelectMode1Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint8_t> selReg;
        MicroAPI::Duplicate(src1Reg, (const T &)src1);
        MicroAPI::UnalignReg ureg;
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
            MicroAPI::MaskUnPack(selMask, selMask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        MicroAPI::RegTensor<T> src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = repeatTime % unRollConstant;
        uint16_t newRepeatTimes = repeatTime / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &)src1);
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
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
        MicroAPI::RegTensor<T> src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src2Reg, src0 + offset);
            MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
            MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        uint32_t sreg = (uint32_t)calCount;
        MicroAPI::MaskReg maskReg, selMask;
        MicroAPI::Duplicate(src1Reg, (const T &)src1);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
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
    VF_CALL<SelectMode1Level2<T, U>>(dst, sel, src0, src1, calCount);
}
// Src0Scalar
template <typename T, typename U>
__aicore__ inline void SelectSrc0ScalarMode1Level2(__ubuf__ T* dst, __ubuf__ U* sel, T src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg selMask, maskReg;
        MicroAPI::RegTensor<uint8_t> selReg;
        MicroAPI::Duplicate(src0Reg, (const T &)src0);
        MicroAPI::UnalignReg ureg;
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
            MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
            MicroAPI::MaskUnPack(selMask, selMask);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
        }
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t unRollConstant = 2;
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
        MicroAPI::RegTensor<T> src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
        MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        uint16_t tail = repeatTime % unRollConstant;
        uint16_t newRepeatTimes = repeatTime / unRollConstant;
        MicroAPI::Duplicate(scalarReg, (const T &)src0);
        for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
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
        MicroAPI::RegTensor<T> src2Reg, dst2Reg;
        MicroAPI::MaskReg selMask2;
        uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
        uint32_t newSelOffset = newRepeatTimes * selOffset;
        for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
            MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign<T>(src2Reg, src1 + offset);
            MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
            MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
        }
    } else {
        constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
        MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
        MicroAPI::MaskReg maskReg, selMask;
        MicroAPI::Duplicate(src0Reg, (const T &)src0);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
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
    VF_CALL<SelectSrc0ScalarMode1Level2<T, U>>(dst, sel, src0, src1, calCount);
}
// both src0 / src1 Tensor
template <typename T, typename U, uint8_t scalarIdx, MicroAPI::LoadDist pattern = MicroAPI::LoadDist::DIST_BRC_B32>
__aicore__ inline void SelectBothTensorMode1Level2(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (scalarIdx == 0) {
        if constexpr (sizeof(T) == 8) {
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg, tmpReg;
            MicroAPI::MaskReg selMask, maskReg;
            MicroAPI::RegTensor<uint8_t> selReg;
            MicroAPI::UnalignReg ureg, uregDup;
            MicroAPI::LoadUnAlignPre(uregDup, (__ubuf__ T *)src0);
            MicroAPI::LoadUnAlign(tmpReg, uregDup, (__ubuf__ T *)src0);
            MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(src0Reg, tmpReg, maskFull);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
                MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
                MicroAPI::MaskUnPack(selMask, selMask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
            }
        } else if constexpr (sizeof(T) == 4) {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            MicroAPI::RegTensor<T> src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg maskReg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = repeatTime % unRollConstant;
            uint16_t newRepeatTimes = repeatTime / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src0);
            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
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
            MicroAPI::RegTensor<T> src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src2Reg, src1 + offset);
                MicroAPI::Select(dst2Reg, scalarReg, src2Reg, selMask2);
                MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
            }
        } else {
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
            MicroAPI::MaskReg maskReg, selMask;
            MicroAPI::LoadAlign<T, pattern>(src0Reg, src0);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
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
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg, tmpReg;
            MicroAPI::MaskReg selMask, maskReg;
            MicroAPI::RegTensor<uint8_t> selReg;
            MicroAPI::UnalignReg ureg, uregDup;
            MicroAPI::LoadUnAlignPre(uregDup, (__ubuf__ T *)src1);
            MicroAPI::LoadUnAlign(tmpReg, uregDup, (__ubuf__ T *)src1);
            MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(src1Reg, tmpReg, maskFull);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
                MicroAPI::LoadUnAlignPre(ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::LoadUnAlign(selReg, ureg, (__ubuf__ uint8_t *)sel + i * selOffset);
                MicroAPI::MaskGenWithRegTensor<uint32_t, 0>(selMask, (MicroAPI::RegTensor<uint32_t> &)selReg);
                MicroAPI::MaskUnPack(selMask, selMask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
                MicroAPI::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
            }
        } else if constexpr (sizeof(T) == 4) {
            constexpr uint32_t unRollConstant = 2;
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T) * unRollConstant;
            MicroAPI::RegTensor<T> src0Reg, src1Reg, scalarReg, dst0Reg, dst1Reg;
            MicroAPI::MaskReg maskReg;
            MicroAPI::MaskReg selMask0, selMask1, tmpMask0;
            MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            uint16_t tail = repeatTime % unRollConstant;
            uint16_t newRepeatTimes = repeatTime / unRollConstant;
            MicroAPI::LoadAlign<T, pattern>(scalarReg, src1);
            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i) {
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
            MicroAPI::RegTensor<T> src2Reg, dst2Reg;
            MicroAPI::MaskReg selMask2;
            uint32_t offset = newRepeatTimes * unRollConstant * repeatElm;
            uint32_t newSelOffset = newRepeatTimes * selOffset;
            for (uint16_t i = 0; i < (uint16_t)tail; ++i) {
                MicroAPI::LoadAlign<uint8_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint8_t *)sel + newSelOffset);
                MicroAPI::MaskUnPack(selMask2, selMask2);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign<T>(src2Reg, src0 + offset);
                MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
                MicroAPI::StoreAlign<T>(dst + offset, dst2Reg, maskReg);
            }
        } else {
            constexpr uint32_t selOffset = GetVecLen() / CmpSelInternal::maskBitToByte / sizeof(T);
            MicroAPI::RegTensor<T> src0Reg, src1Reg, dstReg;
            MicroAPI::MaskReg maskReg, selMask;
            MicroAPI::LoadAlign<T, pattern>(src1Reg, src1);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
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
        VF_CALL<SelectBothTensorMode1Level2<T, U, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B8>>(dst, sel, src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        VF_CALL<SelectBothTensorMode1Level2<T, U, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B16>>(dst, sel, src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        VF_CALL<SelectBothTensorMode1Level2<T, U, scalarIdx, MicroAPI::LoadDist::DIST_BRC_B32>>(dst, sel, src0, src1, calCount);
    } else {
        VF_CALL<SelectBothTensorMode1Level2<T, U, scalarIdx>>(dst, sel, src0, src1, calCount);
    }
}

template <typename T>
__aicore__ inline void GetCmpMaskImpl(__ubuf__ T* dst)
{
    pipe_barrier(PIPE_ALL);
    (*(__ubuf__ uint64_t *)((__ubuf__ uint64_t *)dst)) = Internal::g_cmpMaskLow;
    (*(__ubuf__ uint64_t *)((__ubuf__ uint64_t *)dst + 1)) = Internal::g_cmpMaskHigh;
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void SetCmpMaskImpl(__ubuf__ T* src)
{
    pipe_barrier(PIPE_ALL);
    Internal::g_cmpMaskLow = reinterpret_cast<uint64_t>(((__ubuf__ uint64_t *)src)[0]);
    Internal::g_cmpMaskHigh = reinterpret_cast<uint64_t>(((__ubuf__ uint64_t *)src)[1]);
    pipe_barrier(PIPE_ALL);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
