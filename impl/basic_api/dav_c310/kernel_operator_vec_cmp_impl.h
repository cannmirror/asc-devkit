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
 * \file kernel_operator_vec_cmp_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMP_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMP_IMPL_H

#include "kernel_utils.h"

namespace AscendC {
namespace CmpInternal {
    constexpr uint32_t maskBitToByte = 8;
}
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
// Compare written to CMPMASK
template <typename T, bool isSetMask, CMPMODE cmpMode>
__simd_vf__ inline void CompareWithoutDstCounterModeImplVF(
    __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t* tempBuf, const uint64_t mask, const BinaryRepeatParams repeatParams)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::MaskReg maskReg, dstReg;
    MicroAPI::UnalignReg uReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
    }
    maskReg = MicroAPI::UpdateMask<T>(sreg);
    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0, src0, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1, src1, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
    MicroAPI::Compare<T, cmpMode>(dstReg, srcReg0, srcReg1, maskReg);
    MicroAPI::DataCopyUnAlign((__ubuf__ T *&)tempBuf, dstReg, uReg);
    MicroAPI::DataCopyUnAlignPost<uint64_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(tempBuf, uReg, 0);
}

template <typename T, bool isSetMask, bool isBitMap, CMPMODE cmpMode>
__simd_vf__ inline void CompareWithoutDstNormalModeImplVF(
    __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ uint64_t* tempBuf, const uint64_t mask, const BinaryRepeatParams repeatParams)
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
    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0, src0, static_cast<uint32_t>(repeatParams.src0BlkStride), maskReg);
    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1, src1, static_cast<uint32_t>(repeatParams.src1BlkStride), maskReg);
    MicroAPI::Compare<T, cmpMode>(dstReg, srcReg0, srcReg1, maskReg);
    MicroAPI::DataCopyUnAlign((__ubuf__ T *&)tempBuf, dstReg, uReg);
    MicroAPI::DataCopyUnAlignPost<uint64_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(tempBuf, uReg, 0);
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
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
    if (isCounterMode) {
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LT>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GT>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::EQ>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LE>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GE>(src0, src1, tempBuf, mask[0], repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::NE>(src0, src1, tempBuf, mask[0], repeatParams);
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
                CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::LT>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::GT>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::EQ>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::LE>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::GE>(src0, src1, tempBuf, 0, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, true, CMPMODE::NE>(src0, src1, tempBuf, 0, repeatParams);
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
        Internal::g_cmpMaskLow = (uint64_t)tempBuf[0];
        Internal::g_cmpMaskHigh = (uint64_t)tempBuf[1];
    } else if constexpr (sizeof(T) == 4) {
        Internal::g_cmpMaskLow = (uint64_t)tempBuf[0];
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
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
    if (isCounterMode) {
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LT>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GT>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::EQ>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::LE>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::GE>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareWithoutDstCounterModeImplVF<T, isSetMask, CMPMODE::NE>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            default:
                break;
        }
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::LT>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::GT>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::EQ>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::LE>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::GE>(src0, src1, tempBuf, mask, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareWithoutDstNormalModeImplVF<T, isSetMask, false, CMPMODE::NE>(src0, src1, tempBuf, mask, repeatParams);
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
        Internal::g_cmpMaskLow = (uint64_t)tempBuf[0];
        Internal::g_cmpMaskHigh = (uint64_t)tempBuf[1];
    } else if constexpr (sizeof(T) == 4) {
        Internal::g_cmpMaskLow = (uint64_t)tempBuf[0];
        Internal::g_cmpMaskHigh = static_cast<uint64_t>(0);
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
}

// Compare::Level 0 - bit mode / Continious mode support b16/b32
template <typename T, typename U, CMPMODE cmpMode>
__simd_vf__ inline void CompareLevel0CounterMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const BinaryRepeatParams repeatParams)
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
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, static_cast<uint32_t>(repeatParams.src0BlkStride), static_cast<uint32_t>(repeatParams.src0RepStride), maskReg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, static_cast<uint32_t>(repeatParams.src1BlkStride), static_cast<uint32_t>(repeatParams.src1RepStride), maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

// Compare::Level 0 - bit mode / Continious mode support b8/b16/b32
template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode>
__simd_vf__ inline void CompareLevel0NormalMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams repeatParams)
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
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, static_cast<uint32_t>(repeatParams.src0BlkStride), static_cast<uint32_t>(repeatParams.src0RepStride), maskReg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, static_cast<uint32_t>(repeatParams.src1BlkStride), static_cast<uint32_t>(repeatParams.src1RepStride), maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
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
                CompareLevel0CounterMode<T, U, CMPMODE::LT>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareLevel0CounterMode<T, U, CMPMODE::GT>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareLevel0CounterMode<T, U, CMPMODE::EQ>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareLevel0CounterMode<T, U, CMPMODE::LE>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareLevel0CounterMode<T, U, CMPMODE::GE>(
                    dst, src0, src1, mask[0], repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareLevel0CounterMode<T, U, CMPMODE::NE>(
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
                CompareLevel0NormalMode<T, U, CMPMODE::LT, true>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareLevel0NormalMode<T, U, CMPMODE::GT, true>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareLevel0NormalMode<T, U, CMPMODE::EQ, true>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareLevel0NormalMode<T, U, CMPMODE::LE, true>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareLevel0NormalMode<T, U, CMPMODE::GE, true>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareLevel0NormalMode<T, U, CMPMODE::NE, true>(
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
                CompareLevel0CounterMode<T, U, CMPMODE::LT>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareLevel0CounterMode<T, U, CMPMODE::GT>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareLevel0CounterMode<T, U, CMPMODE::EQ>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareLevel0CounterMode<T, U, CMPMODE::LE>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareLevel0CounterMode<T, U, CMPMODE::GE>(dst, src0, src1, mask, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareLevel0CounterMode<T, U, CMPMODE::NE>(dst, src0, src1, mask, repeatParams);
                break;
            }
            default:
                break;
        }
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareLevel0NormalMode<T, U, CMPMODE::LT, false>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareLevel0NormalMode<T, U, CMPMODE::GT, false>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareLevel0NormalMode<T, U, CMPMODE::EQ, false>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareLevel0NormalMode<T, U, CMPMODE::LE, false>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareLevel0NormalMode<T, U, CMPMODE::GE, false>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareLevel0NormalMode<T, U, CMPMODE::NE, false>(dst, src0, src1, mask, repeatTime, repeatParams);
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
// CompareScalar::Level 0 - bit mode / continious mode
template <typename T, typename U, CMPMODE cmpMode, bool isSetMask>
__simd_vf__ inline void CompareScalarLevel0CounterMode(__ubuf__ U *dst, __ubuf__ T *src0, const T src1, const uint64_t mask, __ubuf__ uint64_t *tempBuf,
    const UnaryRepeatParams repeatParams)
{
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> src0Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    uint32_t sreg = static_cast<uint32_t>(mask);
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
    }
    uint16_t newRepeatTimes = CeilDivision(sreg, oneRepSize);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, bool isSetMask>
__simd_vf__ inline void CompareScalarLevel0NormalMode(__ubuf__ U *dst, __ubuf__ T *src0, const T src1,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams repeatParams)
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
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src0Reg, src0, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
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
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>(
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
                CompareScalarLevel0NormalMode<T, U, CMPMODE::LT, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::GT, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::EQ, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::LE, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::GE, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::NE, true, isSetMask>(
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
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>(
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
                CompareScalarLevel0NormalMode<T, U, CMPMODE::LT, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::GT, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::EQ, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::LE, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::GE, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareScalarLevel0NormalMode<T, U, CMPMODE::NE, false, isSetMask>(
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
__simd_vf__ inline void CompareScalarLevel2(__ubuf__ U *dst, __ubuf__ T *src0, const T src1, const uint32_t calCount)
{
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg;
    uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        repeatElm = repeatElm * 2;
        repeatTime = CeilDivision(calCount, repeatElm);
        if constexpr (Std::is_same_v<T, double>) {
	        MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> src1Reg;
            MicroAPI::RegTensor<double, MicroAPI::RegTraitNumTwo> src0Reg;
            MicroAPI::Duplicate(src1Reg, GetScalarBitcodeValue<double, uint64_t>(src1)); 
            for (uint16_t i = 0; i < repeatTime; ++i) {
                maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
                CompareEqualDouble<uint64_t>(dstReg, (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)src0Reg, src1Reg, maskReg);  
                MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
            }
        } else {        
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
                MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
                MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
            }
        }
        MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
    } else {
        MicroAPI::RegTensor<T> src0Reg;
        constexpr uint32_t offset = GetVecLen() / sizeof(T) / CmpInternal::maskBitToByte;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
            MicroAPI::CompareScalar<T, cmpMode>(dstReg, src0Reg, src1, maskReg);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::DataCopy(dst + i * offset, dstReg);
            } else {
                MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
            }
        }
        if constexpr (sizeof(T) > 1) {
            MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(
    __ubuf__ U *dst, __ubuf__ T *src0, const T src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float,
        uint64_t, int64_t, double>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    switch (cmpMode) {
        case CMPMODE::LT: {
            CompareScalarLevel2<T, U, CMPMODE::LT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            CompareScalarLevel2<T, U, CMPMODE::GT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            CompareScalarLevel2<T, U, CMPMODE::EQ>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            CompareScalarLevel2<T, U, CMPMODE::LE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            CompareScalarLevel2<T, U, CMPMODE::GE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            CompareScalarLevel2<T, U, CMPMODE::NE>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}

/* ***************************************************************************************
 * ************************************** CompareScalar src0 scalar****************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - bit mode / continious mode
template <typename T, typename U, CMPMODE cmpMode, bool isSetMask>
__simd_vf__ inline void CompareSrc0ScalarLevel0CounterMode(__ubuf__ U *dst, const T src0, __ubuf__ T *src1, const uint64_t mask,
    __ubuf__ uint64_t *tempBuf, const UnaryRepeatParams repeatParams)
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
        MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        sreg = static_cast<uint32_t>(tempBuf[0]);
    }
    uint16_t newRepeatTimes = CeilDivision(sreg, oneRepSize);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, bool isSetMask>
__simd_vf__ inline void CompareSrc0ScalarLevel0NormalMode(__ubuf__ U *dst, const T src0, __ubuf__ T *src1,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams repeatParams)
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
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            src1Reg, src1, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
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
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>(
                    dst, src0, src1, mask[0], tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>(
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
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LT, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GT, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::EQ, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LE, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GE, true, isSetMask>(
                    dst, src0, src1, 0, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::NE, true, isSetMask>(
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
         __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LT, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GT, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::EQ, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::LE, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::GE, isSetMask>(
                    dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareSrc0ScalarLevel0CounterMode<T, U, CMPMODE::NE, isSetMask>(
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
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LT, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GT, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::EQ, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::LE, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::GE, false, isSetMask>(
                    dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareSrc0ScalarLevel0NormalMode<T, U, CMPMODE::NE, false, isSetMask>(
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
__simd_vf__ inline void CompareSrc0ScalarLevel2(__ubuf__ U *dst, const T src0, __ubuf__ T *src1, const uint32_t calCount)
{
    uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (sizeof(T) == 8) {
        repeatElm = repeatElm * 2;
        repeatTime = CeilDivision(calCount, repeatElm);
        if constexpr (Std::is_same_v<T, double>) {
	        MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> src0Reg;
            MicroAPI::RegTensor<double, MicroAPI::RegTraitNumTwo> src1Reg;
            MicroAPI::Duplicate(src0Reg, GetScalarBitcodeValue<double, uint64_t>(src0));
            for (uint16_t i = 0; i < repeatTime; ++i) {
                maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
                CompareEqualDouble<uint64_t>(dstReg, (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)src1Reg, src0Reg, maskReg);
                MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
            }        
        } else {
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg, src1Reg;
            MicroAPI::Duplicate(src0Reg, src0);
            for (uint16_t i = 0; i < repeatTime; ++i) {
                maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
                MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
                MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
            }
        }
        MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
    } else {
        MicroAPI::RegTensor<T> src0Reg, src1Reg;
        constexpr uint32_t offset = GetVecLen() / sizeof(T) / CmpInternal::maskBitToByte;
        MicroAPI::Duplicate(src0Reg, src0);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
            MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::DataCopy(dst + i * offset, dstReg);
            } else {
                MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
            }
        }
        if constexpr (sizeof(T) > 1) {
            MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(
    __ubuf__ U *dst, const T src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, uint8_t, int8_t, bfloat16_t, half, int16_t, uint16_t, int32_t, uint32_t, float,
        uint64_t, int64_t, double>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t>(), "current data type is not supported!");
    switch (cmpMode) {
        case CMPMODE::LT: {
            CompareSrc0ScalarLevel2<T, U, CMPMODE::LT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            CompareSrc0ScalarLevel2<T, U, CMPMODE::GT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            CompareSrc0ScalarLevel2<T, U, CMPMODE::EQ>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            CompareSrc0ScalarLevel2<T, U, CMPMODE::LE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            CompareSrc0ScalarLevel2<T, U, CMPMODE::GE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            CompareSrc0ScalarLevel2<T, U, CMPMODE::NE>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}
/* ***************************************************************************************
 * ************************************** CompareScalar Both Tensor****************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - bit mode / continious mode

template <typename T, typename U, CMPMODE cmpMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern>
__simd_vf__ inline void CompareScalarBothTensorLevel2(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg;
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    constexpr uint32_t offset = GetVecLen() / sizeof(T) / CmpInternal::maskBitToByte;
    if constexpr (scalarIdx == 0) {
        MicroAPI::DataCopy<T, pattern>(src0Reg, src0);
    } else {
        MicroAPI::DataCopy<T, pattern>(src1Reg, src1);
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
        } else {
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        if constexpr (sizeof(T) == 1) {
            MicroAPI::DataCopy(dst + i * offset, dstReg);
        } else {
            MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
        }
    }
    if constexpr (sizeof(T) > 1) {
        MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
    }
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1, MicroAPI::LoadDist pattern>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CompareScalarBothTensorLevel2<T, U, CMPMODE::LT, scalarIdx, pattern>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            CompareScalarBothTensorLevel2<T, U, CMPMODE::GT, scalarIdx, pattern>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            CompareScalarBothTensorLevel2<T, U, CMPMODE::EQ, scalarIdx, pattern>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            CompareScalarBothTensorLevel2<T, U, CMPMODE::LE, scalarIdx, pattern>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            CompareScalarBothTensorLevel2<T, U, CMPMODE::GE, scalarIdx, pattern>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            CompareScalarBothTensorLevel2<T, U, CMPMODE::NE, scalarIdx, pattern>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, CMPMODE cmpMode, uint8_t scalarIdx, typename Std::enable_if<!Std::is_same<PrimT<T>, double>::value, bool>::type = true>
__simd_vf__ inline void CompareScalarLevel2B64(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * 2;
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg, dupuReg;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg, src1Reg, dupReg;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> preReg;
    MicroAPI::RegTensor<uint32_t> zeroReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0, maskFull);
    if constexpr (scalarIdx == 0) {
        MicroAPI::DataCopyUnAlignPre(dupuReg, (__ubuf__ T *)src0);
        MicroAPI::DataCopyUnAlign(preReg, dupuReg, (__ubuf__ T *)src0);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1],
            (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(src0Reg, dupReg, maskFull);
    } else {
        MicroAPI::DataCopyUnAlignPre(dupuReg, (__ubuf__ T *)src1);
        MicroAPI::DataCopyUnAlign(preReg, dupuReg, (__ubuf__ T *)src1);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1],
            (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(src1Reg, dupReg, maskFull);
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
        } else {
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, uint8_t scalarIdx, typename Std::enable_if<Std::is_same<PrimT<T>, double>::value, bool>::type = true>
__simd_vf__ inline void CompareScalarLevel2B64(
    __ubuf__ U *dst, __ubuf__ double *src0, __ubuf__ double *src1, const uint32_t calCount)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(double) * 2;
    uint16_t repeatTime = CeilDivision(calCount, repeatElm);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg dstReg, maskReg;
    MicroAPI::UnalignReg uReg, dupuReg;
    MicroAPI::RegTensor<double, MicroAPI::RegTraitNumOne> preReg;
    MicroAPI::RegTensor<uint32_t> zeroReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0, maskFull);
    if constexpr (scalarIdx == 0) {
        MicroAPI::RegTensor<double, MicroAPI::RegTraitNumTwo> src1Reg, dupReg;
        MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> src0Reg;
        MicroAPI::DataCopyUnAlignPre(dupuReg, (__ubuf__ double *)src0);
        MicroAPI::DataCopyUnAlign(preReg, dupuReg, (__ubuf__ double *)src0);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1],
            (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(src0Reg, (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)dupReg, maskFull);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<double, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatElm);
            CompareEqualDouble<uint64_t>(dstReg, src0Reg, (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)src1Reg, maskReg);   
            MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
        }
    } else {
        MicroAPI::RegTensor<double, MicroAPI::RegTraitNumTwo> src0Reg, dupReg;
        MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> src1Reg;
        MicroAPI::DataCopyUnAlignPre(dupuReg, (__ubuf__ double *)src1);
        MicroAPI::DataCopyUnAlign(preReg, dupuReg, (__ubuf__ double *)src1);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1],
            (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(src1Reg, (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)dupReg, maskFull);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            maskReg = MicroAPI::UpdateMask<double, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatElm);
            CompareEqualDouble<uint64_t>(dstReg, (MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>&)src0Reg, src1Reg, maskReg);   
            MicroAPI::DataCopyUnAlign((__ubuf__ uint32_t *&)dst, dstReg, uReg);
        }
    }        
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void VcmpvsImplB64(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CompareScalarLevel2B64<T, U, CMPMODE::LT, scalarIdx>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            CompareScalarLevel2B64<T, U, CMPMODE::GT, scalarIdx>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            CompareScalarLevel2B64<T, U, CMPMODE::EQ, scalarIdx>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            CompareScalarLevel2B64<T, U, CMPMODE::LE, scalarIdx>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            CompareScalarLevel2B64<T, U, CMPMODE::GE, scalarIdx>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            CompareScalarLevel2B64<T, U, CMPMODE::NE, scalarIdx>(dst, src0, src1, calCount);
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
        uint64_t, int64_t, double>(), "current data type is not supported!");
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
__simd_vf__ inline void CompareScalarBothTensorLevel0CounterMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, __ubuf__ uint64_t *tempBuf, const UnaryRepeatParams repeatParams)
{
    MicroAPI::MaskReg maskReg;
    uint32_t countSreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        maskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        countSreg = static_cast<uint32_t>(tempBuf[0]);
    }
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t newRepeatTimes = CeilDivision(countSreg, oneRepSize);
    MicroAPI::RegTensor<T> src0Reg, src1Reg;
    MicroAPI::MaskReg dstReg;
    MicroAPI::UnalignReg uReg;
    if constexpr (scalarIdx == 0) {
        MicroAPI::DataCopy<T, pattern>(src0Reg, src0);
    } else {
        MicroAPI::DataCopy<T, pattern>(src1Reg, src1);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(countSreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src1Reg, src1, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        } else {
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src0Reg, src0, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, CMPMODE cmpMode, bool isBitMapMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern, bool isSetMask>
__simd_vf__ inline void CompareScalarBothTensorLevel0NormalMode(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams repeatParams)
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
        MicroAPI::DataCopy<T, pattern>(src0Reg, src0);
    } else {
        MicroAPI::DataCopy<T, pattern>(src1Reg, src1);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src1Reg, src1, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        } else {
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                src0Reg, src0, static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), maskReg);
        }
        MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, maskReg);
        MicroAPI::DataCopyUnAlign((__ubuf__ T *&)dst, dstReg, uReg);
    }
    MicroAPI::DataCopyUnAlignPost<U, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dst, uReg, 0);
}

template <typename T, typename U, bool isSetMask = true, bool isBitMapMode, uint8_t scalarIdx, MicroAPI::LoadDist pattern>
__aicore__ inline void VcmpvsImpl(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::LT, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::GT, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::EQ, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::LE, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::GE, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareScalarBothTensorLevel0CounterMode<T, U, CMPMODE::NE, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, tempBuf, repeatParams);
                break;
            }
            default:
                break;
        }
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(tempBuf);
    } else {
        switch (cmpMode) {
            case CMPMODE::LT: {
                CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::LT, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GT: {
                CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::GT, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::EQ: {
                CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::EQ, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::LE: {
                CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::LE, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::GE: {
                CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::GE, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, repeatTime, repeatParams);
                break;
            }
            case CMPMODE::NE: {
                CompareScalarBothTensorLevel0NormalMode<T, U, CMPMODE::NE, isBitMapMode, scalarIdx, pattern, isSetMask>(dst, src0, src1, mask, repeatTime, repeatParams);
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
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMP_IMPL_H
