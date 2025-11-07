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
 * \file kernel_operator_vec_gather_impl.h
 * \brief AscendC l300 support vector gather api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#include "kernel_operator_common_impl.h"
namespace AscendC {

template <typename T, bool isNormalMode>
__aicore__ inline void VfGatherApi0B16(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *srcOffsetLocal,
    const uint32_t srcBaseIndex, const uint8_t repeatTime, const uint16_t &dstRepStride, uint32_t dstRepeatCount,
    uint32_t u32OffsetRepeatCount, uint32_t blkCount, const uint64_t maskCount)
{
    MicroAPI::RegTensor<uint32_t> offsetReg0;
    MicroAPI::RegTensor<uint32_t> offsetReg1;
    MicroAPI::RegTensor<uint16_t> indexReg;
    MicroAPI::RegTensor<uint16_t> dstReg;
    uint32_t sregPlt = static_cast<uint32_t>(maskCount);
    MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<uint32_t>();
    MicroAPI::MaskReg selectMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg dstMask;
    if constexpr (isNormalMode) {
        dstMask = MicroAPI::MoveMask<T>();
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        if constexpr (!isNormalMode) {
            dstMask = MicroAPI::UpdateMask<T>(sregPlt);
        }
        DataCopy(offsetReg0, srcOffsetLocal + (2 * i) * u32OffsetRepeatCount);
        DataCopy(offsetReg1, srcOffsetLocal + (2 * i + 1) * u32OffsetRepeatCount);
        // convert addr offset into B16 element index: divide by 2 (implemented by ShiftRight 1 bit)
        ShiftRights(offsetReg0, offsetReg0, (int16_t)1, indexMask);
        ShiftRights(offsetReg1, offsetReg1, (int16_t)1, indexMask);
        // extract the lower 16-bit of uint32_t offset data into uint16_t index data:
        // for offsetReg0，pack every lower 16-bit into the lower half of the vreg：
        // 0x00FF00FE00FD... ->0xFFFEFD...000000...
        // for offsetReg1, pack every higher 16-bit into the higher half of the vreg:
        // 0x001100120013... -> 0x000000...111213...
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t> &)offsetReg0,
            offsetReg0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::HIGHEST>((MicroAPI::RegTensor<uint16_t> &)offsetReg1,
            offsetReg1);
        // Select the effective data in offsetReg0 and offsetReg1 and joint them into a complete uint16_t type
        // indexReg：0xFFFEFD...111213...
        Select(indexReg, (MicroAPI::RegTensor<uint16_t> &)offsetReg0, (MicroAPI::RegTensor<uint16_t> &)offsetReg1,
            selectMask);
        DataCopyGather(dstReg, (__ubuf__ uint16_t *)src + srcBaseIndex, indexReg, dstMask);
        DataCopy((__ubuf__ uint16_t *)dst + i * dstRepStride * blkCount, dstReg, dstMask);
    }
}

template <typename T, bool isNormalMode>
__aicore__ inline void GatherApi0B16Impl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *srcOffsetLocal,
    const uint32_t srcBaseIndex, const uint8_t repeatTime, const uint16_t &dstRepStride, const uint64_t maskCount)
{
    uint32_t dstRepeatCount = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint32_t u32OffsetRepeatCount = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(uint32_t));
    uint32_t blkCount = static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(T));
    VF_CALL<VfGatherApi0B16<T, isNormalMode>>(dst, src, srcOffsetLocal, srcBaseIndex, repeatTime, dstRepStride,
        dstRepeatCount, u32OffsetRepeatCount, blkCount, maskCount);
}

template <typename T, bool isNormalMode>
__aicore__ inline void VfGatherApi0B32(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *srcOffsetLocal,
    const uint32_t srcBaseIndex, const uint8_t repeatTime, const uint16_t &dstRepStride, uint32_t dstRepeatCount,
    uint32_t u32OffsetRepeatCount, uint32_t blkCount, const uint64_t maskCount)
{
    MicroAPI::RegTensor<uint32_t> offsetReg;
    MicroAPI::RegTensor<uint32_t> indexReg;
    MicroAPI::RegTensor<uint32_t> dstReg;
    uint32_t sregPlt = static_cast<uint32_t>(maskCount);
    MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<T>();
    MicroAPI::MaskReg dstMask;
    MicroAPI::MaskReg offsetMask = MicroAPI::CreateMask<uint32_t>();
    if constexpr (isNormalMode) {
        dstMask = MicroAPI::MoveMask<T>();
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        if constexpr (!isNormalMode) {
            dstMask = MicroAPI::UpdateMask<T>(sregPlt);
        }
        DataCopy(offsetReg, srcOffsetLocal + i * u32OffsetRepeatCount);
        // convert addr offset into B32 element index: divide by 4 (implemented by ShiftRight 2 bit)
        ShiftRights(indexReg, offsetReg, (int16_t)2, indexMask);
        DataCopyGather(dstReg, (__ubuf__ uint32_t *)src + srcBaseIndex, indexReg, dstMask);
        DataCopy((__ubuf__ uint32_t *)dst + i * dstRepStride * blkCount, dstReg, dstMask);
    }
}

template <typename T, bool isNormalMode>
__aicore__ inline void GatherApi0B32Impl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *srcOffsetLocal,
    const uint32_t srcBaseIndex, const uint8_t repeatTime, const uint16_t &dstRepStride, const uint64_t maskCount)
{
    uint32_t dstRepeatCount = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint32_t u32OffsetRepeatCount = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(uint32_t));
    uint32_t blkCount = static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(T));
    VF_CALL<VfGatherApi0B32<T, isNormalMode>>(dst, src, srcOffsetLocal, srcBaseIndex, repeatTime, dstRepStride,
        dstRepeatCount, u32OffsetRepeatCount, blkCount, maskCount);
}

/* **************************************************************************************************
 * Gather                                                                                           *
 * **************************************************************************************************/
// gatherb::Level 0
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* offset,
    const uint32_t srcLength, uint8_t repeatTimes, const GatherRepeatParams& repeatParams)
{
    uint32_t repeatStride = VECTOR_REG_WIDTH / ONE_BLK_SIZE;
    __VEC_SCOPE__
    {
        uint16_t dstRptStd = repeatParams.dstRepStride;
        uint8_t dstBlkStd = repeatParams.dstBlkStride;
        RegTensor<T> vDst;
        RegTensor<uint32_t> vregIndex;
        uint32_t sreg = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        MaskReg preg = CreatePredicate<T>(sreg);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            AddrReg indexOffset = CreateAddrReg<uint32_t>(repeatStride);
            DataCopy(vregIndex, offset, indexOffset);
            DataCopyGatherB(vDst, src0, vregIndex, preg);
            DataCopy(dst, vDst, dstBlkStd, i * dstRptStd, preg);
        }
    }
}

// Gather::Level 0 Normal mode
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint16_t &dstRepStride)
{
    static_assert(SupportBytes<T, 2, 4>(), "Gather only support type b16/b32 on current device");

    uint8_t newRepeatTimes = repeatTime;
    bool isNormalMode = !Internal::IsCounterMode();
    if (isNormalMode) {
        if constexpr (sizeof(T) == 2) {
            SetVectorMask<uint16_t>(mask);
        } else {
            SetVectorMask<uint32_t>(mask);
        }
    } else {
        newRepeatTimes = static_cast<uint8_t>(Internal::VecMicroGetRepeatTimes<T, false>(mask, repeatTime));
    }

    uint32_t srcBaseIndex;
    if constexpr (sizeof(T) == 2) {
        srcBaseIndex = srcBaseAddr / sizeof(T);
        if (isNormalMode) {
            GatherApi0B16Impl<T, true>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask);
        } else {
            GatherApi0B16Impl<T, false>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask);
        }
    } else if constexpr (sizeof(T) == 4) {
        srcBaseIndex = srcBaseAddr / sizeof(T);
        if (isNormalMode) {
            GatherApi0B32Impl<T, true>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask);
        } else {
            GatherApi0B32Impl<T, false>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask);
        }
    }
}

/* **************************************************************************************************
 * Gather                                             *
 * ************************************************************************************************* */
// Gather::Level 0 Bit-wise mode
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTime,
    const uint16_t &dstRepStride)
{
    static_assert(SupportBytes<T, 2, 4>(), "Gather only support type b16/b32 on current device");

    uint8_t newRepeatTimes = repeatTime;
    bool isNormalMode = !Internal::IsCounterMode();
    if (isNormalMode) {
        if constexpr (sizeof(T) == 2) {
            SetVectorMask<uint16_t>(mask[1], mask[0]);
        } else {
            SetVectorMask<uint32_t>(mask[1], mask[0]);
        }
    } else {
        newRepeatTimes = static_cast<uint8_t>(Internal::VecMicroGetRepeatTimes<T, false>(mask[0], repeatTime));
    }

    uint32_t srcBaseIndex;
    if constexpr (sizeof(T) == 2) {
        srcBaseIndex = srcBaseAddr / sizeof(T);
        if (isNormalMode) {
            GatherApi0B16Impl<T, true>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask[0]);
        } else {
            GatherApi0B16Impl<T, false>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask[0]);
        }
    } else if constexpr (sizeof(T) == 4) {
        srcBaseIndex = srcBaseAddr / sizeof(T);
        if (isNormalMode) {
            GatherApi0B32Impl<T, true>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask[0]);
        } else {
            GatherApi0B32Impl<T, false>(dst, src, srcOffsetLocal, srcBaseIndex, newRepeatTimes, dstRepStride, mask[0]);
        }
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
