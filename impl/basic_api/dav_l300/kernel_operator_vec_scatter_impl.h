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
 * \file kernel_operator_vec_scatter_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H
#include "kernel_operator_common_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * scatter                                             *
 * ************************************************************************************************* */
constexpr uint32_t mulsScalar = 2;
constexpr uint32_t addsScalar = 1;
constexpr int16_t b32ShiftVal = 2;
constexpr int16_t b64ShiftVal = 3;
constexpr int16_t b16ShiftVal = 1;
constexpr uint32_t b32BlkElems = 8;
constexpr uint32_t b16BlkElems = 16;
constexpr uint32_t b8BlkElems = 32;
constexpr uint32_t indexRepElems = 64;
constexpr uint32_t srcRepElems = 64;
constexpr uint32_t srcRep128 = 128;
constexpr uint32_t b64RepElems = 32;
constexpr MicroAPI::CastTrait castTraitEven = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING };

constexpr MicroAPI::CastTrait castTraitOdd = { MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING };

template <typename T>
__aicore__ inline void ScatterImplB16(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *dstOffsetLocal,
    const uint32_t dstBaseOffset, const uint32_t count)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<uint32_t> indexReg;
        MicroAPI::RegTensor<uint16_t> indexU16;
        MicroAPI::RegTensor<uint32_t> indexRegSec;
        MicroAPI::RegTensor<uint16_t> lowerU16Reg;
        MicroAPI::RegTensor<uint16_t> highU16Reg;
        MicroAPI::MaskReg preg;
        uint32_t sregPlt = static_cast<uint32_t>(count);
        MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg selectMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::H>();
        uint16_t repeatTime = CeilDivision(count, srcRep128);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            preg = MicroAPI::UpdateMask<T>(sregPlt);
            MicroAPI::DataCopy<uint32_t>(indexReg, dstOffsetLocal + 2 * i * indexRepElems);
            MicroAPI::DataCopy<uint32_t>(indexRegSec, dstOffsetLocal + (2 * i + 1) * indexRepElems);
            MicroAPI::ShiftRights<uint32_t, int16_t>(indexReg, indexReg, b16ShiftVal, indexMask);
            MicroAPI::ShiftRights<uint32_t, int16_t>(indexRegSec, indexRegSec, b16ShiftVal, indexMask);
            MicroAPI::Cast<uint16_t, uint32_t, castTraitEven>(lowerU16Reg, indexReg, indexMask);
            MicroAPI::Cast<uint16_t, uint32_t, castTraitOdd>(highU16Reg, indexRegSec, indexMask);
            MicroAPI::DeInterleave(lowerU16Reg, highU16Reg, lowerU16Reg, highU16Reg);
            MicroAPI::Select(indexU16, lowerU16Reg, highU16Reg, selectMask);
            MicroAPI::DataCopy<T>(srcReg, srcLocal + i * srcRep128);
            MicroAPI::DataCopyScatter<T, uint16_t>(dstLocal + dstBaseOffset, srcReg, indexU16, preg);
        }
    }
}

template <typename T, bool isNormalMode = true, bool isMaskBitMode = true>
__aicore__ inline void ScatterImplB16(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *dstOffsetLocal,
    const uint32_t dstLength, const uint32_t dstBaseOffset, const uint64_t mask, const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> indexReg;
        MicroAPI::RegTensor<uint32_t> indexRegSec;
        MicroAPI::RegTensor<uint16_t> indexU16;
        MicroAPI::RegTensor<uint16_t> lowerU16;
        MicroAPI::RegTensor<uint16_t> highU16;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::MaskReg selectMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg b16SrcMask;
        uint32_t maskV = static_cast<uint32_t>(mask);
        if constexpr (isNormalMode) {
            if constexpr (isMaskBitMode) {
                b16SrcMask = MicroAPI::MoveMask<T>();
                MicroAPI::MaskPack(b16SrcMask, b16SrcMask);
                MicroAPI::MaskUnPack(b16SrcMask, b16SrcMask);
            } else {
                b16SrcMask = MicroAPI::UpdateMask<T>(maskV);
            }
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); i++) {
            if constexpr (!isNormalMode) {
                b16SrcMask = MicroAPI::UpdateMask<T>(maskV);
            }
            MicroAPI::DataCopy<uint32_t>(indexReg, dstOffsetLocal + 2 * i * indexRepElems);
            MicroAPI::DataCopy<uint32_t>(indexRegSec, dstOffsetLocal + (2 * i + 1) * indexRepElems);
            MicroAPI::ShiftRights<uint32_t, int16_t>(indexReg, indexReg, b16ShiftVal, indexMask);
            MicroAPI::ShiftRights<uint32_t, int16_t>(indexRegSec, indexRegSec, b16ShiftVal, indexMask);
            MicroAPI::Cast<uint16_t, uint32_t, castTraitEven>(lowerU16, indexReg, indexMask);
            MicroAPI::Cast<uint16_t, uint32_t, castTraitOdd>(highU16, indexRegSec, indexMask);
            MicroAPI::DeInterleave(lowerU16, highU16, lowerU16, highU16);
            MicroAPI::Select(indexU16, lowerU16, highU16, selectMask);
            MicroAPI::DataCopy<T>(srcReg, srcLocal + i * srcRepStride * b16BlkElems);
            MicroAPI::DataCopyScatter<T, uint16_t>(dstLocal + dstBaseOffset, srcReg, indexU16, b16SrcMask);
        }
    }
}

template <typename T>
__aicore__ inline void ScatterImplB32(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *dstOffsetLocal,
    const uint32_t dstBaseOffset, const uint32_t count)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> indexReg;
        MicroAPI::RegTensor<T> srcReg;
        uint32_t sregPlt = static_cast<uint32_t>(count);
        MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg preg;
        uint16_t repeatTime = CeilDivision(count, srcRepElems);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
            preg = MicroAPI::UpdateMask<T>(sregPlt);
            MicroAPI::DataCopy<uint32_t>(indexReg, dstOffsetLocal + i * srcRepElems);
            MicroAPI::ShiftRights<uint32_t, int16_t>(indexReg, indexReg, b32ShiftVal, indexMask);
            MicroAPI::DataCopy<T>(srcReg, srcLocal + i * srcRepElems);
            MicroAPI::DataCopyScatter<T, uint32_t>(dstLocal + dstBaseOffset, srcReg, indexReg, preg);
        }
    }
}

template <typename T, bool isNormalMode = true, bool isMaskBitMode = true>
__aicore__ inline void ScatterImplB32(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *dstOffsetLocal,
    const uint32_t dstLength, const uint32_t dstBaseOffset, const uint64_t mask, const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> indexReg;
        uint32_t maskV = static_cast<uint32_t>(mask);
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg b32SrcMask;
        if constexpr (isNormalMode) {
            if constexpr (isMaskBitMode) {
                b32SrcMask = MicroAPI::MoveMask<T>();
                MicroAPI::MaskPack(b32SrcMask, b32SrcMask);
                MicroAPI::MaskUnPack(b32SrcMask, b32SrcMask);
            } else {
                b32SrcMask = MicroAPI::UpdateMask<T>(maskV);
            }
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); i++) {
            if constexpr (!isNormalMode) {
                b32SrcMask = MicroAPI::UpdateMask<T>(maskV);
            }
            MicroAPI::DataCopy<uint32_t>(indexReg, dstOffsetLocal + i * srcRepElems);
            MicroAPI::ShiftRights<uint32_t, int16_t>(indexReg, indexReg, b32ShiftVal, indexMask);
            MicroAPI::DataCopy<T>(srcReg, srcLocal + i * srcRepStride * b32BlkElems);
            MicroAPI::DataCopyScatter<T, uint32_t>(dstLocal + dstBaseOffset, srcReg, indexReg, b32SrcMask);
        }
    }
}
template <typename T>
__aicore__ inline void ScatterImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *dstOffsetLocal,
    const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    static_assert(SupportBytes<T, 2, 4>(), "Scatter only support type b16/b32 on current device");
    bool isNormalMode = !Internal::IsCounterMode();
    uint32_t dstBaseOffset = dstBaseAddr / sizeof(T);
    if (isNormalMode) {
        if constexpr (sizeof(T) == 2) {
            ScatterImplB16<T, true, false>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask,
                repeatTime, srcRepStride);
        } else if constexpr (sizeof(T) == 4) {
            ScatterImplB32<T, true, false>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask,
                repeatTime, srcRepStride);
        }
    } else {
        uint8_t newRepeatTimes = static_cast<uint8_t>(Internal::VecMicroGetRepeatTimes<T, false>(mask, repeatTime));
        if constexpr (sizeof(T) == 2) {
            ScatterImplB16<T, false, false>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask,
                newRepeatTimes, srcRepStride);
        } else if constexpr (sizeof(T) == 4) {
            ScatterImplB32<T, false, false>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask,
                newRepeatTimes, srcRepStride);
        }
    }
}

template <typename T>
__aicore__ inline void ScatterImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *dstOffsetLocal,
    const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask[], const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    static_assert(SupportBytes<T, 2, 4>(), "Scatter only support type b16/b32 on current device");
    bool isNormalMode = !Internal::IsCounterMode();
    uint32_t dstBaseOffset = dstBaseAddr / sizeof(T);
    if (isNormalMode) {
        SetVectorMask<T>(mask[1], mask[0]);

        if constexpr (sizeof(T) == 2) {
            ScatterImplB16<T, true, true>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask[0],
                repeatTime, srcRepStride);
        } else if constexpr (sizeof(T) == 4) {
            ScatterImplB32<T, true, true>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask[0],
                repeatTime, srcRepStride);
        }
    } else {
        uint8_t newRepeatTimes = static_cast<uint8_t>(Internal::VecMicroGetRepeatTimes<T, false>(mask[0], repeatTime));
        if constexpr (sizeof(T) == 2) {
            ScatterImplB16<T, false, true>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask[0],
                newRepeatTimes, srcRepStride);
        } else if constexpr (sizeof(T) == 4) {
            ScatterImplB32<T, false, true>(dstLocal, srcLocal, dstOffsetLocal, dstLength, dstBaseOffset, mask[0],
                newRepeatTimes, srcRepStride);
        }
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H