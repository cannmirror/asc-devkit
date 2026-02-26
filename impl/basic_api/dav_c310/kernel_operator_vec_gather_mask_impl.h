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
 * \file kernel_operator_vec_gather_mask_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_MASK_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_MASK_IMPL_H
#include "kernel_struct_gather.h"
#include "kernel_common.h"
#include "kernel_operator_sys_var_intf.h"
#include "micro_api/kernel_micro_intf.h"

namespace AscendC {
__aicore__ inline int64_t GetGatherMaskRemainCountImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetGatherMaskRemainCount on current device");
    return 0;
}

template <typename T>
__simd_vf__ inline void GatherMaskAllNormalElePerVec(
    __ubuf__ T *dst, __ubuf__ T *src0, const GatherMaskParams reducev2Params, uint32_t ElePerVec) {
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::MaskReg loadMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::UnalignReg ureg;
    for (uint16_t i = 0; i < reducev2Params.repeatTimes; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            srcReg, src0, reducev2Params.src0BlockStride, reducev2Params.src0RepeatStride, loadMask);
        MicroAPI::StoreUnAlign(dst, srcReg, ureg, ElePerVec);
    }
    MicroAPI::StoreUnAlignPost(dst, ureg, 0);
}

template <typename T>
__aicore__ inline void GatherMaskAllNormal(
    __ubuf__ T *dst, __ubuf__ T *src0, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    constexpr uint32_t ElePerVec = GetVecLen() / sizeof(T);
    GatherMaskAllNormalElePerVec<T>(dst, src0, reducev2Params, ElePerVec);
    rsvdCnt = ElePerVec * reducev2Params.repeatTimes;
}

template <typename T>
__simd_vf__ inline void GatherMaskAllReduceElePerVec(__ubuf__ T *dst, __ubuf__ T *src0,
    const uint32_t mask, const GatherMaskParams reducev2Params, uint16_t innerRepeatTimes, uint8_t ElePerBlkT)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::MaskReg loadMask;
    MicroAPI::UnalignReg ureg;
    uint32_t maskValue = mask;
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < reducev2Params.repeatTimes; ++i) {
        maskValue = mask;
        for (uint16_t j = 0; j < innerRepeatTimes; ++j) {
            loadMask = MicroAPI::UpdateMask<T>(maskValue);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg,
                src0 + i * reducev2Params.src0RepeatStride * ElePerBlkT +
                    j * 8 * reducev2Params.src0BlockStride * ElePerBlkT,
                reducev2Params.src0BlockStride,
                loadMask);
            if constexpr (SupportType<T, bfloat16_t>()) {
                MicroAPI::GatherMask<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(
                    (MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)srcReg, loadMask);
            } else {
                MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(dstReg, srcReg, loadMask);
            }
            MicroAPI::StoreUnAlign(dst, dstReg, ureg);
        }
    }
    MicroAPI::StoreUnAlignPost(dst, ureg);
}

template <typename T>
__aicore__ inline void GatherMaskAllReduce(
    __ubuf__ T *dst, __ubuf__ T *src0, const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t ElePerVec = GetVecLen() / sizeof(T);
    uint16_t innerRepeatTimes = CeilDivision(mask, ElePerVec);
    GatherMaskAllReduceElePerVec<T>(dst, src0, mask, reducev2Params, innerRepeatTimes, ElePerBlkT);
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(T);
}

template <typename T, uint8_t solidPattern>
__simd_vf__ inline void GatherMaskSqueezeNormalTmpBuffer(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ uint8_t *tempBuf, const GatherMaskParams reducev2Params)
{
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::MaskReg loadMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::UnalignReg ureg;
    MicroAPI::MaskReg patternMask;
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    if constexpr (sizeof(T) != 1) {
        patternMask = MicroAPI::MoveMask<T>();
    } else {
        MicroAPI::RegTensor<uint8_t> patternReg;
        MicroAPI::MaskReg tmpMask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL32>();
        if constexpr (solidPattern == 1) {
            MicroAPI::Duplicate(patternReg, 0x55);
        } else if constexpr (solidPattern == 2) {
            MicroAPI::Duplicate(patternReg, 0xaa);
        } else if constexpr (solidPattern == 3) {
            MicroAPI::Duplicate(patternReg, 0x11);
        } else if constexpr (solidPattern == 4) {
            MicroAPI::Duplicate(patternReg, 0x22);
        } else if constexpr (solidPattern == 5) {
            MicroAPI::Duplicate(patternReg, 0x44);
        } else if constexpr (solidPattern == 6) {
            MicroAPI::Duplicate(patternReg, 0x88);
        }
        MicroAPI::StoreAlign(tempBuf, patternReg, tmpMask);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::LoadAlign(patternMask, tempBuf);
    }
    MicroAPI::MaskAnd(patternMask, patternMask, loadMask, loadMask);
    for (uint16_t i = 0; i < reducev2Params.repeatTimes; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            srcReg, src0, reducev2Params.src0BlockStride, reducev2Params.src0RepeatStride, loadMask);
        if constexpr (SupportType<T, bfloat16_t>()) {
            MicroAPI::GatherMask<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(
                (MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)srcReg, patternMask);
        } else {
            MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(dstReg, srcReg, patternMask);
        }
        MicroAPI::StoreUnAlign(dst, dstReg, ureg);
    }
    MicroAPI::StoreUnAlignPost(dst, ureg);
}

template <typename T, uint8_t solidPattern>
__aicore__ inline void SetVectorMaskForGatherMaskSqueeze()
{
    if constexpr (sizeof(T) != 1) {
        if constexpr (sizeof(T) == 2) {
            if constexpr (solidPattern == 1) {
                SetVectorMask<T>(0x5555555555555555, 0x5555555555555555);
            } else if constexpr (solidPattern == 2) {
                SetVectorMask<T>(0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa);
            } else if constexpr (solidPattern == 3) {
                SetVectorMask<T>(0x1111111111111111, 0x1111111111111111);
            } else if constexpr (solidPattern == 4) {
                SetVectorMask<T>(0x2222222222222222, 0x2222222222222222);
            } else if constexpr (solidPattern == 5) {
                SetVectorMask<T>(0x4444444444444444, 0x4444444444444444);
            } else if constexpr (solidPattern == 6) {
                SetVectorMask<T>(0x8888888888888888, 0x8888888888888888);
            }
        } else {
            if constexpr (solidPattern == 1) {
                SetVectorMask<T>(0, 0x5555555555555555);
            } else if constexpr (solidPattern == 2) {
                SetVectorMask<T>(0, 0xaaaaaaaaaaaaaaaa);
            } else if constexpr (solidPattern == 3) {
                SetVectorMask<T>(0, 0x1111111111111111);
            } else if constexpr (solidPattern == 4) {
                SetVectorMask<T>(0, 0x2222222222222222);
            } else if constexpr (solidPattern == 5) {
                SetVectorMask<T>(0, 0x4444444444444444);
            } else if constexpr (solidPattern == 6) {
                SetVectorMask<T>(0, 0x8888888888888888);
            }
        }
    }
}

template <typename T, uint8_t solidPattern>
__aicore__ inline void GatherMaskSqueezeNormal(
    __ubuf__ T *dst, __ubuf__ T *src0, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    SetVectorMaskForGatherMaskSqueeze<T, solidPattern>();
    __ubuf__ uint8_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(GetRuntimeUBSize(), 32);
    GatherMaskSqueezeNormalTmpBuffer<T, solidPattern>(dst, src0, tempBuf, reducev2Params);
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(T);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T, uint8_t solidPattern>
__simd_vf__ inline void GatherMaskSqueezeReduceTmpBuffer(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ uint8_t *tempBuf,
    const uint32_t mask, const GatherMaskParams reducev2Params, uint8_t ElePerBlkT,uint16_t innerRepeatTimes)
{
    uint32_t maskValue = mask;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::MaskReg loadMask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::MaskReg patternMask;
    MicroAPI::MaskReg executeMask;
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    if constexpr (sizeof(T) != 1) {
        patternMask = MicroAPI::MoveMask<T>();
    } else {
        MicroAPI::RegTensor<uint8_t> reducePatternReg;
        MicroAPI::MaskReg tmpMask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL32>();
        if constexpr (solidPattern == 1) {
            MicroAPI::Duplicate(reducePatternReg, 0x55);
        } else if constexpr (solidPattern == 2) {
            MicroAPI::Duplicate(reducePatternReg, 0xaa);
        } else if constexpr (solidPattern == 3) {
            MicroAPI::Duplicate(reducePatternReg, 0x11);
        } else if constexpr (solidPattern == 4) {
            MicroAPI::Duplicate(reducePatternReg, 0x22);
        } else if constexpr (solidPattern == 5) {
            MicroAPI::Duplicate(reducePatternReg, 0x44);
        } else if constexpr (solidPattern == 6) {
            MicroAPI::Duplicate(reducePatternReg, 0x88);
        }
        MicroAPI::StoreAlign(tempBuf, reducePatternReg, tmpMask);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::LoadAlign(patternMask, tempBuf);
    }
    for (uint16_t i = 0; i < reducev2Params.repeatTimes; ++i) {
        maskValue = mask;
        for (uint16_t j = 0; j < innerRepeatTimes; ++j) {
            loadMask = MicroAPI::UpdateMask<T>(maskValue);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg,
                src0 + i * reducev2Params.src0RepeatStride * ElePerBlkT +
                    j * 8 * reducev2Params.src0BlockStride * ElePerBlkT,
                reducev2Params.src0BlockStride,
                loadMask);
            MicroAPI::MaskAnd(executeMask, patternMask, loadMask, loadMask);
            if constexpr (SupportType<T, bfloat16_t>()) {
                MicroAPI::GatherMask<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(
                    (MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)srcReg, executeMask);
            } else {
                MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(dstReg, srcReg, executeMask);
            }
            MicroAPI::StoreUnAlign(dst, dstReg, ureg);
        }
    }
    MicroAPI::StoreUnAlignPost(dst, ureg);
}

template <typename T, uint8_t solidPattern>
__aicore__ inline void GatherMaskSqueezeReduce(
    __ubuf__ T *dst, __ubuf__ T *src0, const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    SetVectorMaskForGatherMaskSqueeze<T, solidPattern>();
    __ubuf__ uint8_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(GetRuntimeUBSize(), 32);
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t ElePerVec = GetVecLen() / sizeof(T);
    uint16_t innerRepeatTimes = CeilDivision(mask, ElePerVec);
    GatherMaskSqueezeReduceTmpBuffer<T, solidPattern>(dst, src0, tempBuf, mask,
        reducev2Params, ElePerBlkT, innerRepeatTimes);
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(T);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T>
__aicore__ inline void GatherMaskAll(__ubuf__ T *dst, __ubuf__ T *src0, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    if (reduceMode) {
        GatherMaskAllReduce(dst, src0, mask, reducev2Params, rsvdCnt);
    } else {
        GatherMaskAllNormal(dst, src0, reducev2Params, rsvdCnt);
    }
}

template <typename T, uint8_t solidPattern>
__aicore__ inline void GatherMaskSqueeze(__ubuf__ T *dst, __ubuf__ T *src0, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    if (reduceMode) {
        GatherMaskSqueezeReduce<T, solidPattern>(dst, src0, mask, reducev2Params, rsvdCnt);
    } else {
        GatherMaskSqueezeNormal<T, solidPattern>(dst, src0, reducev2Params, rsvdCnt);
    }
}

template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T *dst, __ubuf__ T *src0, const uint8_t src1Pattern,
    const bool reduceMode, const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    if (src1Pattern == 1) {
        GatherMaskSqueeze<T, 1>(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else if (src1Pattern == 2) {
        GatherMaskSqueeze<T, 2>(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else if (src1Pattern == 3) {
        GatherMaskSqueeze<T, 3>(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else if (src1Pattern == 4) {
        GatherMaskSqueeze<T, 4>(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else if (src1Pattern == 5) {
        GatherMaskSqueeze<T, 5>(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else if (src1Pattern == 6) {
        GatherMaskSqueeze<T, 6>(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else if (src1Pattern == 7) {
        GatherMaskAll(dst, src0, reduceMode, mask, reducev2Params, rsvdCnt);
    } else {
        ASCENDC_ASSERT((false), "GatherMask Pattern can only be 1~7");
    }
}

template <typename T, typename U>
__simd_vf__ inline void GatherMaskReduceElePerVec(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ U* src1,
    const uint32_t mask, const GatherMaskParams reducev2Params, uint16_t innerRepeatTimes)
{
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint8_t ElePerBlkU = GetDataBlockSizeInBytes() / sizeof(U);
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::MaskReg loadMask;
    MicroAPI::MaskReg patternMask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::UnalignReg maskUreg;
    MicroAPI::RegTensor<U> patternReg;
    uint32_t maskValue = mask;
    uint32_t oneRepMaskOffset = reducev2Params.src1RepeatStride * ElePerBlkU;
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < reducev2Params.repeatTimes; ++i) {
        maskValue = mask;
        for (uint16_t j = 0; j < innerRepeatTimes; ++j) {
            loadMask = MicroAPI::UpdateMask<T>(maskValue);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg,
                src0 + i * reducev2Params.src0RepeatStride * ElePerBlkT + j * 8 * reducev2Params.src0BlockStride * ElePerBlkT,
                reducev2Params.src0BlockStride,
                loadMask);
            if constexpr (sizeof(T) == 1) { // 1bit in ub, 1bit in register
                MicroAPI::LoadAlign(patternMask, src1 + i * oneRepMaskOffset + j * ElePerBlkU);
            } else if constexpr (sizeof(T) == 2) { // 1bit in ub, us to 2bit in register
                MicroAPI::LoadAlign<U, MicroAPI::MaskDist::DIST_US>(patternMask, src1 + i * oneRepMaskOffset + j * ElePerBlkU / sizeof(T));
            } else if constexpr (sizeof(T) == 4) { // 1bit in ub, us to 4bit in register
                MicroAPI::LoadUnAlignPre(maskUreg, src1 + i * oneRepMaskOffset + j * ElePerBlkU / sizeof(T));
                MicroAPI::LoadUnAlign(patternReg, maskUreg, src1 + i * oneRepMaskOffset + j * ElePerBlkU / sizeof(T));
                MicroAPI::MaskGenWithRegTensor<U, 0>(patternMask, patternReg);
            }
            MicroAPI::MaskAnd(patternMask, patternMask, loadMask, loadMask);
            if constexpr (SupportType<T, bfloat16_t>()) {
                MicroAPI::GatherMask<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(
                    (MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)srcReg, patternMask);
            } else {
                MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(dstReg, srcReg, patternMask);
            }
            MicroAPI::StoreUnAlign(dst, dstReg, ureg);
        }
    }
    MicroAPI::StoreUnAlignPost(dst, ureg);
}

template <typename T, typename U>
__aicore__ inline void GatherMaskReduce(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ U* src1,
    const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    constexpr uint32_t ElePerVec = GetVecLen() / sizeof(T);
    uint16_t innerRepeatTimes = CeilDivision(mask, ElePerVec);
    GatherMaskReduceElePerVec<T, U>(dst, src0, src1, mask, reducev2Params, innerRepeatTimes);
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(T);
}

template <typename T, typename U>
__simd_vf__ inline void GatherMaskCalOneRepMask(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ U* src1,
    const GatherMaskParams reducev2Params, uint32_t oneRepMaskOffset)
{
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::MaskReg loadMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg patternMask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::UnalignReg maskUreg;
    MicroAPI::RegTensor<U> patternReg;
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    constexpr uint8_t ElePerBlk = GetDataBlockSizeInBytes() / sizeof(T);
    for (uint16_t i = 0; i < reducev2Params.repeatTimes; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg,
            src0 + i * reducev2Params.src0RepeatStride * ElePerBlk, reducev2Params.src0BlockStride, loadMask);
        if constexpr (sizeof(T) == 1) {
            MicroAPI::LoadAlign(patternMask, src1 + i * oneRepMaskOffset);
        } else if constexpr (sizeof(T) == 2) {
            MicroAPI::LoadAlign<U, MicroAPI::MaskDist::DIST_US>(patternMask, src1 + i * oneRepMaskOffset);
        } else if constexpr (sizeof(T) == 4) {
            MicroAPI::LoadUnAlignPre(maskUreg, src1 + i * oneRepMaskOffset);
            MicroAPI::LoadUnAlign(patternReg, maskUreg, src1 + i * oneRepMaskOffset);
            MicroAPI::MaskGenWithRegTensor<U, 0>(patternMask, patternReg);
        }
        MicroAPI::MaskAnd(patternMask, patternMask, loadMask, loadMask);
        if constexpr (SupportType<T, bfloat16_t>()) {
            MicroAPI::GatherMask<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(
                (MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)srcReg, patternMask);
        } else {
            MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(dstReg, srcReg, patternMask);
        }
        MicroAPI::StoreUnAlign(dst, dstReg, ureg);
    }
    MicroAPI::StoreUnAlignPost(dst, ureg);
}

template <typename T, typename U>
__aicore__ inline void GatherMaskCal(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ U* src1,
    const bool reduceMode, const uint32_t mask, const GatherMaskParams &reducev2Params, uint64_t &rsvdCnt)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, uint8_t, int8_t, bfloat16_t>(),
        "GatherMask only support half/int16_t/uint16_t/int32_t/uint32_t/float/bfloat16_t/int8_t/uint8_t"
        "data type on current device");
    static_assert(SupportType<U, uint8_t, uint16_t, uint32_t>(),
        "GatherMask only support uint8_t/uint16_t/uint32_t pattern type on current device");
    static_assert((sizeof(T) == 1 && IsSameType<U, uint8_t>::value) || (sizeof(T) == 2 && IsSameType<U, uint16_t>::value) ||
        (sizeof(T) == 4 && IsSameType<U, uint32_t>::value),
        "GatherMask only support int8_t/uint8_t data type with uint8_t pattern type, or"
        "GatherMask only support half/int16_t/uint16_t/bfloat16_t data type with uint16_t pattern type, or"
        "int32_t/uint32_t/float data type with uint32_t pattern type on current device");
    if (reduceMode) {
        GatherMaskReduce<T, U>(dst, src0, src1, mask, reducev2Params, rsvdCnt);
        return;
    }
    uint32_t oneRepMaskOffset = reducev2Params.src1RepeatStride * GetDataBlockSizeInBytes() / sizeof(T);
    GatherMaskCalOneRepMask<T, U>(dst, src0,  src1, reducev2Params, oneRepMaskOffset);
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(T);
}

template <typename T>
__simd_vf__ inline void ExtractVf(__ubuf__ T* dstValueLocal, __ubuf__ uint32_t* dstIndexLocal,
    __ubuf__ T* sortedLocal, const int32_t repeatTime)
{
    uint16_t loopTimes = static_cast<uint16_t>(repeatTime / 2);
    uint16_t tail = repeatTime % 2;
    if constexpr (SupportType<T, float>()) {
        MicroAPI::RegTensor<float> vreg0;
        MicroAPI::RegTensor<float> vreg1;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float>();
        uint32_t repeatElm = GetVecLen() / sizeof(float);
        for (uint16_t i = 0; i < loopTimes; ++i) {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg0, vreg1, sortedLocal + i * repeatElm * 2);
            MicroAPI::StoreAlign(dstValueLocal + i * repeatElm, vreg0, preg);
            MicroAPI::StoreAlign(dstIndexLocal + i * repeatElm, (MicroAPI::RegTensor<uint32_t> &)vreg1, preg);
        }
        for (uint16_t i = 0; i < tail; ++i) {
            MicroAPI::LoadAlign(vreg0, sortedLocal + repeatTime / 2 * repeatElm * 2);
            MicroAPI::LoadAlign(vreg1, sortedLocal + repeatTime / 2 * repeatElm * 2);
            MicroAPI::DeInterleave(vreg0, vreg1, vreg0, vreg1);
            preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
            MicroAPI::StoreAlign(dstValueLocal + repeatTime / 2 * repeatElm, vreg0, preg);
            MicroAPI::StoreAlign(
                dstIndexLocal + repeatTime / 2 * repeatElm, (MicroAPI::RegTensor<uint32_t> &)vreg1, preg);
        }
    } else if constexpr (SupportType<T, half>()) {
        MicroAPI::RegTensor<float> vreg0;
        MicroAPI::RegTensor<float> vreg1;
        MicroAPI::RegTensor<half> vreg2;
        MicroAPI::MaskReg indexPreg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::H>();
        uint32_t repeatElm = GetVecLen() / sizeof(float);
        for (uint16_t i = 0; i < loopTimes; ++i) {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg0, vreg1, (__ubuf__ float *)sortedLocal + i * repeatElm * 2);
            MicroAPI::Squeeze<half, MicroAPI::GatherMaskMode::NO_STORE_REG>(vreg2, (MicroAPI::RegTensor<half> &)vreg0, indexPreg);
            MicroAPI::StoreAlign(dstValueLocal + i * repeatElm, vreg2, preg1);
            MicroAPI::StoreAlign(dstIndexLocal + i * repeatElm, (MicroAPI::RegTensor<uint32_t> &)vreg1, indexPreg);
        }
        for (uint16_t i = 0; i < tail; ++i) {
            MicroAPI::LoadAlign(vreg0, (__ubuf__ float *)sortedLocal + repeatTime / 2 * repeatElm * 2);
            MicroAPI::LoadAlign(vreg1, (__ubuf__ float *)sortedLocal + repeatTime / 2 * repeatElm * 2);
            MicroAPI::DeInterleave(vreg0, vreg1, vreg0, vreg1);
            MicroAPI::Squeeze<half, MicroAPI::GatherMaskMode::NO_STORE_REG>(vreg2, (MicroAPI::RegTensor<half> &)vreg0, indexPreg);
            MicroAPI::MaskReg preg2 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::Q>();
            MicroAPI::StoreAlign(dstValueLocal + repeatTime / 2 * repeatElm, vreg2, preg2);
            preg2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::H>();
            MicroAPI::StoreAlign(
                dstIndexLocal + repeatTime / 2 * repeatElm, (MicroAPI::RegTensor<uint32_t> &)vreg1, preg2);
        }
    }
}

template <typename T>
__aicore__ inline void ExtractImpl(__ubuf__ T* dstValueLocal, __ubuf__ uint32_t* dstIndexLocal,
    __ubuf__ T* sortedLocal, const int32_t repeatTime)
{
    ExtractVf<T>(dstValueLocal, dstIndexLocal, sortedLocal, repeatTime);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_MASK_IMPL_H
