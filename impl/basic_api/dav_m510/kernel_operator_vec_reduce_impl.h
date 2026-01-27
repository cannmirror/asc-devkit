/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H

#include "micro_api/kernel_micro_intf.h"
#include "kernel_operator_vec_template_impl.h"

namespace AscendC {
template <bool isBitMask, typename T>
__aicore__ inline void GenPredicate(MicroAPI::MaskReg &preg, uint32_t maskReg)
{
    if constexpr (isBitMask) {
        preg = MicroAPI::MoveMask<T>();
    } else {
        preg = MicroAPI::UpdateMask<T>(maskReg);
    }
}

template <bool isSetMask, bool isBitMask, bool isCounterMode, typename T>
__aicore__ inline void ReduceCommonCall(MicroAPI::MaskReg& mask, uint16_t& newRepeatTimes, uint32_t& countSreg,
                                        uint32_t maskReg, __ubuf__ uint64_t* maskBuf)
{
    if constexpr (isCounterMode) {
        if constexpr (!isSetMask) {
            // get SPR.MASK in VF
            MicroAPI::MaskReg sprLoadMaskReg = MicroAPI::MoveMask<uint16_t>();
            MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(maskBuf, sprLoadMaskReg);
            // insert membar(vec store operation) before load maskBuf[0](scalar load operation)
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
            countSreg = static_cast<uint32_t>(maskBuf[0]);
        }
        constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
        newRepeatTimes = CeilDivision(countSreg, oneRepSize);
    } else {
        if constexpr (isBitMask) {  // mask[]
            mask = MicroAPI::MoveMask<T>();
        } else {  // mask
            if constexpr (!isSetMask) {
                mask = MicroAPI::MoveMask<T>();
            } else {
                mask = MicroAPI::UpdateMask<T>(maskReg);
            }
        }
    }
}

template <bool isSetMask, bool isBitMask, bool isCounterMode, auto func, typename T>
__aicore__ inline void ReduceAlignCall(__ubuf__ T *dst, __ubuf__ T *src, int32_t repeat, uint32_t dstRepOffset,
    uint32_t srcBlkStride, uint32_t srcRepStride, uint32_t maskReg, __ubuf__ uint64_t *maskBuf)
{
    MicroAPI::MaskReg stMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg mask;
    uint16_t newRepeatTimes = static_cast<uint16_t>(repeat);
    uint32_t countSreg = static_cast<uint32_t>(maskReg);
    ReduceCommonCall<isSetMask, isBitMask, isCounterMode, T>(mask, newRepeatTimes, countSreg, maskReg, maskBuf);
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            mask = MicroAPI::UpdateMask<T>(countSreg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            srcVreg, src, srcBlkStride, srcRepStride, mask);
        func(dstVreg, srcVreg, mask);
        MicroAPI::StoreAlign(dst + i * dstRepOffset, dstVreg, stMask);
    }
}

template <bool isSetMask, bool isBitMask, bool isCounterMode, bool withStride, auto func, typename T, typename U = T>
__aicore__ inline void ReduceUnalignCall(__ubuf__ U *dst, __ubuf__ T *src, int32_t repeat, uint32_t oneRepOffset,
    uint32_t dstRepOffsetPost, uint32_t srcBlkStride, uint32_t srcRepStride, uint32_t maskReg, __ubuf__ uint64_t *maskBuf)
{
    MicroAPI::MaskReg mask;
    uint16_t newRepeatTimes = static_cast<uint16_t>(repeat);
    uint32_t countSreg = static_cast<uint32_t>(maskReg);
    ReduceCommonCall<isSetMask, isBitMask, isCounterMode, T>(mask, newRepeatTimes, countSreg, maskReg, maskBuf);
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<U> dstVreg;
    MicroAPI::UnalignReg ureg;
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            mask = MicroAPI::UpdateMask<T>(countSreg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            srcVreg, src, srcBlkStride, srcRepStride, mask);
        func(dstVreg, srcVreg, mask);
        MicroAPI::StoreUnAlign(dst, dstVreg, ureg, oneRepOffset);
        if constexpr (withStride) {
            MicroAPI::StoreUnAlignPost(dst, ureg, dstRepOffsetPost);
        }
    }
    if constexpr (!withStride) {
        MicroAPI::StoreUnAlignPost(dst, ureg, dstRepOffsetPost);
    }
}

template <bool isSetMask, bool isBitMask, auto func, typename T>
__aicore__ inline void PairReduceTemplate(__ubuf__ T *dst, __ubuf__ T *src, int32_t repeat, int32_t dstRepStride,
    uint32_t oneRepOffset, int32_t srcBlkStride, int32_t srcRepStride, uint32_t maskReg)
{
    constexpr uint32_t ONE_BLK_ELEMENT_NUM = GetDataBlockSizeInBytes() / sizeof(T);
    uint32_t dstRepOffset = oneRepOffset * dstRepStride;
    int32_t newRepeat = repeat;
    __ubuf__ T *newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        uint32_t srcStrideOffset = srcRepStride * ONE_BLK_ELEMENT_NUM;
        newSrc += (srcStrideOffset * (repeat - 1));
        newRepeat = 1;
    }
    bool isCounterMode = Internal::IsCounterMode();
     __ubuf__ uint64_t *maskBuf = nullptr;
    if (isCounterMode) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        }
        VF_CALL<ReduceAlignCall<isSetMask, isBitMask, true, func, T>>(
            dst, newSrc, newRepeat, dstRepOffset, srcBlkStride, srcRepStride, maskReg, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    } else {
        VF_CALL<ReduceAlignCall<isSetMask, isBitMask, false, func, T>>(
            dst, newSrc, newRepeat, dstRepOffset, srcBlkStride, srcRepStride, maskReg, maskBuf);
    }
}

template <bool isSetMask, bool isBitMask, auto func, typename T, typename U = T>
__aicore__ inline void ReduceTemplate(__ubuf__ U *dst, __ubuf__ T *src, int32_t repeat, int32_t dstRepStride,
    uint32_t oneRepOffset, int32_t srcBlkStride, int32_t srcRepStride, uint32_t maskReg)
{
    constexpr uint32_t ONE_BLK_ELEMENT_NUM = GetDataBlockSizeInBytes() / sizeof(T);
    bool isCounterMode = Internal::IsCounterMode();
    __ubuf__ uint64_t *maskBuf = nullptr;
    if (isCounterMode) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        }
        if (dstRepStride == 0 && repeat > 0) {
            uint32_t srcStrideOffset = srcRepStride * ONE_BLK_ELEMENT_NUM;
            __ubuf__ T *newSrc = src + srcStrideOffset * (repeat - 1);
            VF_CALL<ReduceUnalignCall<isSetMask, isBitMask, true, true, func, T, U>>(
                dst, newSrc, 1, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf);
        } else if (dstRepStride == 1 && repeat > 0) {
            VF_CALL<ReduceUnalignCall<isSetMask, isBitMask, true, false, func, T, U>>(
                dst, src, repeat, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf);
        } else {
            uint32_t dstRepOffsetPost = oneRepOffset * (dstRepStride - 1);
            VF_CALL<ReduceUnalignCall<isSetMask, isBitMask, true, true, func, T, U>>(
                dst, src, repeat, oneRepOffset, dstRepOffsetPost, srcBlkStride, srcRepStride, maskReg, maskBuf);
        }
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    } else {
        if (dstRepStride == 0 && repeat > 0) {
            uint32_t srcStrideOffset = srcRepStride * ONE_BLK_ELEMENT_NUM;
            __ubuf__ T *newSrc = src + srcStrideOffset * (repeat - 1);
            VF_CALL<ReduceUnalignCall<isSetMask, isBitMask, false, true, func, T, U>>(
                dst, newSrc, 1, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf);
        } else if (dstRepStride == 1 && repeat > 0) {
            VF_CALL<ReduceUnalignCall<isSetMask, isBitMask, false, false, func, T, U>>(
                dst, src, repeat, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf);
        } else {
            uint32_t dstRepOffsetPost = oneRepOffset * (dstRepStride - 1);
            VF_CALL<ReduceUnalignCall<isSetMask, isBitMask, false, true, func, T, U>>(
                dst, src, repeat, oneRepOffset, dstRepOffsetPost, srcBlkStride, srcRepStride, maskReg, maskBuf);
        }
    }
}

/* **************************************** Pair Reduce Impl ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "PairReduceSum current data type is not supported!");
    constexpr uint32_t oneRepOffset = (ONE_REPEAT_BYTE_SIZE / sizeof(T)) / HALF_FACTOR;
    uint32_t maskReg = static_cast<uint32_t>(mask);
    PairReduceTemplate<isSetMask, false, MicroAPI::PairReduceSum<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, maskReg);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat, const uint64_t mask[],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "PairReduceSum current data type is not supported!");
    constexpr uint32_t oneRepOffset = (ONE_REPEAT_BYTE_SIZE / sizeof(T)) / HALF_FACTOR;
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    PairReduceTemplate<isSetMask, true, MicroAPI::PairReduceSum<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, mask[0]);
}

/* **************************************** Block Reduce Impl ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "BlockReduceSum not support current datatype!");
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    ReduceTemplate<isSetMask, true, MicroAPI::ReduceSumWithDataBlock<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, DEFAULT_BLK_NUM, srcBlkStride, srcRepStride, mask[0]);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "BlockReduceSum not support current datatype!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    ReduceTemplate<isSetMask, false,
        MicroAPI::ReduceSumWithDataBlock<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, DEFAULT_BLK_NUM, srcBlkStride, srcRepStride, maskReg);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "BlockReduceMax current data type is not supported!");
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    ReduceTemplate<isSetMask, true, MicroAPI::ReduceMaxWithDataBlock<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, DEFAULT_BLK_NUM, srcBlkStride, srcRepStride, mask[0]);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "BlockReduceMax current data type is not supported!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    ReduceTemplate<isSetMask, false,
        MicroAPI::ReduceMaxWithDataBlock<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, DEFAULT_BLK_NUM, srcBlkStride, srcRepStride, maskReg);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "BlockReduceMin not support current datatype!");
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    ReduceTemplate<isSetMask, true, MicroAPI::ReduceMinWithDataBlock<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, DEFAULT_BLK_NUM, srcBlkStride, srcRepStride, mask[0]);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ T *dst, __ubuf__ T *src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "BlockReduceMin not support current datatype!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    ReduceTemplate<isSetMask, false,
        MicroAPI::ReduceMinWithDataBlock<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dst, src, repeat, dstRepStride, DEFAULT_BLK_NUM, srcBlkStride, srcRepStride, maskReg);
}

template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void RepeatReduceSumImpl(__ubuf__ U *dstLocal, __ubuf__ T *srcLocal, const int32_t repeat,
    const int32_t elemsInOneRepeat, const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride,
    const int32_t srcRepStride)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "RepeatReduceSum current data type is not supported!");
    static_assert(
        (SupportType<U, int32_t, uint32_t, half, float>()), "RepeatReduceSum current data type is not supported!");
    uint32_t maskReg = static_cast<uint32_t>(elemsInOneRepeat);
    ReduceTemplate<isSetMask, false,
        MicroAPI::ReduceSum<U, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<U>, MicroAPI::RegTensor<T>>,
        T,
        U>(dstLocal, srcLocal, repeat, dstRepStride, 1, srcBlkStride, srcRepStride, maskReg);
}

/* **************************************** Whole Reduce Interface ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceMax current data type is not supported!");
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    uint32_t oneRepOffset = (order == ReduceOrder::ORDER_VALUE_INDEX) ? 2 : 1;
    ReduceTemplate<isSetMask, true, MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dstLocal, srcLocal, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, mask[0]);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceMax current data type is not supported!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    uint32_t oneRepOffset = (order == ReduceOrder::ORDER_VALUE_INDEX) ? 2 : 1;
    ReduceTemplate<isSetMask, false, MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dstLocal, srcLocal, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, maskReg);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceMin current data type is not supported!");
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    uint32_t oneRepOffset = (order == ReduceOrder::ORDER_VALUE_INDEX) ? 2 : 1;
    ReduceTemplate<isSetMask, true, MicroAPI::ReduceMin<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dstLocal, srcLocal, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, mask[0]);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceMin current data type is not supported!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    uint32_t oneRepOffset = (order == ReduceOrder::ORDER_VALUE_INDEX) ? 2 : 1;
    ReduceTemplate<isSetMask, false, MicroAPI::ReduceMin<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dstLocal, srcLocal, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, maskReg);
}

template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ U *dstLocal, __ubuf__ T *srcLocal, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceSum current data type is not supported!");
    static_assert(
        (SupportType<U, int32_t, uint32_t, half, float>()), "WholeReduceSum current data type is not supported!");
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    ReduceTemplate<isSetMask, true,
        MicroAPI::ReduceSum<U, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<U>, MicroAPI::RegTensor<T>>,
        T,
        U>(dstLocal, srcLocal, repeat, dstRepStride, 1, srcBlkStride, srcRepStride, mask[0]);
}

template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ U *dstLocal, __ubuf__ T *srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceSum current data type is not supported!");
    static_assert(
        (SupportType<U, int32_t, uint32_t, half, float>()), "WholeReduceSum current data type is not supported!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    ReduceTemplate<isSetMask, false,
        MicroAPI::ReduceSum<U, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<U>, MicroAPI::RegTensor<T>>,
        T,
        U>(dstLocal, srcLocal, repeat, dstRepStride, 1, srcBlkStride, srcRepStride, maskReg);
}

/* **************************************** Reduce Interface ****************************************** */
template <typename T>
__aicore__ inline void ReduceSumCount(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, uint32_t count, int32_t repeat, const int32_t srcRepStride)
{
    uint32_t srcRepOffset = srcRepStride * GetDataBlockSizeInBytes() / sizeof(T);
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::UnalignReg ureg;
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        preg = MicroAPI::UpdateMask<T>(count);
        MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
        MicroAPI::ReduceSum(dstVreg, srcVreg, preg);
        MicroAPI::StoreUnAlign(dstLocal, dstVreg, ureg, 1);
    }
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <typename T, bool isBitMask>
__aicore__ inline void ReduceSumMask(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, uint32_t mask, int32_t repeat, const int32_t srcRepStride)
{
    uint32_t srcRepOffset = srcRepStride * GetDataBlockSizeInBytes() / sizeof(T);
    MicroAPI::MaskReg preg;
    GenPredicate<isBitMask, T>(preg, mask);
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::UnalignReg ureg;
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
        MicroAPI::ReduceSum(dstVreg, srcVreg, preg);
        MicroAPI::StoreUnAlign(dstLocal, dstVreg, ureg, 1);
    }
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <typename T, int shapeScope>
__aicore__ inline void ReduceSumCounterMode(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count, const int32_t srcRepStride)
{
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    if constexpr (shapeScope == 1) {
        ReduceSumCount(dstLocal, srcLocal, count, 1, srcRepStride);
    } else if constexpr (shapeScope == 2) {
        uint32_t count2 = CeilDivision(count, oneRepSize);
        ReduceSumCount(workLocal, srcLocal, count, count2, srcRepStride);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ReduceSumCount(dstLocal, workLocal, count2, 1, 8);
    } else {
        uint32_t count2 = CeilDivision(count, oneRepSize);
        uint32_t count3 = CeilDivision(count2, oneRepSize);
        ReduceSumCount(workLocal, srcLocal, count, count2, srcRepStride);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ReduceSumCount(workLocal, workLocal, count2, count3, 8);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ReduceSumCount(dstLocal, workLocal, count3, 1, 8);
    }
}

template <typename T, int shapeScope, bool isBitMask>
__aicore__ inline void ReduceSumNormalMode(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal,  __ubuf__ T *workLocal, uint32_t mask, int32_t repeat, const int32_t srcRepStride)
{
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    if constexpr (shapeScope == 1) {
        ReduceSumMask<T, isBitMask>(dstLocal, srcLocal, mask, 1, srcRepStride);
    } else if constexpr (shapeScope == 2) {
        ReduceSumMask<T, isBitMask>(workLocal, srcLocal, mask, repeat, srcRepStride);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ReduceSumCount(dstLocal, workLocal, repeat, 1, 8);
    } else {
        uint32_t count = CeilDivision(repeat, oneRepSize);
        ReduceSumMask<T, isBitMask>(workLocal, srcLocal, mask, repeat, srcRepStride);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ReduceSumCount(workLocal, workLocal, repeat, count, 8);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ReduceSumCount(dstLocal, workLocal, count, 1, 8);
    }
}

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    const uint64_t mask[], const int32_t repeat, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "ReduceSum current data type is not supported!");
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        uint32_t count = static_cast<uint32_t>(mask[0]);
        if (count <= oneRepSize) {
            VF_CALL<ReduceSumCounterMode<T, 1>>(dstLocal, srcLocal, workLocal, count, srcRepStride);
        } else if (count <= oneRepSize * oneRepSize) {
            VF_CALL<ReduceSumCounterMode<T, 2>>(dstLocal, srcLocal, workLocal, count, srcRepStride);
        } else {
            VF_CALL<ReduceSumCounterMode<T, 3>>(dstLocal, srcLocal, workLocal, count, srcRepStride);
        }
    } else {
        SetVectorMask<T>(mask[1], mask[0]);
        if (repeat <= 1) {
            VF_CALL<ReduceSumNormalMode<T, 1, true>>(dstLocal, srcLocal, workLocal, 0, 1, srcRepStride);
        } else if (repeat <= oneRepSize) {
            VF_CALL<ReduceSumNormalMode<T, 2, true>>(dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride);
        } else {
            VF_CALL<ReduceSumNormalMode<T, 3, true>>(dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    const int32_t mask, const int32_t repeat, const int32_t srcRepStride)
{
    static_assert((SupportType<T, half, float>()), "ReduceSum current data type is not supported!");
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        uint32_t count = static_cast<uint32_t>(mask);
        if (count <= oneRepSize) {
            VF_CALL<ReduceSumCounterMode<T, 1>>(dstLocal, srcLocal, workLocal, count, srcRepStride);
        } else if (count <= oneRepSize * oneRepSize) {
            VF_CALL<ReduceSumCounterMode<T, 2>>(dstLocal, srcLocal, workLocal, count, srcRepStride);
        } else {
            VF_CALL<ReduceSumCounterMode<T, 3>>(dstLocal, srcLocal, workLocal, count, srcRepStride);
        }
    } else {
        if (repeat <= 1) {
            VF_CALL<ReduceSumNormalMode<T, 1, false>>(dstLocal, srcLocal, workLocal, mask, 1, srcRepStride);
        } else if (repeat <= oneRepSize) {
            VF_CALL<ReduceSumNormalMode<T, 2, false>>(dstLocal, srcLocal, workLocal, mask, repeat, srcRepStride);
        } else {
            VF_CALL<ReduceSumNormalMode<T, 3, false>>(dstLocal, srcLocal, workLocal, mask, repeat, srcRepStride);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceB64SumImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count)
{
    constexpr uint32_t oneRepSize = 2 * GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(count, oneRepSize);
    uint32_t sreg = count;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vregDup;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vregTmp;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vreg0;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vreg1;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, MicroAPI::RegTraitNumTwo>();
    MicroAPI::MaskReg mask;
    MicroAPI::Duplicate(vregDup, T(0), fullMask);

    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        MicroAPI::LoadAlign(vreg0, srcLocal + i * oneRepSize);
        MicroAPI::Add(vregTmp, vregDup, vreg0, mask);
        MicroAPI::Select(vregDup, vregTmp, vregDup, mask);
    }
    MicroAPI::ReduceSum(vreg1, vregDup, fullMask);
    MicroAPI::MaskReg maskFirstVal = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1, MicroAPI::RegTraitNumTwo>();
    MicroAPI::StoreAlign(dstLocal, vreg1, maskFirstVal);
}

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count)
{
    static_assert((SupportType<T, half, float, uint64_t, int64_t>()), "ReduceSum current data type is not supported!");
    if constexpr (SupportType<T, uint64_t, int64_t>()) {
        VF_CALL<ReduceB64SumImpl<T>>(dstLocal, srcLocal, workLocal, count);
    } else {
        constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
        if (count <= oneRepSize) {
            VF_CALL<ReduceSumCounterMode<T, 1>>(dstLocal, srcLocal, workLocal, count, 8);
        } else if (count <= oneRepSize * oneRepSize) {
            VF_CALL<ReduceSumCounterMode<T, 2>>(dstLocal, srcLocal, workLocal, count, 8);
        } else {
            VF_CALL<ReduceSumCounterMode<T, 3>>(dstLocal, srcLocal, workLocal, count, 8);
        }
    }
}

/***************************** Reduce Max & Min ******************/
template <typename T>
__aicore__ inline T GetMinValue()
{
    if constexpr (std::is_same_v<T, half>) {
        return GetScalarBitcodeValue<uint16_t, T>(0xFBFF);
    } else if constexpr (std::is_same_v<T, float>) {
        return GetScalarBitcodeValue<uint32_t, T>(0xFF7FFFFF);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return 0;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return 0x8000;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return 0;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return 0x80000000;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return 0;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return 0x8000000000000000;
    }
}

template <typename T>
__aicore__ inline T GetMaxValue()
{
    if constexpr (std::is_same_v<T, half>) {
        return GetScalarBitcodeValue<uint16_t, T>(0x7BFF);
    } else if constexpr (std::is_same_v<T, float>) {
        return GetScalarBitcodeValue<uint32_t, T>(0x7F7FFFFF);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return 0xFFFF;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return 0x7FFF;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return 0xFFFFFFFF;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return 0x7FFFFFFF;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return 0xFFFFFFFFFFFFFFFF;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return 0x7FFFFFFFFFFFFFFF;
    }
}

template <ReduceMode mode, typename T>
__aicore__ inline void ReduceNoIndexTemplate(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    uint32_t count, const int32_t srcRepStride, T initValue)
{
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t repeat = CeilDivision(count, oneRepSize);
    uint32_t srcRepOffset = srcRepStride * GetDataBlockSizeInBytes() / sizeof(T);
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T> srcVreg, dstVreg, tmpVreg;
    MicroAPI::UnalignReg ureg;
    MicroAPI::Duplicate(dstVreg, initValue);
    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<T>(count);
        MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
        if constexpr (mode == ReduceMode::REDUCE_MAX) {
            MicroAPI::Max(tmpVreg, dstVreg, srcVreg, preg);
        } else {
            MicroAPI::Min(tmpVreg, dstVreg, srcVreg, preg);
        }
        // merge new masked tmpVreg to dstVreg, keep non-masked old value in dstVreg
        MicroAPI::Select(dstVreg, tmpVreg, dstVreg, preg);
    }
    if constexpr (mode == ReduceMode::REDUCE_MAX) {
        MicroAPI::ReduceMax(dstVreg, dstVreg, pregFull);
    } else {
        MicroAPI::ReduceMin(dstVreg, dstVreg, pregFull);
    }
    MicroAPI::StoreUnAlign(dstLocal, dstVreg, ureg, 1);
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <ReduceMode mode, typename T>
__aicore__ inline void ReduceB64NoIndexTemplate(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count, T initValue)
{
    constexpr uint16_t oneRepSize = 2 * GetVecLen() / sizeof(T);
    uint16_t repeat = CeilDivision(count, oneRepSize);
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, MicroAPI::RegTraitNumTwo>();
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> b64SrcVreg, b64DstVreg, b64TmpVreg;
    MicroAPI::UnalignReg ureg;
    MicroAPI::Duplicate(b64DstVreg, initValue, pregFull);
    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(count);
        MicroAPI::LoadAlign(b64SrcVreg, srcLocal + i * oneRepSize);
        if constexpr (mode == ReduceMode::REDUCE_MAX) {
            MicroAPI::Max(b64TmpVreg, b64DstVreg, b64SrcVreg, preg);
        } else {
            MicroAPI::Min(b64TmpVreg, b64DstVreg, b64SrcVreg, preg);
        }
        // merge new masked b64TmpVreg to b64DstVreg, keep non-masked old value in b64DstVreg
        MicroAPI::Select(b64DstVreg, b64TmpVreg, b64DstVreg, preg);
    }
    if constexpr (mode == ReduceMode::REDUCE_MAX) {
        MicroAPI::ReduceMax(b64DstVreg, b64DstVreg, pregFull);
    } else {
        MicroAPI::ReduceMin(b64DstVreg, b64DstVreg, pregFull);
    }
    MicroAPI::StoreUnAlign(dstLocal, b64DstVreg, ureg, 1);
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <ReduceMode mode, bool isBitMask, typename T>
__aicore__ inline void ReduceNoIndexTemplate(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    uint32_t maskReg, const int32_t repeat, const int32_t srcRepStride, T initValue)
{
    MicroAPI::MaskReg preg;
    GenPredicate<isBitMask, T>(preg, maskReg);
    MicroAPI::RegTensor<T> srcVreg, dstVreg;
    MicroAPI::UnalignReg ureg;
    MicroAPI::Duplicate(dstVreg, initValue);
    int32_t postUpdateStride = srcRepStride * GetDataBlockSizeInBytes() / sizeof(T);
    for (uint16_t i = 0; i < repeat; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, postUpdateStride);
        if constexpr (mode == ReduceMode::REDUCE_MAX) {
            MicroAPI::Max(dstVreg, dstVreg, srcVreg, preg);
        } else {
            MicroAPI::Min(dstVreg, dstVreg, srcVreg, preg);
        }
    }
    if constexpr (mode == ReduceMode::REDUCE_MAX) {
        MicroAPI::ReduceMax(dstVreg, dstVreg, preg);
    } else {
        MicroAPI::ReduceMin(dstVreg, dstVreg, preg);
    }
    MicroAPI::StoreUnAlign(dstLocal, dstVreg, ureg, 1);
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <ReduceMode mode, typename T, typename IndexT>
__aicore__ inline void ReduceIndexTemplate(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    uint32_t count, const int32_t srcRepStride, T initValue)
{
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t repeat = CeilDivision(count, oneRepSize);
    uint32_t srcRepOffset = srcRepStride * GetDataBlockSizeInBytes() / sizeof(T);
    MicroAPI::MaskReg preg, pregCond;
    MicroAPI::MaskReg pregIndexFull = MicroAPI::CreateMask<IndexT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::UnalignReg ureg;
    MicroAPI::RegTensor<T> srcVreg, dstValueVreg, tmpValueVreg;
    MicroAPI::RegTensor<IndexT> dstIndexVreg, subIndexVreg, tmpIndexVreg, maskIndexVreg;
    MicroAPI::Duplicate(subIndexVreg, (IndexT)1);
    MicroAPI::Duplicate(maskIndexVreg, (IndexT)0);
    MicroAPI::Duplicate(dstValueVreg, initValue);
    if constexpr (std::is_same_v<IndexT, uint16_t>) {
        MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)tmpIndexVreg, 1);
    } else {
        MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)tmpIndexVreg, 1);
    }
    dstIndexVreg = tmpIndexVreg;
    // step1: from [count] to [oneRepSize] value index pair
    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<T>(count);
        MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
        if constexpr (mode == ReduceMode::REDUCE_MAX) {
            MicroAPI::Max(tmpValueVreg, dstValueVreg, srcVreg, preg);
        } else {
            MicroAPI::Min(tmpValueVreg, dstValueVreg, srcVreg, preg);
        }
        // merge old non-masked masked dstValueVreg to tmpValue, keep masked new value in tmpValue
        // now tmpValueVreg is this round new value, dstValueVreg is previous round value
        MicroAPI::Select(tmpValueVreg, tmpValueVreg, dstValueVreg, preg);
        // if previous round and this round value is change, update index
        MicroAPI::Compare<T, CMPMODE::NE>(pregCond, dstValueVreg, tmpValueVreg, pregFull);
        MicroAPI::Select(dstIndexVreg, tmpIndexVreg, dstIndexVreg, pregCond);
        // make next round index
        MicroAPI::Adds(tmpIndexVreg, tmpIndexVreg, (IndexT)oneRepSize, pregFull);
        // update value
        dstValueVreg = tmpValueVreg;
    }
    // step2: from [oneRepSize] to [1] value index and store it to ub
    if constexpr (mode == ReduceMode::REDUCE_MAX) {
        MicroAPI::ReduceMax(tmpValueVreg, dstValueVreg, pregFull);
    } else {
        MicroAPI::ReduceMin(tmpValueVreg, dstValueVreg, pregFull);
    }
    MicroAPI::StoreUnAlign(dstLocal, tmpValueVreg, ureg, 1);  // store value
    // get dst value mask and squeeze dst index
    MicroAPI::Duplicate(tmpValueVreg, tmpValueVreg, pregFull);
    MicroAPI::Compare<T, CMPMODE::EQ>(pregCond, dstValueVreg, tmpValueVreg, pregFull);
    MicroAPI::GatherMask<IndexT, MicroAPI::GatherMaskMode::NO_STORE_REG>(tmpIndexVreg, dstIndexVreg, pregCond);
    // cal preg for how much index has the same max or min value
    MicroAPI::Compare<IndexT, CMPMODE::NE>(pregCond, tmpIndexVreg, maskIndexVreg, pregIndexFull);
    MicroAPI::ReduceMin(tmpIndexVreg, tmpIndexVreg, pregCond);
    MicroAPI::Sub(tmpIndexVreg, tmpIndexVreg, subIndexVreg, pregIndexFull);
    MicroAPI::StoreUnAlign((__ubuf__ IndexT *&)dstLocal, tmpIndexVreg, ureg, 1);
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <ReduceMode mode, typename T, typename IndexT>
__aicore__ inline void ReduceB64IndexTemplate(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count, T initValue)
{
    constexpr uint16_t oneRepSize = 2 * GetVecLen() / sizeof(T);
    uint16_t repeat = CeilDivision(count, oneRepSize);
    MicroAPI::MaskReg preg, pregCond;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, MicroAPI::RegTraitNumTwo>();
    MicroAPI::MaskReg pregIndexFull = MicroAPI::CreateMask<IndexT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> b64SrcVreg, b64DstValueVreg, b64TmpValueVreg;
    MicroAPI::RegTensor<IndexT> dstIndexVreg, tmpIndexVreg, maskIndexVreg, subIndexVreg;
    MicroAPI::UnalignReg ureg;
    MicroAPI::Duplicate(b64DstValueVreg, initValue, pregFull);
    MicroAPI::Duplicate(maskIndexVreg, (IndexT)0);
    MicroAPI::Duplicate(subIndexVreg, (IndexT)1);
    // b64 type, index is uint32_t
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)tmpIndexVreg, 1);
    dstIndexVreg = tmpIndexVreg;
    // step1: from [count] to [oneRepSize] value index pair
    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(count);
        MicroAPI::LoadAlign(b64SrcVreg, srcLocal + i * oneRepSize);
        if constexpr (mode == ReduceMode::REDUCE_MAX) {
            MicroAPI::Max(b64TmpValueVreg, b64DstValueVreg, b64SrcVreg, preg);
        } else {
            MicroAPI::Min(b64TmpValueVreg, b64DstValueVreg, b64SrcVreg, preg);
        }
        // merge old non-masked masked b64DstValueVreg to tmpValue, keep masked new value in tmpValue
        // now b64TmpValueVreg is this round new value, b64DstValueVreg is previous round value
        MicroAPI::Select(b64TmpValueVreg, b64TmpValueVreg, b64DstValueVreg, preg);
        // if previous round and this round value is change, update index
        MicroAPI::Compare<T, CMPMODE::NE>(pregCond, b64DstValueVreg, b64TmpValueVreg, pregFull);
        MicroAPI::Select(dstIndexVreg, tmpIndexVreg, dstIndexVreg, pregCond);
        // make next round index
        MicroAPI::Adds(tmpIndexVreg, tmpIndexVreg, (IndexT)oneRepSize, pregIndexFull);
        // update value
        b64DstValueVreg = b64TmpValueVreg;
    }
    // step2: from [oneRepSize] to [1] value index and store it to ub
    if constexpr (mode == ReduceMode::REDUCE_MAX) {
        MicroAPI::ReduceMax(b64TmpValueVreg, b64DstValueVreg, pregFull);
    } else {
        MicroAPI::ReduceMin(b64TmpValueVreg, b64DstValueVreg, pregFull);
    }
    MicroAPI::StoreUnAlign(dstLocal, b64TmpValueVreg, ureg, 1);  // store value
    // get dst value mask and squeeze dst index
    MicroAPI::Duplicate(b64TmpValueVreg, b64TmpValueVreg, pregFull);
    MicroAPI::Compare<T, CMPMODE::EQ>(pregCond, b64DstValueVreg, b64TmpValueVreg, pregFull);
    // gather mask index
    MicroAPI::GatherMask<IndexT, MicroAPI::GatherMaskMode::NO_STORE_REG>(tmpIndexVreg, dstIndexVreg, pregCond);
    // cal preg for how much index has the same max or min value
    MicroAPI::Compare<IndexT, CMPMODE::NE>(pregCond, tmpIndexVreg, maskIndexVreg, pregIndexFull);
    MicroAPI::ReduceMin(tmpIndexVreg, tmpIndexVreg, pregCond);
    MicroAPI::Sub(tmpIndexVreg, tmpIndexVreg, subIndexVreg, pregIndexFull);
    MicroAPI::StoreUnAlign((__ubuf__ IndexT *&)dstLocal, tmpIndexVreg, ureg, 1);
    // for b64 type, pad 0 to b64 bytes, which is 4 bytes
    MicroAPI::StoreUnAlign((__ubuf__ IndexT *&)dstLocal, maskIndexVreg, ureg, 1);
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <ReduceMode mode, bool isBitMask, typename T, typename IndexT>
__aicore__ inline void ReduceIndexTemplate(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    uint32_t maskReg, const int32_t repeat, const int32_t srcRepStride, T initValue)
{
    MicroAPI::MaskReg preg, pregCond;
    GenPredicate<isBitMask, T>(preg, maskReg);
    MicroAPI::MaskReg pregIndexFull = MicroAPI::CreateMask<IndexT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T> srcVreg, dstValueVreg, tmpValueVreg;
    MicroAPI::RegTensor<IndexT> dstIndexVreg, tmpIndexVreg, maskIndexVreg, subIndexVreg;
    MicroAPI::UnalignReg ureg;
    MicroAPI::Duplicate(dstValueVreg, initValue);
    MicroAPI::Duplicate(maskIndexVreg, (IndexT)0);
    MicroAPI::Duplicate(subIndexVreg, (IndexT)1);
    if constexpr (std::is_same_v<IndexT, uint16_t>) {
        MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)tmpIndexVreg, 1);
    } else {
        MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)tmpIndexVreg, 1);
    }
    dstIndexVreg = tmpIndexVreg;
    int32_t postUpdateStride = srcRepStride * GetDataBlockSizeInBytes() / sizeof(T);
    // step1: from [count] to [oneRepSize] value index pair
    for (uint16_t i = 0; i < repeat; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, postUpdateStride);
        if constexpr (mode == ReduceMode::REDUCE_MAX) {
            MicroAPI::Max(tmpValueVreg, dstValueVreg, srcVreg, preg);
        } else {
            MicroAPI::Min(tmpValueVreg, dstValueVreg, srcVreg, preg);
        }
        // now tmpValueVreg is this round new value, dstValueVreg is previous round value
        // if previous round and this round value is change, update index
        MicroAPI::Compare<T, CMPMODE::NE>(pregCond, dstValueVreg, tmpValueVreg, preg);
        MicroAPI::Select(dstIndexVreg, tmpIndexVreg, dstIndexVreg, pregCond);
        // make next round index
        MicroAPI::Adds(tmpIndexVreg, tmpIndexVreg, (IndexT)postUpdateStride, preg);
        // update value
        dstValueVreg = tmpValueVreg;
    }
    // step2: from [oneRepSize] to [1] value index and store it to ub
    if constexpr (mode == ReduceMode::REDUCE_MAX) {
        MicroAPI::ReduceMax(tmpValueVreg, dstValueVreg, preg);
    } else {
        MicroAPI::ReduceMin(tmpValueVreg, dstValueVreg, preg);
    }
    MicroAPI::StoreUnAlign(dstLocal, tmpValueVreg, ureg, 1);  // store value
    // get dst value mask and squeeze dst index
    MicroAPI::Duplicate(tmpValueVreg, tmpValueVreg, preg);
    MicroAPI::Compare<T, CMPMODE::EQ>(pregCond, dstValueVreg, tmpValueVreg, preg);
    // gather mask index
    MicroAPI::GatherMask<IndexT, MicroAPI::GatherMaskMode::NO_STORE_REG>(tmpIndexVreg, dstIndexVreg, pregCond);
    // cal preg for how much index has the same max or min value
    MicroAPI::Compare<IndexT, CMPMODE::NE>(pregCond, tmpIndexVreg, maskIndexVreg, pregIndexFull);
    MicroAPI::ReduceMin(tmpIndexVreg, tmpIndexVreg, pregCond);
    MicroAPI::Sub(tmpIndexVreg, tmpIndexVreg, subIndexVreg, pregIndexFull);
    MicroAPI::StoreUnAlign((__ubuf__ IndexT *&)dstLocal, tmpIndexVreg, ureg, 1);
    MicroAPI::StoreUnAlignPost(dstLocal, ureg, 0);
}

template <typename T>
__aicore__ inline void ReduceMaxImpl(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count, bool calIndex)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float, uint64_t, int64_t>()),
        "ReduceMax current data type is not supported!");
    T initValue = GetMinValue<T>();
    if constexpr (sizeof(T) == 8) {
        if (calIndex) {
            VF_CALL<ReduceB64IndexTemplate<ReduceMode::REDUCE_MAX, T, uint32_t>>(
                dstLocal, srcLocal, workLocal, count, initValue);
        } else {
            VF_CALL<ReduceB64NoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                dstLocal, srcLocal, workLocal, count, initValue);
        }
    } else if constexpr (sizeof(T) == 4) {
        if (calIndex) {
            VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, T, uint32_t>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        } else {
            VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        }
    } else {
        if (calIndex) {
            VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, T, uint16_t>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        } else {
            VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceMaxImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    const uint64_t mask[], const int32_t repeat, const int32_t srcRepStride, bool calIndex)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "ReduceMax current data type is not supported!");
    T initValue = GetMinValue<T>();
    uint32_t count = static_cast<uint32_t>(mask[0]);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            }
        }
    } else {
        SetVectorMask<T>(mask[1], mask[0]);
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, true, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, true, T>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, true, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, true, T>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ReduceMaxImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    const int32_t mask, const int32_t repeat, const int32_t srcRepStride, bool calIndex)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "ReduceMax current data type is not supported!");
    T initValue = GetMinValue<T>();
    uint32_t maskReg = static_cast<uint32_t>(mask);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            }
        }
    } else {
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, false, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, false, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MAX, false, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MAX, false, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ReduceMinImpl(
    __ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count, bool calIndex)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float, uint64_t, int64_t>()),
        "ReduceMin current data type is not supported!");
    T initValue = GetMaxValue<T>();
    if constexpr (sizeof(T) == 8) {
        if (calIndex) {
            VF_CALL<ReduceB64IndexTemplate<ReduceMode::REDUCE_MIN, T, uint32_t>>(
                dstLocal, srcLocal, workLocal, count, initValue);
        } else {
            VF_CALL<ReduceB64NoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                dstLocal, srcLocal, workLocal, count, initValue);
        }
    } else if constexpr (sizeof(T) == 4) {
        if (calIndex) {
            VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, T, uint32_t>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        } else {
            VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        }
    } else {
        if (calIndex) {
            VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, T, uint16_t>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        } else {
            VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                dstLocal, srcLocal, workLocal, count, DEFAULT_BLK_NUM, initValue);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceMinImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    const uint64_t mask[], const int32_t repeat, const int32_t srcRepStride, bool calIndex)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "ReduceMin current data type is not supported!");
    T initValue = GetMaxValue<T>();
    uint32_t count = static_cast<uint32_t>(mask[0]);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                    dstLocal, srcLocal, workLocal, count, srcRepStride, initValue);
            }
        }
    } else {
        SetVectorMask<T>(mask[1], mask[0]);
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, true, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, true, T>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, true, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, true, T>>(
                    dstLocal, srcLocal, workLocal, 0, repeat, srcRepStride, initValue);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ReduceMinImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal,
    const int32_t mask, const int32_t repeat, const int32_t srcRepStride, bool calIndex)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "ReduceMin current data type is not supported!");
    T initValue = GetMaxValue<T>();
    uint32_t maskReg = static_cast<uint32_t>(mask);
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, srcRepStride, initValue);
            }
        }
    } else {
        if constexpr (sizeof(T) == 4) {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, false, T, uint32_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, false, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            }
        } else {
            if (calIndex) {
                VF_CALL<ReduceIndexTemplate<ReduceMode::REDUCE_MIN, false, T, uint16_t>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            } else {
                VF_CALL<ReduceNoIndexTemplate<ReduceMode::REDUCE_MIN, false, T>>(
                    dstLocal, srcLocal, workLocal, maskReg, repeat, srcRepStride, initValue);
            }
        }
    }
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue, uint32_t &maxMinIndex)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline T GetAccValImpl()
{
    ASCENDC_ASSERT((false), "GetAccVal is not supported on current device");
    return 0;
}
}  // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
