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
 * \file kernel_operator_vec_reduce_impl.h
 * \brief AscendC l300 support reduce api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H

#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_template_impl.h"

namespace AscendC {

#define FLOAT_MAX (__FLT_MAX__)
#define FLOAT_MIN (-__FLT_MAX__)
#define HALF_MAX (65504.0)
#define HALF_MIN (-65504.0)

template <bool isBitMask, typename T>
__aicore__ inline void GenPredicate(MicroAPI::MaskReg &preg, uint32_t maskReg)
{
    if constexpr (isBitMask) {
        preg = MicroAPI::MoveMask<T>();
    } else {
        preg = MicroAPI::UpdateMask<T>(maskReg);
    }
}

#define REDUCE_ADJUST_REPEAT_PARAM(src, repeat, dstRepStride, srcRepStride)                                     \
    uint16_t newRepeat = static_cast<uint16_t>(repeat);                                                         \
    uint32_t newDstRepStride = dstRepStride;                                                                    \
    __ubuf__ _Tp* newSrc = src;                                                                                 \
    if (dstRepStride == 0 && repeat > 0) {                                                                      \
        newRepeat = 1;                                                                                          \
        newDstRepStride = 1;                                                                                    \
        uint32_t srcStrideOffset = srcRepStride * ONE_BLK_SIZE / sizeof(_Tp) * (repeat - 1);                    \
        newSrc += srcStrideOffset;                                                                              \
    }

#define CONTINUOUS_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                 \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<_Tp> vreg0;                                                                                   \
        RegTensor<_Tp> vreg1;                                                                                   \
        UnalignReg ureg;                                                                                        \
        uint32_t sreg = mask;                                                                                   \
        MaskReg preg = CreatePredicate<_Tp>(sreg);                                                              \
        for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
            DataCopy<_Tp, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
            REDUCE_FUNC<_Tp>(vreg1, vreg0, preg);                                                               \
            DataCopyUnAlign<_Tp>(dst, vreg1, ureg, dstStrideOffset);                                            \
            DataCopyUnAlignPost<_Tp>(dst, ureg, dstStrideOffset * (newDstRepStride - 1));                       \
        }                                                                                                       \
    }

#define REDUCE_CONTINUOUS_MODE_IMPL(REDUCE_FUNC, DATA_TYPE, dstStrideOffset)                                            \
    template <class _Tp, bool isSetMask>                                                                                \
    __aicore__ inline void REDUCE_FUNC##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t repeat,    \
        const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)         \
    {                                                                                                                   \
        REDUCE_ADJUST_REPEAT_PARAM(src, repeat, dstRepStride, srcRepStride)                                             \
        CONTINUOUS_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                         \
    }


#define BLOCK_REDUCE_IMPL_NOT_SUPPORT(REDUCE_FUNC, DATA_TYPE)                                                           \
    template <typename _Tp, bool isSetMask>                                                                             \
    __aicore__ inline void REDUCE_FUNC##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t repeat,    \
        const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)         \
    {                                                                                                                   \
        static_assert(!std::is_same_v<uint32_t, _Tp>, "current data type is not supported!");                           \
    }

#define BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)   \
    REDUCE_CONTINUOUS_MODE_IMPL(BlockReduce##REDUCE_TYPE, DATA_TYPE, VECTOR_REG_WIDTH / ONE_BLOCK_SIZE)


BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Min, half)
BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Min, float)

#define BITBYBIT_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                   \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<_Tp> vreg0;                                                                                   \
        RegTensor<_Tp> vreg1;                                                                                   \
        UnalignReg ureg;                                                                                        \
        MaskReg preg = MovePredicate<_Tp>();                                                                    \
        for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
            DataCopy<_Tp, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
            REDUCE_FUNC<_Tp>(vreg1, vreg0, preg);                                                               \
            DataCopyUnAlign<_Tp>(dst, vreg1, ureg, dstStrideOffset);                                            \
            DataCopyUnAlignPost<_Tp>(dst, ureg, dstStrideOffset * (newDstRepStride - 1));                       \
        }                                                                                                       \
    }

#define REDUCE_BITBYBIT_MODE_IMPL(REDUCE_FUNC, DATA_TYPE, dstStrideOffset)                                              \
    template <typename _Tp, bool isSetMask>                                                                             \
    __aicore__ inline void REDUCE_FUNC##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t repeat,    \
        const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)     \
    {                                                                                                                   \
        if constexpr (isSetMask) {                                                                                      \
            SetVectorMask<_Tp>(mask[1], mask[0]);                                                                       \
        }                                                                                                               \
                                                                                                                        \
        REDUCE_ADJUST_REPEAT_PARAM(src, repeat, dstRepStride, srcRepStride)                                             \
        BITBYBIT_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                           \
    }

#define BLOCK_REDUCE_BITBYBIT_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)   \
    REDUCE_BITBYBIT_MODE_IMPL(BlockReduce##REDUCE_TYPE, DATA_TYPE, VECTOR_REG_WIDTH / ONE_BLOCK_SIZE)

BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Min, half)
BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Min, float)

/* **************************************** Whole Reduce Interface ****************************************** */
#define WHOLE_REDUCE_ADJUST_REPEAT_PARAM(src, repeat, dstRepStride, srcRepStride)                               \
    uint16_t newRepeat = static_cast<uint16_t>(repeat);                                                         \
    uint32_t newDstRepStride = dstRepStride;                                                                    \
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;                                                      \
    __ubuf__ _Tp* newSrc = src;                                                                                 \
    if (dstRepStride == 0 && repeat > 0) {                                                                      \
        newRepeat = 1;                                                                                          \
        newDstRepStride = 1;                                                                                    \
        uint32_t srcStrideOffset = srcRepStride * ONE_BLK_SIZE / sizeof(_Tp) * (repeat - 1);                    \
        newSrc += srcStrideOffset;                                                                              \
        dstStrideOffset = 1;                                                                                    \
    }

#define WHOLE_REDUCE_CONTINUOUS_MODE_REDUCE_VF(REDUCE_TYPE)                                                     \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<_Tp> vreg0;                                                                                   \
        RegTensor<_Tp> vreg1;                                                                                   \
        UnalignReg ureg;                                                                                        \
        uint32_t sreg = mask;                                                                                   \
        MaskReg preg = CreatePredicate<_Tp>(sreg);                                                              \
        for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
            DataCopy<_Tp, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
            Reduce##REDUCE_TYPE<_Tp>(vreg1, vreg0, preg);                                                       \
            DataCopyUnAlign<_Tp>(dst, vreg1, ureg, dstStrideOffset);                                            \
            DataCopyUnAlignPost<_Tp>(dst, ureg, newDstRepStride - dstStrideOffset);                             \
        }                                                                                                       \
    }

#define WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)                                               \
    template <class _Tp, bool isSetMask>                                                                        \
    __aicore__ inline void WholeReduce##REDUCE_TYPE##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src,     \
        const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,  \
        const int32_t srcRepStride, ReduceOrder order)                                                          \
    {                                                                                                           \
        WHOLE_REDUCE_ADJUST_REPEAT_PARAM(src, repeatTimes, dstRepStride, srcRepStride)                          \
        WHOLE_REDUCE_CONTINUOUS_MODE_REDUCE_VF(REDUCE_TYPE)                                                     \
    }

WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, int8_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, int16_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, int32_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, uint8_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, uint16_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, uint32_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, half)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, float)

template <bool isSetMask, bool isBitMask, bool isCounterMode, typename T>
__aicore__ inline void ReduceCommonCall(MicroAPI::MaskReg& mask, uint16_t& newRepeatTimes, uint32_t& countSreg,
                                        uint32_t maskReg, __ubuf__ uint64_t* maskBuf)
{
    if constexpr (isCounterMode) {
        if constexpr (!isSetMask) {
            // get SPR.MASK in VF
            MicroAPI::MaskReg sprLoadMaskReg = MicroAPI::MoveMask<uint16_t>();
            MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(maskBuf, sprLoadMaskReg);
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
template <bool isSetMask, bool isBitMask, bool isCounterMode, bool withStride, auto func, typename T, typename U = T>
__aicore__ void ReduceUnalignCall(__ubuf__ U *dst, __ubuf__ T *src, int32_t repeat, uint32_t oneRepOffset,
    uint32_t dstRepOffsetPost, uint32_t srcBlkStride, uint32_t srcRepStride, uint32_t maskReg, __ubuf__ uint64_t *maskBuf)
{
    MicroAPI::MaskReg mask;
    uint16_t newRepeatTimes = static_cast<uint16_t>(repeat);
    uint32_t countSreg = static_cast<uint32_t>(maskReg);
    if constexpr (!isCounterMode || !withStride) {
        ReduceCommonCall<isSetMask, isBitMask, isCounterMode, T>(mask, newRepeatTimes, countSreg, maskReg, maskBuf);
    } else {
        if (dstRepOffsetPost != 0) {
            ReduceCommonCall<isSetMask, isBitMask, isCounterMode, T>(mask, newRepeatTimes, countSreg, maskReg, maskBuf);
        }
    }
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<U> dstVreg;
    MicroAPI::UnalignReg ureg;
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            mask = MicroAPI::UpdateMask<T>(countSreg);
        }
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            srcVreg, src, srcBlkStride, srcRepStride, mask);
        func(dstVreg, srcVreg, mask);
        MicroAPI::DataCopyUnAlign(dst, dstVreg, ureg, oneRepOffset);
        if constexpr (withStride) {
            MicroAPI::DataCopyUnAlignPost(dst, ureg, dstRepOffsetPost);
        }
    }
    if constexpr (!withStride) {
        MicroAPI::DataCopyUnAlignPost(dst, ureg, dstRepOffsetPost);
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
            constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
            uint32_t newRepeatTimes = CeilDivision(maskReg, oneRepSize);
            __ubuf__ T *newSrc = src + srcStrideOffset * (newRepeatTimes - 1);
            maskReg = maskReg - oneRepSize * (newRepeatTimes - 1);
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
/* **************************************** Whole Reduce Impl ****************************************** */
template <bool isSetMask, bool isBitMask, bool isCounterMode, bool withStride, auto func, typename T, typename U = T>
__aicore__ void WholeReduceUnalignCall(__ubuf__ U *dst, __ubuf__ T *src, int32_t repeat, uint32_t oneRepOffset,
    uint32_t dstRepOffsetPost, uint32_t srcBlkStride, uint32_t srcRepStride, uint32_t maskReg, __ubuf__ uint64_t *maskBuf, ReduceOrder order)
{
    MicroAPI::MaskReg mask;
    uint16_t newRepeatTimes = static_cast<uint16_t>(repeat);
    uint32_t countSreg = static_cast<uint32_t>(maskReg);
    if constexpr (!isCounterMode || !withStride) {
        ReduceCommonCall<isSetMask, isBitMask, isCounterMode, T>(mask, newRepeatTimes, countSreg, maskReg, maskBuf);
    } else {
        if (dstRepOffsetPost != 0) {
            ReduceCommonCall<isSetMask, isBitMask, isCounterMode, T>(mask, newRepeatTimes, countSreg, maskReg, maskBuf);
        }
    }
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<U> dstVreg;
    MicroAPI::RegTensor<U> dstValueVreg0;
    MicroAPI::RegTensor<U> dstIndexVreg1;
    MicroAPI::RegTensor<U> dstAbandonVreg;
    MicroAPI::UnalignReg ureg;
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (isCounterMode) {
            mask = MicroAPI::UpdateMask<T>(countSreg);
        }
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            srcVreg, src, srcBlkStride, srcRepStride, mask);
        func(dstVreg, srcVreg, mask);
        if(order == ReduceOrder::ORDER_INDEX_VALUE || order == ReduceOrder::ORDER_ONLY_INDEX) {
            MicroAPI::DeInterleave(dstValueVreg0, dstIndexVreg1, dstVreg, dstVreg);
            MicroAPI::Interleave(dstVreg, dstAbandonVreg, dstIndexVreg1, dstValueVreg0);
        }
        MicroAPI::DataCopyUnAlign(dst, dstVreg, ureg, oneRepOffset);
        if constexpr (withStride) {
            MicroAPI::DataCopyUnAlignPost(dst, ureg, dstRepOffsetPost);
        }
    }
    if constexpr (!withStride) {
        MicroAPI::DataCopyUnAlignPost(dst, ureg, dstRepOffsetPost);
    }
}

template <bool isSetMask, bool isBitMask, auto func, typename T, typename U = T>
__aicore__ inline void WholeReduceTemplate(__ubuf__ U *dst, __ubuf__ T *src, int32_t repeat, int32_t dstRepStride,
    uint32_t oneRepOffset, int32_t srcBlkStride, int32_t srcRepStride, uint32_t maskReg, ReduceOrder order)
{
    constexpr uint32_t ONE_BLK_ELEMENT_NUM = GetDataBlockSizeInBytes() / sizeof(T);
    bool isCounterMode = Internal::IsCounterMode();
    __ubuf__ uint64_t *maskBuf = nullptr;
    if (isCounterMode) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        }
        if (dstRepStride == 0) {
            uint32_t srcStrideOffset = srcRepStride * ONE_BLK_ELEMENT_NUM;
            constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
            uint32_t newRepeatTimes = CeilDivision(maskReg, oneRepSize);
            __ubuf__ T *newSrc = src + srcStrideOffset * (newRepeatTimes - 1);
            maskReg = maskReg - oneRepSize * (newRepeatTimes - 1);
            VF_CALL<WholeReduceUnalignCall<isSetMask, isBitMask, true, true, func, T, U>>(
                dst, newSrc, 1, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf, order);
        } else if (dstRepStride == 1) {
            VF_CALL<WholeReduceUnalignCall<isSetMask, isBitMask, true, false, func, T, U>>(
                dst, src, repeat, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf, order);
        } else {
            uint32_t dstRepOffsetPost = oneRepOffset * (dstRepStride - 1);
            VF_CALL<WholeReduceUnalignCall<isSetMask, isBitMask, true, true, func, T, U>>(
                dst, src, repeat, oneRepOffset, dstRepOffsetPost, srcBlkStride, srcRepStride, maskReg, maskBuf, order);
        }
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    } else {
        if (dstRepStride == 0 && repeat > 0) {
            uint32_t srcStrideOffset = srcRepStride * ONE_BLK_ELEMENT_NUM;
            __ubuf__ T *newSrc = src + srcStrideOffset * (repeat - 1);
            VF_CALL<WholeReduceUnalignCall<isSetMask, isBitMask, false, true, func, T, U>>(
                dst, newSrc, 1, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf, order);
        } else if (dstRepStride == 1 && repeat > 0) {
            VF_CALL<WholeReduceUnalignCall<isSetMask, isBitMask, false, false, func, T, U>>(
                dst, src, repeat, oneRepOffset, 0, srcBlkStride, srcRepStride, maskReg, maskBuf, order);
        } else {
            uint32_t dstRepOffsetPost = oneRepOffset * (dstRepStride - 1);
            VF_CALL<WholeReduceUnalignCall<isSetMask, isBitMask, false, true, func, T, U>>(
                dst, src, repeat, oneRepOffset, dstRepOffsetPost, srcBlkStride, srcRepStride, maskReg, maskBuf, order);
        }
    }
}
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
    uint32_t oneRepOffset = (order == ReduceOrder::ORDER_VALUE_INDEX || order == ReduceOrder::ORDER_INDEX_VALUE) ? 2 : 1;
    WholeReduceTemplate<isSetMask, true, MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dstLocal, srcLocal, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, mask[0], order);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t, half, float>()),
        "WholeReduceMax current data type is not supported!");
    uint32_t maskReg = static_cast<uint32_t>(mask);
    uint32_t oneRepOffset = (order == ReduceOrder::ORDER_VALUE_INDEX || order == ReduceOrder::ORDER_INDEX_VALUE) ? 2 : 1;
    WholeReduceTemplate<isSetMask, false, MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
        dstLocal, srcLocal, repeat, dstRepStride, oneRepOffset, srcBlkStride, srcRepStride, maskReg, order);
}
// WholeReduceSum mask连续模式
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


#define WHOLE_REDUCE_BITBYBIT_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)                                                     \
    template <typename _Tp, bool isSetMask>                                                                         \
    __aicore__ inline void WholeReduce##REDUCE_TYPE##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src,         \
        const uint64_t mask[2], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,  \
        const int32_t srcRepStride, ReduceOrder order)                                                              \
    {                                                                                                               \
        if constexpr (isSetMask) {                                                                                  \
            SetVectorMask<_Tp>(mask[1], mask[0]);                                                                   \
        }                                                                                                           \
        WHOLE_REDUCE_ADJUST_REPEAT_PARAM(src, repeatTimes, dstRepStride, srcRepStride)                              \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<_Tp> vreg0;                                                                                   \
            RegTensor<_Tp> vreg1;                                                                                   \
            UnalignReg ureg;                                                                                        \
            MaskReg preg = MovePredicate<_Tp>();                                                                    \
            for (uint16_t i = 0; i < newRepeat; ++i) {                                                              \
                DataCopy<_Tp, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
                Reduce##REDUCE_TYPE<_Tp>(vreg1, vreg0, preg);                                                       \
                DataCopyUnAlign<_Tp>(dst, vreg1, ureg, dstStrideOffset);                                            \
                DataCopyUnAlignPost<_Tp>(dst, ureg, newDstRepStride - dstStrideOffset);                             \
            }                                                                                                       \
        }                                                                                                           \
    }

WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, int16_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, int32_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, uint16_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, uint32_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, half)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, float)

/* **************************************** Pair Reduce Interface ****************************************** */

REDUCE_CONTINUOUS_MODE_IMPL(PairReduceSum, half, VECTOR_REG_WIDTH / sizeof(half) / HALF_FACTOR)
REDUCE_CONTINUOUS_MODE_IMPL(PairReduceSum, float, VECTOR_REG_WIDTH / sizeof(float) / HALF_FACTOR)
REDUCE_BITBYBIT_MODE_IMPL(PairReduceSum, half, VECTOR_REG_WIDTH / sizeof(half) / HALF_FACTOR)
REDUCE_BITBYBIT_MODE_IMPL(PairReduceSum, float, VECTOR_REG_WIDTH / sizeof(float) / HALF_FACTOR)


/* ****************************************** Reduce Interface ******************************************** */

// Level 2
// 将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量
#define REDUCE_TO_VEC(opName, vregType, type, dataCount, offsetNum, initVal, typeName)       \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        uint16_t loop_num = (dataCount + ELE_CNT_B##type - 1) / ELE_CNT_B##type;             \
        for (uint16_t i = 0; i < 1; i++) {                                                   \
            MaskReg pgAll = CreatePredicate<typeName>();                                     \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                    \
            RegTensor<typeName> vreg1;                                                       \
            Duplicate<typeName, typeName>(vreg1, initVal);                                   \
            uint32_t sreg = (uint32_t)dataCount;                                             \
            for (uint16_t j = 0; j < loop_num; ++j) {                                        \
                MaskReg preg =  CreatePredicate<typeName>(sreg);                              \
                AddrReg offset1 = CreateAddrReg<typeName>(0, static_cast<uint16_t>(offsetNum)); \
                RegTensor<typeName> vreg0;                                                    \
                DataCopy<typeName>(vreg0, src, offset1);                                     \
                opName<typeName, Mode::MERGING>(vreg1, vreg0, vreg1, preg);               \
            }                                                                                \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, vreg1, offset0, pgAll);     \
        }                                                                                    \
    }

#define VEC_REDUCE_SUM_IMPL(opName, dst, src, type, dataCount, typeName)           \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        uint16_t loop_num = (uint16_t)(dataCount + ELE_CNT_B##type - 1) / ELE_CNT_B##type;   \
        uint32_t sreg = (uint32_t)dataCount;                                                 \
        for (uint16_t i = 0; i < loop_num; i++) {                                            \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                    \
            MaskReg preg =  CreatePredicate<typeName>(sreg);                                 \
            MaskReg preg1 = CreatePredicate<typeName, Pat::VL1>();                          \
            RegTensor<typeName> dstVreg;                                                     \
            RegTensor<typeName> srcVreg;                                                     \
            DataCopy<typeName>(srcVreg, src, offset0);                                       \
            opName<typeName, typeName, Mode::MERGING>(dstVreg, srcVreg, preg);             \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(dst, dstVreg, offset0, preg1);    \
        }                                                                                    \
    }

#define VEC_REDUCE_IMPL(opName, dst, src, type, dataCount, typeName)                        \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        uint16_t loop_num = (uint16_t)(dataCount + ELE_CNT_B##type - 1) / ELE_CNT_B##type;   \
        uint32_t sreg = (uint32_t)dataCount;                                                 \
        for (uint16_t i = 0; i < loop_num; i++) {                                            \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                    \
            MaskReg preg =  CreatePredicate<typeName>(sreg);                                 \
            MaskReg preg1 = CreatePredicate<typeName,  Pat::VL1>();                          \
            RegTensor<typeName> dstVreg;                                                     \
            RegTensor<typeName> srcVreg;                                                     \
            DataCopy<typeName>(srcVreg, src, offset0);                                       \
            opName<typeName, Mode::MERGING>(dstVreg, srcVreg, preg);                         \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(dst, dstVreg, offset0, preg1);    \
        }                                                                                    \
    }

#define REDUCE_NO_INDEX_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal)               \
    template <>                                                                                          \
    __aicore__ inline void reduceFuncName##Impl<typeName>(                                               \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        REDUCE_TO_VEC(instrName, vregType, type, count, elementNumPerInstr, initVal, typeName);          \
        VEC_REDUCE_IMPL(Reduce##instrName, dst, work, type, elementNumPerInstr, typeName);               \
    }

template <typename T>
__aicore__ inline void ReduceMaxNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, int16_t, s16, 16, INT16_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, uint8_t, u8, 8, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, int8_t, s8, 8, INT8_MIN)

template <typename T>
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, half, f16, 16, HALF_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, float, f32, 32, FLOAT_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, uint32_t, u32, 32, UINT32_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, int32_t, s32, 32, INT32_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, uint16_t, u16, 16, UINT16_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, int16_t, s16, 16, INT16_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, uint8_t, u8, 8, UINT8_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, int8_t, s8, 8, INT8_MAX)

// vcmp_lt/vcmp_gt指令用于寻找在当前加载的数据中找到比之前数据更小/更大的数的下标
// vsel指令用于更新已加载的最大/最小值数据下标
#define REDUCE_TO_VEC_WITH_INDEX(opName, cmpOp, type, count, offsetNum, initVal, typeName, indexType, \
    initIndexType)                                                                                              \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        uint16_t loop_num = (uint16_t)(count + ELE_CNT_B##type - 1) / ELE_CNT_B##type;                          \
        for (uint16_t i = 0; i < 1; i++) {                                                                      \
            MaskReg pgAll = CreatePredicate<typeName>();                                                        \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                                       \
            RegTensor<indexType> dstIndexVreg;                                                                  \
            RegTensor<indexType> indexVreg;                                                                     \
            RegTensor<initIndexType> initIndexVreg;                                                             \
            RegTensor<typeName> dstVreg;                                                                        \
            Duplicate<indexType, indexType>(dstIndexVreg, 0);                                                   \
            CreateVecIndex<initIndexType, Order::INC_ORDER_VALUE, indexType>(initIndexVreg, 0);                \
            Duplicate<typeName, typeName>(dstVreg, initVal);                                                   \
            indexVreg = (RegTensor<indexType> &)initIndexVreg;                                                  \
            uint32_t sreg = (uint32_t)count;                                                                    \
            for (uint16_t j = 0; j < loop_num; j++) {                                                           \
                AddrReg offset1 = CreateAddrReg<typeName>(0, static_cast<uint16_t>(offsetNum));                 \
                MaskReg preg = CreatePredicate<typeName>(sreg);                                                 \
                MaskReg pd;                                                                                     \
                MaskReg indexPg;                                                                                \
                RegTensor<typeName> srcVreg;                                                                    \
                DataCopy<typeName>(srcVreg, src, offset1);                                                      \
                Compare<typeName, cmpOp>(indexPg, srcVreg, dstVreg, preg);                                      \
                Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                              \
                opName<typeName, Mode::MERGING>(dstVreg, srcVreg, dstVreg, preg);                               \
                Adds<indexType, uint32_t, Mode::MERGING>(indexVreg, indexVreg, offsetNum, preg);                \
            }                                                                                                   \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, dstVreg, offset0, pgAll);                      \
            DataCopy<indexType, DistVST::DIST_NORM_B##type>(tempBuf, dstIndexVreg, offset0, pgAll);             \
        }                                                                                                       \
    }

#define REDUCE_MAX_MIN_WITH_INDEX_IMPL(                                                   \
    opName, type, count, initVal, slideNum, typeName, indexType, initIndexType)           \
    __VEC_SCOPE__                                                                         \
    {                                                                                     \
        uint16_t loop_num = (uint16_t)(count + ELE_CNT_B##type - 1) / ELE_CNT_B##type;    \
        uint32_t sreg = (uint32_t)count;                                                  \
        for (uint16_t i = 0; i < loop_num; i++) {                                         \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                 \
            MaskReg preg = CreatePredicate<typeName>(sreg);                               \
            MaskReg preg1 = CreatePredicate<typeName, Pat::VL2>();                        \
            MaskReg pd;                                                                   \
            RegTensor<indexType> indexVreg, zeroVreg;                                     \
            RegTensor<indexType> dstIndexVreg;                                            \
            RegTensor<typeName> srcVreg;                                                  \
            RegTensor<typeName> dstVreg;                                                  \
            Duplicate<indexType, initIndexType>(dstIndexVreg, initVal);                   \
            Duplicate<indexType, indexType>(zeroVreg, 0);                                 \
            DataCopy<typeName>(srcVreg, work, offset0);                                   \
            DataCopy<indexType>(indexVreg, tempBuf, offset0);                             \
            opName<typeName, Mode::MERGING>(dstVreg, pd, srcVreg, preg);                  \
            Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, pd);                 \
            ReduceMin<indexType>(dstIndexVreg, dstIndexVreg, preg);                       \
            RegTensor<typeName> temZeroVreg = (RegTensor<typeName> &)zeroVreg;            \
            Slide<typeName>(dstVreg, temZeroVreg, dstVreg, (uint16_t)1);                  \
            RegTensor<typeName> temDestIndexVreg = (RegTensor<typeName> &)dstIndexVreg;   \
            Slide<typeName>(dstVreg, dstVreg, temDestIndexVreg, (uint16_t)slideNum);      \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(dst, dstVreg, offset0, preg1); \
        }                                                                                 \
    }

#define REDUCE_MAX_WITH_INDEX_IMPL(typeName, indexType, initIndexType, type, initVal, indexInitVal)           \
    template <>                                                                                          \
    __aicore__ inline void ReduceMaxWithIndexImpl<typeName>(                                             \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                            \
        __ubuf__ indexType *tempBuf =                                                                    \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);          \
        REDUCE_TO_VEC_WITH_INDEX(Max, CMPMODE::GT, type, count, elementNumPerInstr, initVal, typeName, indexType, initIndexType); \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMax, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType)  \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                           \
    }

#define REDUCE_MIN_WITH_INDEX_IMPL(typeName, indexType, initIndexType, type, initVal, indexInitVal)           \
    template <>                                                                                          \
    __aicore__ inline void ReduceMinWithIndexImpl<typeName>(                                             \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                            \
        __ubuf__ indexType *tempBuf =                                                                    \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);          \
        REDUCE_TO_VEC_WITH_INDEX(Min, CMPMODE::LT, type, count, elementNumPerInstr, initVal, typeName, indexType, initIndexType);\
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMin, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType)  \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                           \
    }

template <typename T>
__aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MAX_WITH_INDEX_IMPL(half, uint16_t, int16_t, 16, HALF_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(float, uint32_t, int32_t, 32, FLOAT_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint32_t, uint32_t,int32_t,   32, 0, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int32_t, uint32_t, int32_t, 32, INT32_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint16_t, uint16_t, int16_t, 16, 0, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int16_t, uint16_t, int16_t,  16, INT16_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint8_t, uint8_t, int8_t,  8, 0, UINT8_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int8_t, uint8_t, int8_t, 8, INT8_MIN, UINT8_MAX)

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL(half, uint16_t, int16_t, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(float, uint32_t, int32_t, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint32_t, uint32_t, int32_t, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int32_t, uint32_t, int32_t, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint16_t, uint16_t, int16_t, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int16_t, uint16_t, int16_t, 16, INT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint8_t, uint8_t, int8_t, 8, UINT8_MAX, UINT8_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int8_t, uint8_t, int8_t, 8, INT8_MAX, UINT8_MAX)

// level 0 mask 连续模式
// mask 连续模式将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量后累加
#define REDUCE_TO_VEC_CONTINUOUS_MODE(opName, typeName, type, initVal)                                         \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        int32_t newRepeat = repeatTimes;                                                                        \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                                              \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                                                  \
        __ubuf__ typeName *newSrc = src;                                                                        \
        for (uint16_t i = 0; i < 1; i++) {                                                                      \
            MaskReg pgAll = CreatePredicate<typeName>();                                                        \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                                       \
            RegTensor<typeName> vreg1;                                                                          \
            Duplicate<typeName, typeName>(vreg1, initVal);                                                      \
            uint32_t sreg = (uint32_t)mask;                                                                     \
            MaskReg preg = CreatePredicate<typeName>(sreg);                                                     \
            for (uint16_t j = 0; j < (uint16_t)newRepeat; ++j) {                                                \
                RegTensor<typeName> vreg0;                                                                      \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(vreg0, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, \
                    srcRepStride, preg);                                                                        \
                opName<typeName, Mode::MERGING>(vreg1, vreg0, vreg1, preg);                                     \
            }                                                                                                   \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, vreg1, offset0, pgAll);                        \
        }                                                                                                       \
    }

// L0 连续模式 no index
#define REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal) \
    template <>                                                                                             \
    __aicore__ inline void reduceFuncName##Impl(__ubuf__ typeName *dst,                                    \
        __ubuf__ typeName *src,                                                                            \
        __ubuf__ typeName *work,                                                                           \
        const int32_t mask,                                                                                \
        const int32_t repeatTimes,                                                                         \
        const int32_t srcRepStride)                                                                        \
    {                                                                                                      \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                              \
        REDUCE_TO_VEC_CONTINUOUS_MODE(instrName, typeName, type, initVal)                                \
        VEC_REDUCE_IMPL(Reduce##instrName, dst, work, type, elementNumPerInstr, typeName)            \
    }

// level 0 ReduceSum
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
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
        MicroAPI::ReduceSum(dstVreg, srcVreg, preg);
        MicroAPI::DataCopyUnAlign(dstLocal, dstVreg, ureg, 1);
    }
    MicroAPI::DataCopyUnAlignPost(dstLocal, ureg, 0);
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
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
        MicroAPI::ReduceSum(dstVreg, srcVreg, preg);
        MicroAPI::DataCopyUnAlign(dstLocal, dstVreg, ureg, 1);
    }
    MicroAPI::DataCopyUnAlignPost(dstLocal, ureg, 0);
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
// level 0 ReduceSum mask连续模式
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
// level 0 ReduceSum mask 逐bit模式
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

// lv2
template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ T *workLocal, uint32_t count)
{
    static_assert((SupportType<T, half, float>()), "ReduceSum current data type is not supported!");
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    if (count <= oneRepSize) {
        VF_CALL<ReduceSumCounterMode<T, 1>>(dstLocal, srcLocal, workLocal, count, 8);
    } else if (count <= oneRepSize * oneRepSize) {
        VF_CALL<ReduceSumCounterMode<T, 2>>(dstLocal, srcLocal, workLocal, count, 8);
    } else {
        VF_CALL<ReduceSumCounterMode<T, 3>>(dstLocal, srcLocal, workLocal, count, 8);
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
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
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
    MicroAPI::DataCopyUnAlign(dstLocal, dstVreg, ureg, 1);
    MicroAPI::DataCopyUnAlignPost(dstLocal, ureg, 0);
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
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, postUpdateStride);
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
    MicroAPI::DataCopyUnAlign(dstLocal, dstVreg, ureg, 1);
    MicroAPI::DataCopyUnAlignPost(dstLocal, ureg, 0);
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
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, srcRepOffset);
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
    MicroAPI::DataCopyUnAlign(dstLocal, tmpValueVreg, ureg, 1);  // store value
    // get dst value mask and squeeze dst index
    MicroAPI::Duplicate(tmpValueVreg, tmpValueVreg, pregFull);
    MicroAPI::Compare<T, CMPMODE::EQ>(pregCond, dstValueVreg, tmpValueVreg, pregFull);
    MicroAPI::GatherMask<IndexT, MicroAPI::GatherMaskMode::NO_STORE_REG>(tmpIndexVreg, dstIndexVreg, pregCond);
    // cal preg for how much index has the same max or min value
    MicroAPI::Compare<IndexT, CMPMODE::NE>(pregCond, tmpIndexVreg, maskIndexVreg, pregIndexFull);
    MicroAPI::ReduceMin(tmpIndexVreg, tmpIndexVreg, pregCond);
    MicroAPI::Sub(tmpIndexVreg, tmpIndexVreg, subIndexVreg, pregIndexFull);
    MicroAPI::DataCopyUnAlign((__ubuf__ IndexT *&)dstLocal, tmpIndexVreg, ureg, 1);
    MicroAPI::DataCopyUnAlignPost(dstLocal, ureg, 0);
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
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcVreg, srcLocal, postUpdateStride);
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
    MicroAPI::DataCopyUnAlign(dstLocal, tmpValueVreg, ureg, 1);  // store value
    // get dst value mask and squeeze dst index
    MicroAPI::Duplicate(tmpValueVreg, tmpValueVreg, preg);
    MicroAPI::Compare<T, CMPMODE::EQ>(pregCond, dstValueVreg, tmpValueVreg, preg);
    // gather mask index
    MicroAPI::GatherMask<IndexT, MicroAPI::GatherMaskMode::NO_STORE_REG>(tmpIndexVreg, dstIndexVreg, pregCond);
    // cal preg for how much index has the same max or min value
    MicroAPI::Compare<IndexT, CMPMODE::NE>(pregCond, tmpIndexVreg, maskIndexVreg, pregIndexFull);
    MicroAPI::ReduceMin(tmpIndexVreg, tmpIndexVreg, pregCond);
    MicroAPI::Sub(tmpIndexVreg, tmpIndexVreg, subIndexVreg, pregIndexFull);
    MicroAPI::DataCopyUnAlign((__ubuf__ IndexT *&)dstLocal, tmpIndexVreg, ureg, 1);
    MicroAPI::DataCopyUnAlignPost(dstLocal, ureg, 0);
}

// level 0 ReduceMax mask连续模式
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

// level 0 ReduceMax mask逐bit模式
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
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, half, f16, 16, HALF_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, float, f32, 32, FLOAT_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, uint32_t, u32, 32, UINT32_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, int32_t, s32, 32, INT32_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, uint16_t, u16, 16, UINT16_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, int16_t, s16, 16, INT16_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, uint8_t, u8, 8, UINT8_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, int8_t, s8, 8, INT8_MAX)

// level 0 mark 连续 with index
#define REDUCE_TO_VEC_WITH_INDEX_CONTINUOUS_MODE(                                                    \
    opName, cmpOp, type, offsetNum, initVal, typeName, indexType, initIndexType)                     \
    __VEC_SCOPE__                                                                                    \
    {                                                                                                \
        int32_t newRepeat = repeatTimes;                                                             \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                                   \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                                       \
        __ubuf__ typeName *newSrc = src;                                                             \
        for (uint16_t i = 0; i < 1; i++) {                                                           \
            MaskReg pgAll = CreatePredicate<typeName>();                                             \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                            \
            RegTensor<indexType> dstIndexVreg;                                                       \
            RegTensor<indexType> indexVreg;                                                          \
            RegTensor<initIndexType> initIndexVreg;                                                  \
            RegTensor<typeName> dstVreg;                                                             \
            Duplicate<indexType, indexType>(dstIndexVreg, 0);                                    \
            CreateVecIndex<initIndexType, Order::INC_ORDER_VALUE, indexType>(initIndexVreg, 0);   \
            Duplicate<typeName, typeName>(dstVreg, initVal);                                     \
            indexVreg = (RegTensor<indexType> &)initIndexVreg;                                       \
            uint32_t sreg = (uint32_t)mask;                                                          \
            MaskReg preg = CreatePredicate<typeName>(sreg);                                          \
            uint32_t strideConfig = ((uint32_t)DEFAULT_BLK_STRIDE) << 16 | (srcRepStride & 0xFFFFU); \
            for (uint16_t j = 0; j < (uint16_t)newRepeat; j++) {                                     \
                MaskReg pd;                                                                          \
                MaskReg indexPg;                                                                     \
                RegTensor<typeName> srcVreg;                                                         \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(                                  \
                    srcVreg, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, srcRepStride, preg);              \
                Compare<typeName, cmpOp>(indexPg, srcVreg, dstVreg, preg);                \
                Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                   \
                opName<typeName, Mode::MERGING>(dstVreg, srcVreg, dstVreg, preg);                  \
                Adds<indexType, uint32_t, Mode::MERGING>(indexVreg, indexVreg, offsetNum, preg);     \
            }                                                                                        \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, dstVreg, offset0, pgAll);           \
            DataCopy<indexType, DistVST::DIST_NORM_B##type>(tempBuf, dstIndexVreg, offset0, pgAll);  \
        }                                                                                            \
    }


#define REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(typeName, indexType, initIndexType, type, initVal, indexInitVal)   \
    template <typename T = typeName>                                                                             \
    __aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ typeName *dst,                                        \
        __ubuf__ typeName *src,                                                                                  \
        __ubuf__ typeName *work,                                                                                 \
        const int32_t mask,                                                                                      \
        const int32_t repeatTimes,                                                                               \
        const int32_t srcRepStride)                                                                              \
    {                                                                                                            \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                    \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                    \
        __ubuf__ indexType *tempBuf =                                                                            \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);                  \
        REDUCE_TO_VEC_WITH_INDEX_CONTINUOUS_MODE(Min, CMPMODE::LT, type, elementNumPerInstr, initVal, typeName, indexType, initIndexType) \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMin, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType) \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                   \
    }

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(half, uint16_t, int16_t, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(float, uint32_t, int32_t, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint32_t, uint32_t, int32_t, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(int32_t, uint32_t, int32_t, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint16_t, uint16_t, int16_t, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(int16_t, uint16_t, int16_t, 16, INT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint8_t, uint8_t, int8_t, 8, UINT8_MAX, UINT8_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(int8_t, uint8_t, int8_t, 8, INT8_MAX, UINT8_MAX)

// level 0  bit by bit
// mask bit by bit模式将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量后累加
#define REDUCE_TO_VEC_BIT_BY_BIT_MODE(opName, type, initVal, typeName)                   \
    SetVectorMask<typeName>(mask[1], mask[0]);                                           \
    __VEC_SCOPE__                                                                        \
    {                                                                                    \
        int32_t newRepeat = repeatTimes;                                                 \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                       \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                           \
        __ubuf__ typeName *newSrc = src;                                                 \
        for (uint16_t i = 0; i < 1; i++) {                                               \
            MaskReg pgAll = CreatePredicate<typeName>();                                 \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                \
            RegTensor<typeName> vreg1;                                                   \
            Duplicate<typeName, typeName>(vreg1, initVal);                               \
            MaskReg preg = MovePredicate<typeName>();                                    \
            for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {                         \
                RegTensor<typeName> vreg0;                                               \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(                      \
                    vreg0, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, srcRepStride, preg);    \
                opName<typeName, Mode::MERGING>(vreg1, vreg0, vreg1, preg);              \
            }                                                                            \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, vreg1, offset0, pgAll); \
        }                                                                                \
    }

// L0 REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL
#define REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal) \
    template <typename T = typeName>                                                                       \
    __aicore__ inline void reduceFuncName##Impl(__ubuf__ typeName *dst,                                    \
        __ubuf__ typeName *src,                                                                            \
        __ubuf__ typeName *work,                                                                           \
        const uint64_t mask[2],                                                                            \
        const int32_t repeatTimes,                                                                         \
        const int32_t srcRepStride)                                                                        \
    {                                                                                                      \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                              \
        REDUCE_TO_VEC_BIT_BY_BIT_MODE(instrName, type, initVal, typeName);                                 \
        VEC_REDUCE_IMPL(Reduce##instrName, dst, work, type, elementNumPerInstr, typeName);                 \
    }

template <typename T>
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const uint64_t mask[2],
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, int16_t, s16, 16, INT16_MIN)

// level 0 bit by bit with index
#define REDUCE_TO_VEC_WITH_INDEX_BIT_BY_BITS_MODE(                                                   \
    opName, cmpOp, type, offsetNum, initVal, typeName, indexType, initIndexType)                     \
    SetVectorMask<typeName>(mask[1], mask[0]);                                                       \
    __VEC_SCOPE__                                                                                    \
    {                                                                                                \
        int32_t newRepeat = repeatTimes;                                                             \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                                   \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                                       \
        __ubuf__ typeName *newSrc = src;                                                             \
        for (uint16_t i = 0; i < 1; i++) {                                                           \
            MaskReg pgAll = CreatePredicate<typeName>();                                             \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                            \
            RegTensor<indexType> dstIndexVreg;                                                       \
            RegTensor<indexType> indexVreg;                                                          \
            RegTensor<initIndexType> initIndexVreg;                                                  \
            RegTensor<typeName> dstVreg;                                                             \
            Duplicate<indexType, indexType>(dstIndexVreg, 0);                                        \
            CreateVecIndex<initIndexType, Order::INC_ORDER_VALUE, indexType>(initIndexVreg, 0);      \
            Duplicate<typeName, typeName>(dstVreg, initVal);                                         \
            indexVreg = (RegTensor<indexType> &)initIndexVreg;                                       \
            MaskReg preg = MovePredicate<typeName>();                                                \
            for (uint16_t j = 0; j < (uint16_t)newRepeat; j++) {                                     \
                MaskReg pd;                                                                          \
                MaskReg indexPg;                                                                     \
                RegTensor<typeName> srcVreg;                                                         \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(                                  \
                    srcVreg, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, srcRepStride, preg);              \
                Compare<typeName, cmpOp>(indexPg, srcVreg, dstVreg, preg);                           \
                Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                   \
                opName<typeName, Mode::MERGING>(dstVreg, srcVreg, dstVreg, preg);                    \
                Adds<indexType, uint32_t, Mode::MERGING>(indexVreg, indexVreg, offsetNum, preg);     \
            }                                                                                        \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, dstVreg, offset0, pgAll);           \
            DataCopy<indexType, DistVST::DIST_NORM_B##type>(tempBuf, dstIndexVreg, offset0, pgAll);  \
        }                                                                                            \
    }

#define REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(typeName, indexType, initIndexType, type, initVal, indexInitVal)   \
    template <typename T = typeName>                                                                              \
    __aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ typeName *dst,                                         \
        __ubuf__ typeName *src,                                                                                   \
        __ubuf__ typeName *work,                                                                                  \
        const uint64_t mask[2],                                                                                   \
        const int32_t repeatTimes,                                                                                \
        const int32_t srcRepStride)                                                                               \
    {                                                                                                             \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                     \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                     \
        __ubuf__ indexType *tempBuf =                                                                             \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);                   \
        REDUCE_TO_VEC_WITH_INDEX_BIT_BY_BITS_MODE(Min, CMPMODE::LT, type, elementNumPerInstr, initVal, typeName, indexType, initIndexType) \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMin, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType)           \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                    \
    }

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work,
    const uint64_t mask[2], const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(half, uint16_t, int16_t, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(float, uint32_t, int32_t, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(uint32_t, uint32_t, int32_t, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(int32_t, uint32_t, int32_t, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(uint16_t, uint16_t, int16_t, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(int16_t, uint16_t, int16_t, 16, INT16_MAX, UINT16_MAX)

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue, uint32_t &maxMinIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue, T &maxMinIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline T GetAccValImpl()
{
    int64_t accVal = Internal::g_accVal;
    return *(reinterpret_cast<T*>(&accVal));
}

} // end namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H